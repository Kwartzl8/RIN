import math
from pathlib import Path
from random import random
from functools import partial
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
from torch.special import expm1
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torchvision import transforms as T, utils

from beartype import beartype

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from rin_pytorch.attend import Attend

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator, DistributedDataParallelKwargs

# helpers functions

def exists(x):
    return x is not None

def identity(x):
    return x

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def safe_div(numer, denom, eps = 1e-10):
    return numer / denom.clamp(min = eps)

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    num_sqrt = math.sqrt(num)
    return int(num_sqrt) == num_sqrt

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

# use layernorm without bias, more stable

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class MultiHeadedRMSNorm(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# positional embeds

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        norm = False,
        qk_norm = False,
        time_cond_dim = None
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.time_cond = None

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        self.norm = LayerNorm(dim) if norm else nn.Identity()

        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = MultiHeadedRMSNorm(dim_head, heads)
            self.k_norm = MultiHeadedRMSNorm(dim_head, heads)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(
        self,
        x,
        time = None
    ):
        h = self.heads
        x = self.norm(x)

        if exists(self.time_cond):
            assert exists(time)
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = torch.einsum('b h n d, b h n e -> b h d e', k, v)

        out = torch.einsum('b h d e, b h n d -> b h n e', context, q)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context = None,
        heads = 4,
        dim_head = 32,
        norm = False,
        norm_context = False,
        time_cond_dim = None,
        flash = False,
        qk_norm = False
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        self.time_cond = None

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.norm = LayerNorm(dim) if norm else nn.Identity()
        self.norm_context = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, hidden_dim * 2, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = MultiHeadedRMSNorm(dim_head, heads)
            self.k_norm = MultiHeadedRMSNorm(dim_head, heads)

        self.attend = Attend(flash = flash)

    def forward(
        self,
        x,
        context = None,
        time = None
    ):
        h = self.heads

        if exists(context):
            context = self.norm_context(context)

        x = self.norm(x)

        context = default(context, x)

        if exists(self.time_cond):
            assert exists(time)
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PEG(nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.ds_conv = nn.Conv2d(dim, dim, 3, padding = 1, groups = dim)

    def forward(self, x):
        b, n, d = x.shape
        hw = int(math.sqrt(n))
        x = rearrange(x, 'b (h w) d -> b d h w', h = hw)
        x = self.ds_conv(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, time_cond_dim = None):
        super().__init__()
        self.norm = LayerNorm(dim)

        self.time_cond = None

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x, time = None):
        x = self.norm(x)

        if exists(self.time_cond):
            assert exists(time)
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        return self.net(x)

# model

class RINBlock(nn.Module):
    def __init__(
        self,
        dim,
        latent_self_attn_depth,
        dim_latent = None,
        final_norm = True,
        patches_self_attn = True,
        **attn_kwargs
    ):
        super().__init__()
        dim_latent = default(dim_latent, dim)

        self.latents_attend_to_patches = Attention(dim_latent, dim_context = dim, norm = True, norm_context = True, **attn_kwargs)
        self.latents_cross_attn_ff = FeedForward(dim_latent)

        self.latent_self_attns = nn.ModuleList([])
        for _ in range(latent_self_attn_depth):
            self.latent_self_attns.append(nn.ModuleList([
                Attention(dim_latent, norm = True, **attn_kwargs),
                FeedForward(dim_latent)
            ]))

        self.latent_final_norm = LayerNorm(dim_latent) if final_norm else nn.Identity()

        self.patches_peg = PEG(dim)
        self.patches_self_attn = patches_self_attn

        if patches_self_attn:
            self.patches_self_attn = LinearAttention(dim, norm = True, **attn_kwargs)
            self.patches_self_attn_ff = FeedForward(dim)

        self.patches_attend_to_latents = Attention(dim, dim_context = dim_latent, norm = True, norm_context = True, **attn_kwargs)
        self.patches_cross_attn_ff = FeedForward(dim)

    def forward(self, patches, latents, t):
        patches = self.patches_peg(patches) + patches

        # latents extract or cluster information from the patches

        latents = self.latents_attend_to_patches(latents, patches, time = t) + latents

        latents = self.latents_cross_attn_ff(latents, time = t) + latents

        # latent self attention

        for attn, ff in self.latent_self_attns:
            latents = attn(latents, time = t) + latents
            latents = ff(latents, time = t) + latents

        if self.patches_self_attn:
            # additional patches self attention with linear attention

            patches = self.patches_self_attn(patches, time = t) + patches
            patches = self.patches_self_attn_ff(patches) + patches

        # patches attend to the latents

        patches = self.patches_attend_to_latents(patches, latents, time = t) + patches

        patches = self.patches_cross_attn_ff(patches, time = t) + patches

        latents = self.latent_final_norm(latents)
        return patches, latents

class RIN(nn.Module):
    def __init__(
        self,
        dim,
        image_size,
        patch_size = 16,
        channels = 3,
        depth = 6,                      # number of RIN blocks
        latent_self_attn_depth = 2,     # how many self attentions for the latent per each round of cross attending from pixel space to latents and back
        dim_latent = None,              # will default to image dim (dim)
        num_latents = 256,              # they still had to use a fair amount of latents for good results (256), in line with the Perceiver line of papers from Deepmind
        learned_sinusoidal_dim = 16,
        self_conditioning_prob = 0.9,
        latent_token_time_cond = False, # whether to use 1 latent token as time conditioning, or do it the adaptive layernorm way (which is highly effective as shown by some other papers "Paella" - Dominic Rampas et al.)
        dual_patchnorm = True,
        patches_self_attn = True,       # the self attention in this repository is not strictly with the design proposed in the paper. offer way to remove it, in case it is the source of instability
        **attn_kwargs
    ):
        super().__init__()
        assert divisible_by(image_size, patch_size)
        dim_latent = default(dim_latent, dim)

        self.image_size = image_size
        self.channels = channels # times 2 due to self-conditioning

        patch_height_width = image_size // patch_size
        num_patches = patch_height_width ** 2
        pixel_patch_dim = channels * (patch_size ** 2)

        # time conditioning

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        time_dim = dim * 4
        fourier_dim = learned_sinusoidal_dim + 1

        self.train_prob_self_cond = self_conditioning_prob
        self.latent_token_time_cond = latent_token_time_cond
        time_output_dim = dim_latent if latent_token_time_cond else time_dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_output_dim)
        )

        # pixels to patch and back

        self.to_patches = Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(pixel_patch_dim * 2) if dual_patchnorm else None,
            nn.Linear(pixel_patch_dim * 2, dim),
            nn.LayerNorm(dim) if dual_patchnorm else None,
        )

        # axial positional embeddings, parameterized by an MLP

        pos_emb_dim = dim // 2

        self.axial_pos_emb_height_mlp = nn.Sequential(
            Rearrange('... -> ... 1'),
            nn.Linear(1, pos_emb_dim),
            nn.SiLU(),
            nn.Linear(pos_emb_dim, pos_emb_dim),
            nn.SiLU(),
            nn.Linear(pos_emb_dim, dim)
        )

        self.axial_pos_emb_width_mlp = nn.Sequential(
            Rearrange('... -> ... 1'),
            nn.Linear(1, pos_emb_dim),
            nn.SiLU(),
            nn.Linear(pos_emb_dim, pos_emb_dim),
            nn.SiLU(),
            nn.Linear(pos_emb_dim, dim)
        )

        # nn.Parameter(torch.randn(2, patch_height_width, dim) * 0.02)

        self.to_pixels = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, pixel_patch_dim),
            Rearrange('b (h w) (c p1 p2) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size, h = patch_height_width)
        )

        self.latents = nn.Parameter(torch.randn(num_latents, dim_latent))
        nn.init.normal_(self.latents, std = 0.02)

        self.init_self_cond_latents = nn.Sequential(
            FeedForward(dim_latent),
            LayerNorm(dim_latent)
        )

        nn.init.zeros_(self.init_self_cond_latents[-1].gamma)

        # the main RIN body parameters  - another attention is all you need moment

        if not latent_token_time_cond:
            attn_kwargs = {**attn_kwargs, 'time_cond_dim': time_dim}

        self.blocks = nn.ModuleList([RINBlock(dim, dim_latent = dim_latent, latent_self_attn_depth = latent_self_attn_depth, patches_self_attn = patches_self_attn, **attn_kwargs) for _ in range(depth)])

    @property
    def device(self):
        return next(self.parameters()).device

    def model_step(
        self,
        x,
        time,
        x_self_cond = None,
        latent_self_cond = None,
        return_latents = False
    ):
        batch = x.shape[0]

        x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))

        x = torch.cat((x_self_cond, x), dim = 1)

        # prepare time conditioning

        t = self.time_mlp(time)

        # prepare latents

        latents = repeat(self.latents, 'n d -> b n d', b = batch)

        # the warm starting of latents as in the paper

        if exists(latent_self_cond):
            latents = latents + self.init_self_cond_latents(latent_self_cond)

        # whether the time conditioning is to be treated as one latent token or for projecting into scale and shift for adaptive layernorm

        if self.latent_token_time_cond:
            t = rearrange(t, 'b d -> b 1 d')
            latents = torch.cat((latents, t), dim = -2)

        # to patches
        
        patches = self.to_patches(x)

        height_range = width_range = torch.linspace(0., 1., steps = int(math.sqrt(patches.shape[-2])), device = self.device)
        pos_emb_h, pos_emb_w = self.axial_pos_emb_height_mlp(height_range), self.axial_pos_emb_width_mlp(width_range)

        pos_emb = rearrange(pos_emb_h, 'i d -> i 1 d') + rearrange(pos_emb_w, 'j d -> 1 j d')
        patches = patches + rearrange(pos_emb, 'i j d -> (i j) d')

        # the recurrent interface network body

        for block in self.blocks:
            patches, latents = block(patches, latents, t)

        # to pixels

        pixels = self.to_pixels(patches)

        if not return_latents:
            return pixels

        # remove time conditioning token, if that is the settings

        if self.latent_token_time_cond:
            latents = latents[:, :-1]

        return pixels, latents

    def forward(
        self,
        x,
        time,
        x_self_cond = None,
        latent_self_cond = None,
        return_latents = False
    ):
        self_cond = self_latents = None
        if random() < self.train_prob_self_cond:
            with torch.no_grad():
                model_output, self_latents = self.model_step(x, time, return_latents=True)
                self_latents = self_latents.detach()
                # I removed velocity conversion for pixel self-conditioning. This is not even in the paper.
                self_cond = model_output

                # we are clamping to bring it back to image space, but this is not strictly necessary as it goes back to the model input
                # Not clamping has been shown to improve results.
                self_cond.clamp_(-1., 1.)
                self_cond = self_cond.detach()

        # predict and take gradient step

        estimate = self.model_step(x, time, self_cond, self_latents)
        return estimate