from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

# flow_matching
from flow_matching.path.scheduler import (
    Scheduler,
    CondOTScheduler,
    PolynomialConvexScheduler,
)
from flow_matching.path import AffineProbPath
from torchdiffeq import odeint

from src.models.components.bregman_divergence import BregmanDivergence

class ImageDiffusionLitModule(LightningModule):
    """Example of a `LightningModule` for Image Diffusion.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        noise_scheduler: Scheduler,
        bregman_divergence: BregmanDivergence,
        objective: str, # 'velocity', 'denoiser'
        num_sampling_steps: int,
        inference_batch_size: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `DiffusionLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param lr_scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # model to use (UNet, MLP, RIN, etc)
        self.net = net
        # noise scheduler
        self.noise_scheduler = noise_scheduler
        # objective to optimize (x1, velocity, x0, etc)
        self.objective = objective
        # loss function
        self.criterion = bregman_divergence
        # for averaging loss across batches
        self.train_loss = MeanMetric()

        # Inference time parameters.
        self.num_samples_to_generate = inference_batch_size
        self.num_sampling_steps = num_sampling_steps

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.net.train()
        

    def training_step(
        self,
        batch: torch.Tensor,
    ):
        # TODO pre-condition batch if necessary

        # This is unguided diffusion, get rid of the conditioning (second element in tuple).
        batch = batch[0] if isinstance(batch, (list, tuple)) else batch

        t = torch.rand((batch.shape[0],), device=batch.device)
        t = torch.distributions.Beta(
            torch.tensor(4., device=batch.device),
            torch.tensor(2., device=batch.device)
            ).sample(t.shape).to(batch.device)
        noise = torch.randn_like(batch)

        probability_path = AffineProbPath(scheduler=self.noise_scheduler)

        # x_0 is noise x_1 is data, t goes from 0 (noise) to 1 (data)
        path_sample = probability_path.sample(
            x_0=noise,
            x_1=batch,
            t=t,
        )
        
        # truth = target.
        if self.objective == 'velocity':
            truth = path_sample.dx_t
        elif self.objective == 'x1':
            truth = batch
        elif self.objective == 'x0':
            truth = path_sample.x_0
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        # estimate = model output
        estimate = self.net(path_sample.x_t, t)
        loss = self.criterion(truth, estimate)

        # average loss across batches
        self.train_loss(loss)
        # log the loss
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def ode_sample_rin(
        self,
        num_samples: int,
        num_steps: int,
        batch_shape: Tuple[int, ...],
    ):
        """
        Sample from the RIN model using ODE solver.
        :param num_samples: The number of samples to generate.
        :param batch_shape: The shape of the batch to generate.
        :return: A tensor of shape (num_samples, *batch_shape).
        """
        assert self.objective == 'velocity', "ODE sampling is only supported for velocity objective."

        # Make sure the latent is reset.
        self.net.current_latents = None

        # Starting noise.
        x_init = torch.randn(num_samples, *batch_shape, device=self.device)
        # Time grid for the ODE solver.
        time_grid = torch.linspace(0, 1, num_steps, device=self.device)

        assert not self.training, "ODE sampling should not be done in training mode, as latents are reset every forward pass."

        def ode_func(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return self.net(x, t)
        
        # Set torch.no_grad() just to make sure, it should be set upstream by PyTorch Lightning.
        with torch.no_grad():
            # Solve using super fast torchdiffeq solver.
            solution = odeint(
                func=ode_func,
                y0=x_init,
                t=time_grid,
                method="midpoint",
            )
        
        # This saves all noisy intermediates, only keep the fully denoised sample.
        solution = solution[-1]
        assert solution.shape == (num_samples, *batch_shape), f"Expected solution shape {(num_samples, *batch_shape)}, got {solution.shape}"

        # Log the generated images, one at a time.
        # Probably only works with wandb.
        self.logger.log_image(key="samples", images=[solution[i] for i in range(num_samples)])


    def validation_step(self, batch, batch_idx):
        # batch_shape should be of the form (batch_size, channels, height, width)
        batch_shape = batch[0].shape if isinstance(batch, (list, tuple)) else batch.shape
        self.ode_sample_rin(
            num_samples=1,
            num_steps=self.num_sampling_steps,
            batch_shape=batch_shape[1:],    # Only send in the shape of the image, not the batch size, as that changes for inference time.
        )
        pass

    def test_step(self, batch, batch_idx):
        """
        I will use this to generate samples from the model at test time.
        The input batch is a tuple of (data, guiding_variable). For guided (or conditional) diffusion, guiding_variable is used to generate samples.
        The generated samples can be compared against data with some metric (RMSE reconstruction error, etc).
        We are only doing unguided diffusion for now, and so batch will not be used at all.
        The generated images will be logged.
        :param batch: The input batch, which is a tuple of (data, guiding_variable).
        :param batch_idx: The index of the batch.
        return: None
        """
        # batch_shape should be of the form (batch_size, channels, height, width)
        batch_shape = batch[0].shape if isinstance(batch, (list, tuple)) else batch.shape
        self.ode_sample_rin(
            num_samples=self.num_samples_to_generate,
            num_steps=self.num_sampling_steps,
            batch_shape=batch_shape[1:],    # Only send in the shape of the image, not the batch size, as that changes for inference time.
        )

    def predict_step(self, batch, batch_idx):
        # ?
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.net.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}