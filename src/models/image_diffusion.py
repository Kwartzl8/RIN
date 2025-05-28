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
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

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

        # This is unguided diffusion, get rid of the condition (second in tuple).
        batch = batch[0] if isinstance(batch, (list, tuple)) else batch

        t = torch.rand((batch.shape[0],), device=batch.device)
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
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Only makes sense if we have a metric to validate against. FID works with unguided diffusion. FWD only works with guided diffusion.
        pass

    def test_step(self, batch, batch_idx):
        # Same as validation step.
        pass
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