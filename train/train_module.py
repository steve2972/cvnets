import torch
from torch import Tensor
import lightning

from utils.optim import get_optimizer, get_scheduler
from utils import accuracy


class TrainModule(lightning.LightningModule):
    def __init__(self,
        model,
        optimizer:str="sgd",
        learning_rate:float=1e-3,
        scheduler:str="steplr",
        num_gpus:int=4,
    ):
        super().__init__()
        self.sync_dist = num_gpus > 1
        self.optimizer = optimizer
        self.lr = learning_rate
        self.scheduler = scheduler
        self.model = model


    def forward(self, x:Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(), opt=self.optimizer, lr=self.lr)
        lr_scheduler = get_scheduler(optimizer, lr_scheduler=self.scheduler)

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx) -> Tensor:
        x, y = batch
        y_hat = self(x)
        scheduler = self.lr_schedulers()
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log(
            "train_loss", loss, 
            prog_bar=True,
            sync_dist=self.sync_dist
        )

        if self.trainer.is_last_batch:
            scheduler.step()
        return loss

    def eval_step(self, batch, batch_idx, prefix: str="test"):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
        self.log(f"{prefix}_loss", loss, on_epoch=True, sync_dist=self.sync_dist)
        self.log(f"{prefix}_accuracy", {"acc1":acc1, "acc5":acc5}, on_epoch=True, sync_dist=self.sync_dist)

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")