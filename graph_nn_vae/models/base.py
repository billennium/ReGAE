from abc import ABCMeta
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor, nn
import pytorch_lightning as pl

from graph_nn_vae.models.utils import get_loss, get_optimizer


class BaseModel(pl.LightningModule, metaclass=ABCMeta):
    model_name = ""

    def __init__(
        self,
        loss_function: str,
        loss_weight: torch.Tensor,
        learning_rate: float,
        optimizer: str,
        weight_decay: float,
        scheduler_gamma: float,
        **kwargs,
    ):
        super(BaseModel, self).__init__()
        self.loss_function = get_loss(loss_function, loss_weight)
        self.learning_rate = learning_rate
        self.optimizer = get_optimizer(optimizer)
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma

        # self.min_loss = MinimumSaver()

    def forward(self, **kwargs) -> Tensor:
        raise NotImplementedError

    def step(self, batch) -> Tensor:
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        # for metric in self.metrics:
        #     self.log(f"{metric.label}/train_avg", metric, on_step=False, on_epoch=True)
        self.log("loss/train_avg", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        # for metric in self.metrics:
        #     self.log(f"{metric.label}/val", metric, prog_bar=True)
        self.log("loss/val", loss, prog_bar=True)
        # self.min_loss.log("loss/val_min", loss.item(), batch[0].shape[0])
        return loss

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        pass
        # self.min_loss.calculate("loss/val_min")
        # self.log(
        #     "loss/val_min",
        #     # self.min_loss.get_min()["loss/val_min"],
        #     prog_bar=True,
        #     logger=False,
        # )

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        # for metric in self.metrics:
        #     self.log(f"{metric.label}/test", metric, prog_bar=True)
        self.log("loss/test", loss)
        return loss

    def on_fit_end(self) -> None:
        if isinstance(self.logger, pl.loggers.TensorBoardLogger):
            # TensorBoardLogger does not always flush the logs.
            # To ensure this, we run it manually
            self.logger.experiment.flush()

    def on_test_epoch_end(self):
        for key, value in self.min_loss.get_min().items():
            self.logger.log_hyperparams({key: value})

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 1, gamma=self.scheduler_gamma
        )
        return [optimizer], [scheduler]

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--loss-function",
            dest="loss_function",
            default="MSE",
            type=str,
            metavar="LOSS",
            help="name of loss function",
        )
        parser.add_argument(
            "--lr",
            "--learning-rate",
            dest="learning_rate",
            default=0.01,
            type=float,
            metavar="LR",
            help="initial learning rate",
        )
        parser.add_argument(
            "--optimizer",
            dest="optimizer",
            default="Adam",
            type=str,
            metavar="OPT",
            help="name of optimizer",
        )
        parser.add_argument(
            "--weight-decay",
            dest="weight_decay",
            default=0.0,
            type=float,
            metavar="FLOAT",
            help="weight decay",
        )
        parser.add_argument(
            "--scheduler-gamma",
            dest="scheduler_gamma",
            default=1.0,
            type=float,
            metavar="FLOAT",
            help="scheduler gamma",
        )
        return parser
