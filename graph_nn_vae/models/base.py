from abc import ABCMeta
from argparse import ArgumentParser
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch import Tensor, nn
import pytorch_lightning as pl
from graph_nn_vae.models.utils.getters import (
    get_metrics,
    get_loss,
    get_optimizer,
    get_lr_scheduler,
)


class BaseModel(pl.LightningModule, metaclass=ABCMeta):
    model_name = ""

    def __init__(
        self,
        loss_function: str,
        loss_weight: torch.Tensor,
        learning_rate: float,
        optimizer: str,
        weight_decay: float,
        lr_scheduler_name: str,
        lr_scheduler_params: dict,
        lr_scheduler_metric: str,
        metrics: List[str],
        metric_update_interval: int = 1,
        data_module=None,  # only for returning test_datamodule()
        **kwargs,
    ):
        super(BaseModel, self).__init__()
        self.loss_function = get_loss(loss_function, loss_weight)
        self.learning_rate = learning_rate
        self.optimizer = get_optimizer(optimizer)
        self.weight_decay = weight_decay
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_scheduler_params = lr_scheduler_params
        self.lr_scheduler_metric = lr_scheduler_metric
        self.metrics_train = nn.ModuleList(get_metrics(metrics))
        self.metrics_val = nn.ModuleList(get_metrics(metrics))
        self.metrics_test = nn.ModuleList(get_metrics(metrics))
        self.metric_update_interval = metric_update_interval
        self.metric_update_counter = 0
        self.data_module = data_module

    def forward(self, **kwargs) -> Tensor:
        raise NotImplementedError

    def adjust_y_to_prediction(self, batch, y_predicted) -> Tuple[Tensor, Tensor]:
        """
        Returns Tuple[ground_truth_y, y_predicted) after adjustments to their shape for loss calculation.
        Overload this function if this kind of adjustment is needed.
        """
        return batch, y_predicted

    def step(self, batch, metrics: List[Callable] = []) -> Tensor:
        y_predicted = self(batch)
        y, y_predicted = self.adjust_y_to_prediction(batch, y_predicted)
        loss = self.loss_function(y_predicted, y)

        for metric in metrics:
            metric(y_predicted, y)

        return loss

    def training_step(self, batch, batch_idx):
        should_update_metric = (
            self.metric_update_counter % self.metric_update_interval == 0
        )
        self.metric_update_counter += 1
        metrics = self.metrics_train if should_update_metric else []

        loss = self.step(batch, metrics)
        for metric in metrics:
            self.log(f"{metric.label}/train_avg", metric, on_step=False, on_epoch=True)
        self.log("loss/train_avg", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, self.metrics_val)
        for metric in self.metrics_val:
            self.log(f"{metric.label}/val", metric, prog_bar=True)
        self.log("loss/val", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, self.metrics_test)
        for metric in self.metrics_test:
            self.log(f"{metric.label}/test", metric, prog_bar=True)
        self.log("loss/test", loss)
        return loss

    def test_dataloader(self):
        # For some reason newer versions of Lightning require the model to define it's own
        # test dataloader. We consider that to be the responsibilty of the DataModule.
        return self.data_module.test_dataloader()

    def on_fit_end(self) -> None:
        if isinstance(self.logger, pl.loggers.TensorBoardLogger):
            # TensorBoardLogger does not always flush the logs.
            # To ensure this, we run it manually
            self.logger.experiment.flush()

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = get_lr_scheduler(self.lr_scheduler_name)(
            optimizer=optimizer, **self.lr_scheduler_params
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": self.lr_scheduler_metric,
        }

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
            "--learning_rate",
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
            "--metrics",
            default=[],
            nargs="+",
            type=str,
            metavar="METRIC",
            help="list of names of metrics to be logged",
        )
        parser.add_argument(
            "--metric_update_interval",
            dest="metric_update_interval",
            default=1,
            type=int,
            help="Every how many steps to update the training metrics (other than loss). \
                Higher values decrease computation costs related to metric updating at the expense of precision.",
        )
        parser.add_argument(
            "--lr_scheduler_name",
            dest="lr_scheduler_name",
            default="ReduceLROnPlateau",
            type=str,
            help="name of learning rate scheduler",
        )
        parser.add_argument(
            "--lr_scheduler_params",
            dest="lr_scheduler_params",
            default={},
            type=str,
            help="params for learning rate scheduler, only when lr_schduler is set",
        )
        parser.add_argument(
            "--lr_scheduler_metric",
            dest="lr_scheduler_metric",
            default="loss/train_avg",
            type=str,
            help="metric to monitor for learning rate scheduler, only when lr_schduler is set",
        )
        return parser
