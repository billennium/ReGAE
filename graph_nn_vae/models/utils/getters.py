from typing import List

import torch
from torch import optim
from torch import nn

from graph_nn_vae.metrics.edge_accuracy import EdgeAccuracy
from graph_nn_vae.metrics.precision_recall import *


def get_loss(name: str, loss_weight: torch.Tensor = None):
    losses = {
        "MSE": torch.nn.MSELoss,
        "BCE": torch.nn.BCELoss,
    }
    if loss_weight is not None:
        return losses[name](weight=loss_weight)
    else:
        return losses[name]()


def get_optimizer(name: str):
    optimizers = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "AdamWAmsgrad": AdamWAmsgrad,
        "SGD": optim.SGD,
    }
    return optimizers[name]


def AdamWAmsgrad(*args, **kwargs):
    return optim.AdamW(amsgrad=True, *args, **kwargs)


def get_lr_scheduler(name: str):
    optimizers = {
        "NoSched": NoSched,
        "StepLR": torch.optim.lr_scheduler.StepLR,
        "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
        "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    }
    return optimizers.get(name, NoSched)


class NoSched(torch.optim.lr_scheduler._LRScheduler):
    """
    Does not decay the lr whatsoever.
    """

    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        super(NoSched, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return self.base_lrs


def get_metrics(metrics: List[str]):
    metrics_dict = {
        "Accuracy": EdgeAccuracy,
        "PositivePrecision": PositivePrecision,
        "PositiveRecall": PositiveRecall,
        "NegativePrecision": NegativePrecision,
        "NegativeRecall": NegativeRecall,
    }
    return [metrics_dict[m]() for m in metrics]


def get_activation_function(name: str = "ReLU"):
    d = {
        "ReLU": nn.ReLU,
        "ELU": nn.ELU,
        "CELU": nn.CELU,
    }
    return d[name]
