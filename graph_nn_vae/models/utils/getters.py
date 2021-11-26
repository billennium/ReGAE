from typing import List

import torch
from torch import optim

from graph_nn_vae.metrics.edge_accuracy import EdgeAccuracy, MaskAccuracy
from graph_nn_vae.metrics.precision_recall import *


def get_loss(name: str, loss_weight: torch.Tensor = None):
    losses = {
        "MSE": torch.nn.MSELoss,
        "BCE": torch.nn.BCELoss,
        "BCEWithLogits": torch.nn.BCEWithLogitsLoss,
    }
    if loss_weight is not None:
        return losses[name](weight=loss_weight)
    else:
        return losses[name]()


def get_optimizer(name: str):
    optimizers = {"Adam": optim.Adam, "SGD": optim.SGD}
    return optimizers[name]


def get_lr_scheduler(name: str):
    optimizers = {
        "StepLR": torch.optim.lr_scheduler.StepLR,
        "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau
    }
    return optimizers[name]


def get_metrics(metrics: List[str]):
    metrics_dict = {
        "EdgeAccuracy": EdgeAccuracy,
        "MaskAccuracy": MaskAccuracy,
        "EdgePrecision": EdgePrecision,
        "EdgeRecall": EdgeRecall,
        "MaskPrecision": MaskPrecision,
        "MaskRecall": MaskRecall,
    }
    return [metrics_dict[m]() for m in metrics]
