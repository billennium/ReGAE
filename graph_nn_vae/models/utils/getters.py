from typing import List

import torch
from torch import optim

from graph_nn_vae.metrics.edge_accuracy import EdgeAccuracy


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
    optimizers = {"Adam": optim.Adam, "SGD": optim.SGD}
    return optimizers[name]


def get_metrics(metrics: List[str]):
    metrics_dict = {
        "Accuracy": EdgeAccuracy,
    }
    return [metrics_dict[m]() for m in metrics]
