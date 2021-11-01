import torch
from torch import nn, optim


def get_loss(name: str, loss_weight: torch.Tensor = None):
    losses = {
        "MSE": nn.MSELoss,
        "BCE": nn.BCELoss,
    }
    if loss_weight is not None:
        return losses[name](weight=loss_weight)
    else:
        return losses[name]()


def get_optimizer(name: str):
    optimizers = {"Adam": optim.Adam, "SGD": optim.SGD}
    return optimizers[name]
