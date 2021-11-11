from typing import List
import torch
from torch import nn, optim
from torch.functional import Tensor


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


def weighted_average(v1: Tensor, v2: Tensor, weight: Tensor) -> Tensor:
    """
    Weighted average of tensors `v1`, `v2` with weigth `w` is defined as:
    avg = v1 * sig(w) + v2 * (1 - sig(w))

    All Tensors have to have the same dimensions.
    """
    weight = torch.sigmoid(weight)
    ones = torch.ones(v1.shape, device=v1.device, requires_grad=v1.requires_grad)
    return v1 * weight + (ones - weight) * v2


def sequential_from_layer_sizes(
    input_size: int, output_size: int, hidden_sizes: List[int]
) -> nn.Sequential:
    layer_sizes = hidden_sizes + [output_size]
    layers = [nn.Linear(input_size, layer_sizes[0])]
    for i in range(len(layer_sizes)):
        if i == len(layer_sizes) - 1:
            break
        layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
    return nn.Sequential(*layers)
