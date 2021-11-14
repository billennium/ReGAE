import torch
from torch.functional import Tensor



def weighted_average(v1: Tensor, v2: Tensor, weight: Tensor) -> Tensor:
    """
    Weighted average of tensors `v1`, `v2` with weigth `w` is defined as:
    avg = v1 * sig(w) + v2 * (1 - sig(w))

    All Tensors have to have the same dimensions.
    """
    weight = torch.sigmoid(weight)
    ones = torch.ones(v1.shape, device=v1.device, requires_grad=v1.requires_grad)
    return v1 * weight + (ones - weight) * v2