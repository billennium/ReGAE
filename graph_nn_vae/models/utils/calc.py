import torch
from torch.functional import Tensor


def weighted_average(v1: Tensor, v2: Tensor, weight: Tensor) -> Tensor:
    """
    Weighted average of tensors `v1`, `v2` with weigth `w` is defined as:
    avg = v1 * sig(w) + v2 * (1 - sig(w))

    All Tensors have to have the same dimensions.
    """
    weight = torch.sigmoid(weight)
    # ones = torch.ones(v1.shape, device=v1.device, requires_grad=v1.requires_grad)
    return v1 * weight + (1 - weight) * v2


def torch_bincount(t: Tensor) -> Tensor:
    """
    torch.bincount() when used on CUDA may lead to nondeterministic gradients. From testing, this isn't an issue in our use case.
    However, moving the tensor to the CPU isn't slower on the tested machines, so this method was used instead.
    """
    # was_deterministic = torch.are_deterministic_algorithms_enabled()
    # torch.use_deterministic_algorithms(False)
    # t = torch.bincount(t)
    # torch.use_deterministic_algorithms(was_deterministic)
    # return t

    original_device = t.device
    t = t.cpu()
    t = torch.bincount(t)
    return t.to(original_device)
