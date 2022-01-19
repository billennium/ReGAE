import torch
from torch import Tensor


def to_sparse_if_not(t: Tensor) -> Tensor:
    if t.layout == torch.strided:
        return t.to_sparse()
    return t


def to_dense_if_not(t: Tensor) -> Tensor:
    if t.layout == torch.sparse_coo:
        return t.to_dense()
    return t
