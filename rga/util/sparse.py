from attr import has
import torch
import numpy as np
from torch import Tensor


def to_sparse_if_not(t: Tensor) -> Tensor:
    if isinstance(t, (np.ndarray, np.generic)):
        return t
    if t.layout == torch.strided:
        return t.to_sparse()
    return t


def to_dense_if_not(t: Tensor) -> Tensor:
    if isinstance(t, (np.ndarray, np.generic)):
        return t
    if t.layout == torch.sparse_coo:
        if not t.is_cuda and t.dtype == torch.float16:
            t = t.float()
        return t.to_dense()
    return t
