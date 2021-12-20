from . import random_permute

import numpy as np
import torch
from torch import Tensor
import networkx as nx


def minimize_adj_matrix(adj_matrix: np.ndarray, target_num_nodes: int) -> Tensor:
    adj_matrix = np.tril(adj_matrix)
    torch_matrix = torch.Tensor(adj_matrix)
    if len(torch_matrix.shape) == 2:
        torch_matrix = torch_matrix[:, :, None]
    return torch_matrix
