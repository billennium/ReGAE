import numpy as np
import torch
from torch import Tensor


def remove_duplicates(graphs: list):
    indices = get_unique_indices(graphs)
    return [graphs[i] for i in indices]


def get_unique_indices(graphs: list) -> list[int]:
    hashes = [hash_graph(g) for g in graphs]
    _, indices = np.unique(hashes, return_index=True)
    return indices


def hash_graph(g):
    if isinstance(g, Tensor):
        g = g.numpy()
    return hash(g.tobytes())
