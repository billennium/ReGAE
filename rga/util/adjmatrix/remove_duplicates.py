import numpy as np
from scipy import sparse
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
    if isinstance(g, sparse.csr_matrix):
        g = g.todense()
    if isinstance(g, sparse.coo_matrix):
        g = g.todense()
    return hash(g.tobytes())
