import numpy as np
from scipy import sparse
from torch import Tensor
from typing import List


def remove_duplicates(graphs: List):
    indices = get_unique_indices(graphs)
    return [graphs[i] for i in indices]


def get_unique_indices(graphs: List) -> List[int]:
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
