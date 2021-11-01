import numpy as np
import numpy.typing as npt
import networkx as nx


def random_permute(adjacency_matrix: np.ndarray) -> np.ndarray:
    x_idx = np.random.permutation(adjacency_matrix.shape[0])
    adjacency_matrix = adjacency_matrix[np.ix_(x_idx, x_idx)]
    return adjacency_matrix
