import numpy as np
from typing import List

from .bfs import bfs_ordering
from .remove_duplicates import remove_duplicates


def random_permute(adjacency_matrix: np.ndarray) -> np.ndarray:
    x_idx = np.random.permutation(adjacency_matrix.shape[0])
    adjacency_matrix = adjacency_matrix[np.ix_(x_idx, x_idx)]
    return adjacency_matrix


def permute_unique_bfs(matrix: np.ndarray, num_permutations: int) -> List[np.ndarray]:
    matrices = [matrix]
    for _ in range(num_permutations - 1):
        matrices.append(random_permute(matrix))
    matrices = [bfs_ordering(m) for m in matrices]
    matrices = remove_duplicates(matrices)
    return matrices
