import pytest

import numpy as np
import torch
from graph_nn_vae.util.adjmatrix.diagonal_representation import (
    adj_matrix_to_diagonal_representation,
)

from graph_nn_vae.util.adjmatrix.pad import minimize_and_pad


@pytest.mark.parametrize(
    "matrix,num_nodes,max_num_nodes_padding,expected",
    [
        # ([[0]], 1, None, [0]), one node is not compatible as is's not really a graph
        ([[0, 0], [1, 0]], 2, None, [1]),
        ([[0, 0], [1, 0]], 2, 3, [1, -1, -1]),
        ([[0, 0, 0], [0, 0, 0], [1, 1, 0]], 3, 3, [1, 0, 1]),
        (
            [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0]],
            4,
            4,
            [0, 1, 1, 1, 0, 1],
        ),
    ],
)
def test_adj_matrix_to_diagonal_representation(
    matrix, num_nodes, max_num_nodes_padding, expected
):
    input_matrix = torch.Tensor(matrix)
    input_matrix = input_matrix[:, :, None]
    expected = torch.Tensor(expected)
    expected = expected[:, None]
    output = adj_matrix_to_diagonal_representation(
        input_matrix, num_nodes, max_num_nodes_padding, -1.0
    )
    assert torch.equal(output, expected)
