import pytest

import torch
from graph_nn_vae.util.adjmatrix.diagonal_block_representation import (
    adj_matrix_to_diagonal_block_representation,
)


@pytest.mark.parametrize(
    "matrix,num_nodes,block_size,expected",
    [
        # fmt: off
        (
            [[0, 0],
             [1, 0]],
            2,
            1,
            [[[1]]]
        ),
        (
            [[0, 0],
             [1, 0]],
            2,
            2,
            [[[0, 0],
              [1, 0]]]
        ),
        (
            [[0, 0],
             [1, 0]],
            2,
            3,
            [[[0, 0, 0],
              [0, 0, 0],
              [1, 0, 0]]]
        ),
        (
            [[0, 0, 0],
             [0, 0, 0],
             [1, 1, 0]],
            3,
            3,
            [[[0, 0, 0],
              [0, 0, 0],
              [1, 1, 0]]],
        ),
        (
            [[0, 0, 0],
             [0, 0, 0],
             [1, 1, 0]],
            3,
            2,
            [[[0, 0],
              [1, 1]]],
        ),
        (
            [[0, 0, 0, 0],
             [1, 0, 0, 0],
             [1, 0, 0, 0],
             [0, 1, 1, 0]],
            4,
            2,
            [[[1, 0],
              [0, 1]],
             [[0, 0],
              [1, 0]],
             [[0, 0],
              [1, 0]]],
        ),
        (
            [[0, 0, 0, 0],
             [1, 0, 0, 0],
             [1, 0, 0, 0],
             [0, 1, 1, 0]],
            4,
            3,
            [[[1, 0, 0],
              [1, 0, 0],
              [0, 1, 1]]],
        ),
        (
            [[0, 0, 0, 0],
             [1, 0, 0, 0],
             [1, 0, 0, 0],
             [0, 1, 1, 0]],
            4,
            1,
            [[[0]], [[1]], [[1]], [[1]], [[0]], [[1]]],
        ),
        # fmt: on
    ],
)
def test_adj_matrix_to_diagonal_block_representation(
    matrix, num_nodes, block_size, expected
):
    input_matrix = torch.Tensor(matrix)
    input_matrix = input_matrix[:, :, None]
    expected = torch.Tensor(expected)
    expected = expected[:, :, :, None]
    output = adj_matrix_to_diagonal_block_representation(
        input_matrix, num_nodes, block_size
    )
    assert torch.equal(output, expected)
