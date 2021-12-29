import pytest

import torch
from graph_nn_vae.util.adjmatrix.diagonal_block_representation import (
    adj_matrix_to_diagonal_block_representation,
    diagonal_block_to_adj_matrix_representation,
)


@pytest.mark.parametrize(
    "matrix,num_nodes,block_size,expected,pad_value",
    [
        # fmt: off
        (
            [[0, 1],
             [1, 0]],
            2,
            1,
            [[[1]]],
            0
        ),
        (
            [[0, 1],
             [1, 0]],
            2,
            2,
            [[[0, 0],
              [1, 0]]],
            0
        ),
        (
            [[0, 1],
             [1, 0]],
            2,
            3,
            [[[0, 0, 0],
              [0, 0, 0],
              [1, 0, 0]]],
            0
        ),
        (
            [[0, 1],
             [1, 0]],
            2,
            3,
            [[[-1, -1, -1],
              [-1, -1, -1],
              [1, -1, -1]]],
            -1
        ),
        (
            [[0, 0, 1],
             [0, 0, 1],
             [1, 1, 0]],
            3,
            3,
            [[[0, 0, 0],
              [0, 0, 0],
              [1, 1, 0]]],
            0
        ),
        (
            [[0, 0, 1],
             [0, 0, 1],
             [1, 1, 0]],
            3,
            3,
            [[[-1, -1, -1],
              [0, -1, -1],
              [1, 1, -1]]],
            -1
        ),
        (
            [[0, 0, 1],
             [0, 0, 1],
             [1, 1, 0]],
            3,
            2,
            [[[0, 0],
              [1, 1]]],
            0
        ),
        (
            [[0, 1, 1, 0],
             [1, 0, 0, 1],
             [1, 0, 0, 1],
             [0, 1, 1, 0]],
            4,
            2,
            [[[1, 0],
              [0, 1]],
             [[0, 0],
              [1, 0]],
             [[0, 0],
              [1, 0]]],
            0
        ),
        (
            [[0, 1, 1, 1],
             [1, 0, 0, 1],
             [1, 1, 0, 1],
             [1, 1, 1, 0]],
            4,
            2,
            [[[1, 1],
              [1, 1]],
             [[0, 0],
              [1, 0]],
             [[0, 0],
              [1, 0]]],
            0
        ),
        (
            [[0, 1, 1, 0],
             [1, 0, 0, 1],
             [1, 0, 0, 1],
             [0, 1, 1, 0]],
            4,
            2,
            [[[1, 0],
              [0, 1]],
             [[-1, -1],
              [1, -1]],
             [[-1, -1],
              [1, -1]]],
            -1
        ),
        (
            [[0, 1, 1, 0],
             [1, 0, 0, 1],
             [1, 0, 0, 1],
             [0, 1, 1, 0]],
            4,
            3,
            [[[1, 0, 0],
              [1, 0, 0],
              [0, 1, 1]]],
            0
        ),
        (
            [[0, 1, 1, 1 , 1],
             [1, 0, 1, 1, 1],
             [1, 1, 0, 1, 1],
             [1, 1, 1, 0, 1],
             [1, 1, 1, 1, 1]],
            5,
            3,
            [[[1, 1, 0],
              [1, 1, 1],
              [1, 1, 1]],
             [[0, 0, 0],
              [0, 0, 0],
              [1, 0, 0]],
             [[0, 0, 0],
              [0, 0, 0],
              [1, 0, 0]]],
            0
        ),
        (
            [[0, 1, 1, 0],
             [1, 0, 0, 1],
             [1, 0, 0, 1],
             [0, 1, 1, 0]],
            4,
            1,
            [[[0]], [[1]], [[1]], [[1]], [[0]], [[1]]],
            0
        ),
        # fmt: on
    ],
)
def test_adj_matrix_to_diagonal_block_representation(
    matrix, num_nodes, block_size, expected, pad_value
):
    input_matrix = torch.Tensor(matrix)
    input_matrix = input_matrix[:, :, None]
    expected = torch.Tensor(expected)
    expected = expected[:, :, :, None]
    output = adj_matrix_to_diagonal_block_representation(
        input_matrix, num_nodes, block_size, pad_value=pad_value
    )
    assert torch.equal(output, expected)


@pytest.mark.parametrize(
    "diagonal_block,num_nodes,expected",
    [
        # fmt: off
        (
            [[[1]]],
            2,
            [[0, 0],
             [1, 0]],
        ),
        (
            [[[0, 0],
              [1, 0]]],
            2,
            [[0, 0],
             [1, 0]],
        ),
        (
            [[[0, 0, 0],
              [0, 0, 0],
              [1, 0, 0]]],
            2,
            [[0, 0],
             [1, 0]],
        ),
        (
            [[[0, 0, 0],
              [0, 0, 0],
              [1, 1, 0]]],
            3,
            [[0, 0, 0],
             [0, 0, 0],
             [1, 1, 0]],
        ),
        (
            [[[0, 0],
              [1, 1]]],
            3,
            [[0, 0, 0],
             [0, 0, 0],
             [1, 1, 0]],
        ),
        (
            [[[1, 0],
              [0, 1]],
             [[0, 0],
              [1, 0]],
             [[0, 0],
              [1, 0]]],
            4,
            [[0, 0, 0, 0],
             [1, 0, 0, 0],
             [1, 0, 0, 0],
             [0, 1, 1, 0]],
        ),
        (
            [[[1, 0, 0],
              [1, 0, 0],
              [0, 1, 1]]],
            4,
            [[0, 0, 0, 0],
             [1, 0, 0, 0],
             [1, 0, 0, 0],
             [0, 1, 1, 0]],
        ),
        (
            [[[0]], [[1]], [[1]], [[1]], [[0]], [[1]]],
            4,
            [[0, 0, 0, 0],
             [1, 0, 0, 0],
             [1, 0, 0, 0],
             [0, 1, 1, 0]],
        ),
        (
            [[[0]], [[1]], [[1]], [[1]], [[0]], [[1]], [[0]], [[0]], [[0]], [[0]]],
            4,
            [[0, 0, 0, 0],
             [1, 0, 0, 0],
             [1, 0, 0, 0],
             [0, 1, 1, 0]],
        ),
        (
            [[[1, 0, 0],
              [1, 0, 0],
              [0, 1, 1]],
             [[0, 0, 0],
              [0, 0, 0],
              [1, 0, 0]],
             [[0, 0, 0],
              [0, 0, 0],
              [1, 0, 0]]],
            5,
            [[0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [0, 1, 1, 1, 0]],
        ),
        # fmt: on
    ],
)
def test_to_diagonal_block_to_adj_matrix_representation(
    diagonal_block, num_nodes, expected
):
    input_diagonal = torch.Tensor(diagonal_block)
    input_diagonal = input_diagonal[:, :, :, None]
    expected = torch.Tensor(expected)
    expected = expected[:, :, None]
    output = diagonal_block_to_adj_matrix_representation(input_diagonal, num_nodes)
    assert torch.equal(output, expected)
