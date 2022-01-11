import pytest

import numpy as np
import torch

from rga.util.adjmatrix.pad import minimize_and_pad


@pytest.mark.parametrize(
    "input_matrix,input_target_num_nodes,expected",
    [
        ([[0]], 1, [[0]]),
        ([[0, 1], [1, 0]], 2, [[0, 0], [1, 0]]),
        ([[0, 1], [1, 0]], 3, [[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
    ],
)
def test_minimize_and_pad(input_matrix, input_target_num_nodes, expected):
    input_matrix = np.array(input_matrix, dtype=np.float32)
    expected = torch.Tensor(np.array(expected, dtype=np.float32))
    expected = expected[:, :, None]
    output = minimize_and_pad(input_matrix, input_target_num_nodes)
    assert torch.equal(output, expected)
