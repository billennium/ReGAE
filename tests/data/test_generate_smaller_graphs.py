import pytest

import numpy as np
import torch
from graph_nn_vae.data.smooth_learning_step_graph_data_module import (
    generate_subgraphs,
)


@pytest.mark.parametrize(
    "graph,graph_size,n,stride,probability,expected",
    [
        ([0, 1, 1, 1, 0, 1], 4, 3, 1, 1.0, [[1, 1, 0], [1, 0, 1]]),
        ([0, 1, 1, 1, 0, 1], 4, 4, 1, 1.0, [[0, 1, 1, 1, 0, 1]]),
        ([0, 1, 1, 1, 0, 1], 4, 8, 1, 1.0, [[0, 1, 1, 1, 0, 1]]),
        ([0, 1, 1, 1, 0, 1], 4, 8, 5, 1.0, [[0, 1, 1, 1, 0, 1]]),
        (
            [0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1],
            6,
            3,
            1,
            1.0,
            [[1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
        ),
        (
            [0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1],
            6,
            4,
            1,
            1.0,
            [[1, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 1], [0, 0, 1, 0, 1, 1]],
        ),
        (
            [0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1],
            6,
            4,
            2,
            1.0,
            [[1, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 1]],
        ),
    ],
)
def test_generate_subgraphs(graph, graph_size, n, stride, probability, expected):
    graph = torch.Tensor(np.array(graph, dtype=np.float32))
    expected = torch.Tensor(np.array(expected, dtype=np.float32))
    graphs, graph_masks, graph_sizes = generate_subgraphs(
        graph=graph,
        graph_size=graph_size,
        n=n,
        stride=stride,
        probability=probability,
    )
    assert torch.equal(torch.stack(graphs), expected)
