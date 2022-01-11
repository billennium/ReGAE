import pytest

import numpy as np
import torch

# from rga.util.adjmatrix.remove_duplicates import remove_duplicates


# @pytest.mark.parametrize(
#     "grafs,unique_graphs",
#     [
#         ([(torch.Tensor([1, 2, 3]), 1)], [(torch.Tensor([1, 2, 3]), 1)]),
#     ],
# )
# def test_remove_dupliactes(grafs, unique_graphs):
#     output = remove_duplicates(grafs)
#     assert len(output) == len(unique_graphs)
#     assert all([torch.equal(a, b) for (a, b) in zip(output, unique_graphs)])
