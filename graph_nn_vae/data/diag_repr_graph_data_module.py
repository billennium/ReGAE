from typing import List, Tuple

import torch

from graph_nn_vae.data.adj_matrix_data_module import AdjMatrixDataModule
from graph_nn_vae.util.adjmatrix.diagonal_representation import (
    adj_matrix_to_diagonal_representation,
)


class DiagonalRepresentationGraphDataModule(AdjMatrixDataModule):
    data_name = "DiagRepr"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.collate_fn = collate_graph_batch

    def _adj_batch_to_diagonal(
        self, batch: List[Tuple[torch.Tensor, int]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:

        diag_represented_batch = []
        for m in batch:
            matrix = m[0]
            num_nodes = m[1]
            diag_represented_matrix = adj_matrix_to_diagonal_representation(
                matrix, num_nodes
            )
            graph_mask = torch.ones(diag_represented_matrix.shape)
            diag_represented_batch.append(
                (diag_represented_matrix, graph_mask, num_nodes)
            )

        return diag_represented_batch

    def prepare_data(self, *args, **kwargs):
        super().prepare_data(*args, **kwargs)
        self.train_dataset = self._adj_batch_to_diagonal(self.train_dataset)
        self.val_dataset = self._adj_batch_to_diagonal(self.val_dataset)
        self.test_dataset = self._adj_batch_to_diagonal(self.test_dataset)

    def train_dataloader(self, **kwargs):
        dl = super().train_dataloader(**kwargs)
        dl.collate_fn = self.collate_fn
        return dl

    def val_dataloader(self, **kwargs):
        dl = super().val_dataloader(**kwargs)
        dl.collate_fn = self.collate_fn
        return dl

    def test_dataloader(self, **kwargs):
        dl = super().test_dataloader(**kwargs)
        dl.collate_fn = self.collate_fn
        return dl


def collate_graph_batch(batch):
    # As part of the collation graph diag_repr are padded with 0.0 and the graph masks
    # are padded with 1.0 to represent the end of the graphs.
    graphs = torch.nn.utils.rnn.pad_sequence(
        [g[0] for g in batch], batch_first=True, padding_value=0.0
    )
    graph_masks = torch.nn.utils.rnn.pad_sequence(
        [g[1] for g in batch], batch_first=True, padding_value=0.0
    )
    num_nodes = torch.tensor([g[2] for g in batch])
    return graphs, graph_masks, num_nodes
