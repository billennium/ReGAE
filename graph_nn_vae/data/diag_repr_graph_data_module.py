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
        self.collate_fn_train = self.collate_graph_batch
        self.collate_fn_val = self.collate_graph_batch
        self.collate_fn_test = self.collate_graph_batch

    def _adj_batch_to_diagonal(
        self, batch: List[Tuple[torch.Tensor, int]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:

        diag_represented_batch = []
        for example in batch:
            m = example[0] if self.use_labels else example
            matrix = m[0]
            num_nodes = m[1]
            diag_represented_matrix = adj_matrix_to_diagonal_representation(
                matrix, num_nodes
            )
            graph_mask = torch.ones(diag_represented_matrix.shape)

            if self.use_labels:
                processed_example = (
                    (diag_represented_matrix, graph_mask, num_nodes),
                    example[1],
                )
            else:
                processed_example = (diag_represented_matrix, graph_mask, num_nodes)

            diag_represented_batch.append(processed_example)

        return diag_represented_batch

    def prepare_data(self, *args, **kwargs):
        super().prepare_data(*args, **kwargs)
        self.train_dataset = self._adj_batch_to_diagonal(self.train_dataset)
        self.val_dataset = self._adj_batch_to_diagonal(self.val_dataset)
        self.test_dataset = self._adj_batch_to_diagonal(self.test_dataset)

    def collate_graph_batch(self, batch):
        # As part of the collation graph diag_repr are padded with 0.0 and the graph masks
        # are padded with 1.0 to represent the end of the graphs.

        graphs = torch.nn.utils.rnn.pad_sequence(
            [g[0][0] if self.use_labels else g[0] for g in batch],
            batch_first=True,
            padding_value=0.0,
        )
        graph_masks = torch.nn.utils.rnn.pad_sequence(
            [g[0][1] if self.use_labels else g[1] for g in batch],
            batch_first=True,
            padding_value=0.0,
        )
        num_nodes = torch.tensor([g[0][2] if self.use_labels else g[2] for g in batch])

        if self.use_labels:
            labels = torch.LongTensor([g[1] for g in batch])
            return graphs, graph_masks, num_nodes, labels

        else:
            return graphs, graph_masks, num_nodes
