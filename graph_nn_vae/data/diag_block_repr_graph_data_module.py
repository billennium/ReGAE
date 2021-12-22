from typing import List, Tuple
from argparse import ArgumentError, ArgumentParser

import torch

from graph_nn_vae.data.diag_repr_graph_data_module import (
    DiagonalRepresentationGraphDataModule,
)
from graph_nn_vae.util.adjmatrix.diagonal_block_representation import (
    adj_matrix_to_diagonal_block_representation,
)


class DiagonalBlockRepresentationGraphDataModule(DiagonalRepresentationGraphDataModule):
    data_name = "DiagBlockRepr"

    def __init__(self, block_size: int, **kwargs):
        self.block_size = block_size
        super().__init__(**kwargs)

    # override
    def adjust_batch_representation(
        self, batch: List[Tuple[torch.Tensor, int]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:

        diag_block_represented_batch = []
        for graph_info_set in batch:
            graph_info = graph_info_set[0] if self.use_labels else graph_info_set
            matrix = graph_info[0]
            num_nodes = graph_info[1]
            diag_block_graph = adj_matrix_to_diagonal_block_representation(
                matrix, num_nodes, self.block_size
            )
            adj_matrix_mask = torch.tril(
                torch.ones((num_nodes, num_nodes)), diagonal=-1
            )[:, :, None]
            diag_block_mask = adj_matrix_to_diagonal_block_representation(
                adj_matrix_mask, num_nodes, self.block_size
            )

            processed_example = (diag_block_graph, diag_block_mask, num_nodes)
            if self.use_labels:
                processed_example = (processed_example, graph_info_set[1])

            diag_block_represented_batch.append(processed_example)

        return diag_block_represented_batch

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

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = DiagonalRepresentationGraphDataModule.add_model_specific_args(
            parent_parser
        )
        try:  # may collide with an autoencoder module, but that's fine
            parser.add_argument(
                "--block_size",
                dest="block_size",
                default=1,
                type=int,
                help="size (width or height) of a block of adjacency matrix edges",
            )
        except ArgumentError:
            pass

        return parser
