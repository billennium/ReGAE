from argparse import ArgumentParser, ArgumentError
from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from graph_nn_vae.models.base import BaseModel
from graph_nn_vae.models.edge_decoders import MemoryEdgeDecoder
from graph_nn_vae.models.edge_encoders import MemoryEdgeEncoder
from graph_nn_vae.models.utils.getters import get_activation_function
from graph_nn_vae.models.utils.calc import weighted_average
from graph_nn_vae.models.utils.layers import (
    sequential_from_layer_sizes,
    parse_layer_sizes_list,
)


class GraphEncoder(BaseModel):
    def __init__(
        self,
        embedding_size: int,
        edge_size: int,
        **kwargs,
    ):
        self.embedding_size = embedding_size
        self.edge_size = edge_size
        super(GraphEncoder, self).__init__(**kwargs)
        self.edge_encoder = MemoryEdgeEncoder(embedding_size, edge_size, **kwargs)

    def forward(self, input_batch: Tensor) -> Tensor:
        """
        :param input_batch:
            Batch consisting of a tuple of diagonally represented graphs, their masks (unused here) and their number of nodes.

            The graphs shall be represented in the "diagonal form" with a shape like [summed_diagonal_length+padding, edge_size].
            For example, skipping the edge_size dimension for a graph of adjacency matrix:
             x 0 1 2 3
            y
            0  0 0 0 0
            1  1 0 0 0
            2  1 0 0 0
            3  0 1 1 0
                |
                V
            011101 + -1 padding

        :return:
            Graph embedding Tensor of dimensions [batch_size x embedding_size]
        """

        diagonal_repr_graphs_batch = input_batch[0]
        diagonal_repr_graphs_batch.requires_grad = True
        num_nodes_batch = input_batch[2]

        sorted_num_nodes_batch, ordered_indices = num_nodes_batch.sort(descending=True)
        _, indices_in_original_batch_order = ordered_indices.sort()

        diagonal_repr_graphs_batch = diagonal_repr_graphs_batch[ordered_indices]

        max_num_nodes = sorted_num_nodes_batch[0]
        num_diagonals = max_num_nodes - 1
        first_diag_length = num_diagonals
        diag_right_pos = int((1 + num_diagonals) * num_diagonals / 2)

        graph_counts_per_size = torch.bincount(num_nodes_batch)

        # Embedding batch is represented in the shape: [graph_idx, embedding_idx, embedding]
        # Starting with `0` for no graphs yet. Will get filled approprately in the recurrent loop.
        prev_embedding = torch.zeros(
            (0, max_num_nodes, self.embedding_size),
            requires_grad=True,
            device=diagonal_repr_graphs_batch.device,
        )

        for diagonal_offset in range(max_num_nodes - 1):
            # Some graphs from the input batch may have been too small for the previous diagonal.
            # Check if they should be added now and init their embeddings.
            graphs_to_add_in_curr_diag = graph_counts_per_size[
                max_num_nodes - diagonal_offset
            ]
            if graphs_to_add_in_curr_diag != 0:
                new_graph_init_tokens = torch.zeros(
                    (
                        graphs_to_add_in_curr_diag,
                        max_num_nodes - diagonal_offset,
                        self.embedding_size,
                    ),
                    requires_grad=True,
                    device=diagonal_repr_graphs_batch.device,
                )
                prev_embedding = torch.cat([prev_embedding, new_graph_init_tokens])

            diag_length = first_diag_length - diagonal_offset
            diag_left_pos = diag_right_pos - diag_length
            current_diagonal = diagonal_repr_graphs_batch[
                : prev_embedding.shape[0], diag_left_pos:diag_right_pos, :
            ]

            embeddings_left = prev_embedding[:, :-1, :]
            embeddings_right = prev_embedding[:, 1:, :]

            new_embedding = self.edge_encoder(
                current_diagonal, embeddings_left, embeddings_right
            )
            prev_embedding = new_embedding

            diag_right_pos = diag_left_pos

        # Reorder back to the original batch order and skip the no longer needed second dimension.
        return prev_embedding[indices_in_original_batch_order, 0, :]

    def step(self, batch: Tensor) -> Tensor:
        embeddings = self(batch)
        diagonal_repr_graphs_batch = batch[0]
        num_nodes_batch = batch[1]
        num_edges = torch.zeros(
            (len(diagonal_repr_graphs_batch), self.embedding_size),
            device=diagonal_repr_graphs_batch.device,
        )
        for i, diagonal_repr_graph in enumerate(diagonal_repr_graphs_batch):
            num_nodes = num_nodes_batch[i].item()
            non_padded_diagonal_length = int(((1 + num_nodes) * num_nodes) / 2)
            diagonal_repr_graph = diagonal_repr_graph[:non_padded_diagonal_length]
            num_edges[i, 0] = torch.sum(diagonal_repr_graph)
        return F.mse_loss(embeddings, num_edges)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MemoryEdgeEncoder.add_model_specific_args(parent_parser=parser)
        try:  # these may collide with an upper autoencoder, but that's fine
            parser = BaseModel.add_model_specific_args(parent_parser=parser)
        except ArgumentError:
            pass
        try:  # these may collide with an encoder module, but that's fine
            parser = BaseModel.add_model_specific_args(parent_parser=parser)
            parser.add_argument(
                "--embedding-size",
                dest="embedding_size",
                default=32,
                type=int,
                metavar="EMB_SIZE",
                help="size of the encoder output graph embedding",
            )
            parser.add_argument(
                "--edge-size",
                dest="edge_size",
                default=1,
                type=int,
                metavar="EDGE_SIZE",
                help="number of dimensions of a graph's edge",
            )
        except ArgumentError:
            pass
        return parser


class GraphDecoder(BaseModel):
    def __init__(
        self,
        embedding_size: int,
        edge_size: int,
        max_number_of_nodes: int,
        **kwargs,
    ):
        if embedding_size % 2 != 0:
            raise ValueError(
                "graph decoder's input graph embedding size must be divisible by 2"
            )
        self.internal_embedding_size = int(embedding_size / 2)
        self.edge_size = edge_size
        self.max_number_of_nodes = max_number_of_nodes
        super().__init__(**kwargs)

        self.edge_decoder = MemoryEdgeDecoder(
            embedding_size=self.internal_embedding_size, edge_size=edge_size, **kwargs
        )

    def forward(self, graph_encoding_batch: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param graph_encoding_batch: batch of graph encodings (products of an encoder) of dimensions [batch_size, embedding_size]
        :return: graph adjacency matrices tensor of dimensions [batch_size, num_nodes, num_nodes, edge_size]
        """
        decoded_diagonals_with_masks = []
        # The working embeddings batch has this shape: [graph_idx x embdedding_idx x embedding]
        prev_doubled_embeddings = graph_encoding_batch[:, None, :]
        prev_embeddings_l, prev_embeddings_r = torch.split(
            prev_doubled_embeddings,
            (self.internal_embedding_size, self.internal_embedding_size),
            dim=-1,
        )

        original_indices = torch.IntTensor(list(range(graph_encoding_batch.shape[0])))
        indices_of_finished_graphs = []

        for _ in range(self.max_number_of_nodes):
            (
                decoded_edges_with_mask,
                new_embedding_l,
                new_embedding_r,
            ) = self.edge_decoder(prev_embeddings_l, prev_embeddings_r)

            # decoded_edges_with_mask = torch.sigmoid(decoded_edges_with_mask)
            masks = decoded_edges_with_mask[:, :, 0]
            # just here, not part of the output - used for checking if the graphs are finished in the loop
            masks = torch.sigmoid(masks)

            decoded_edges_with_mask_padded = decoded_edges_with_mask
            for i in sorted(indices_of_finished_graphs):
                decoded_edges_with_mask_padded = torch.cat(
                    [
                        decoded_edges_with_mask_padded[:i],
                        torch.full(
                            (
                                1,
                                decoded_edges_with_mask_padded.shape[1],
                                decoded_edges_with_mask_padded.shape[2],
                            ),
                            fill_value=float("-inf"),
                            device=decoded_edges_with_mask_padded.device,
                        ),
                        decoded_edges_with_mask_padded[i:],
                    ],
                )
            decoded_diagonals_with_masks.append(decoded_edges_with_mask_padded)

            indices_graphs_still_generating = torch.mean(masks, dim=1) > 0.5

            indices_of_finished_graphs.extend(
                original_indices[~indices_graphs_still_generating].tolist()
            )
            original_indices = original_indices[indices_graphs_still_generating]

            new_embedding_l = new_embedding_l[indices_graphs_still_generating]
            new_embedding_r = new_embedding_r[indices_graphs_still_generating]

            if new_embedding_l.shape[0] == 0:
                break

            # add zeroes to both sides - these are the empty embeddings of the far-out edges
            prev_embeddings_l = torch.nn.functional.pad(new_embedding_l, (0, 0, 1, 0))
            prev_embeddings_r = torch.nn.functional.pad(new_embedding_r, (0, 0, 0, 1))

        concatenated_diagonals_with_masks = torch.cat(
            decoded_diagonals_with_masks, dim=1
        )
        masks, concatenated_diagonals = torch.split(
            concatenated_diagonals_with_masks, (1, self.edge_size), dim=2
        )

        return concatenated_diagonals, masks

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MemoryEdgeDecoder.add_model_specific_args(parent_parser=parser)
        try:  # these may collide with an upper autoencoder, but that's fine
            parser = BaseModel.add_model_specific_args(parent_parser=parser)
        except ArgumentError:
            pass
        try:  # these may collide with an encoder module, but that's fine
            parser.add_argument(
                "--embedding-size",
                dest="embedding_size",
                default=32,
                type=int,
                metavar="EMB_SIZE",
                help="size of the encoder output graph embedding",
            )
            parser.add_argument(
                "--edge-size",
                dest="edge_size",
                default=1,
                type=int,
                metavar="EDGE_SIZE",
                help="number of dimensions of a graph's edge",
            )
        except ArgumentError:
            pass
        parser.add_argument(
            "--max-num-nodes",
            "--max-number-of-nodes",
            dest="max_number_of_nodes",
            default=50,
            type=int,
            metavar="NUM_NODES",
            help="max number of nodes of generated graphs",
        )
        return parser
