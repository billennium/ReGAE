from argparse import ArgumentParser, ArgumentError
from typing import List

import numpy as np
import torch
from torch import nn, Tensor
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.nn.modules import activation

from graph_nn_vae.models.base import BaseModel
from graph_nn_vae.models.utils.layers import sequential_from_layer_sizes
from graph_nn_vae.models.utils.getters import get_activation_function
from graph_nn_vae.models.utils.calc import weighted_average


class GraphEncoder(BaseModel):
    def __init__(
        self,
        embedding_size: int,
        edge_size: int,
        encoder_hidden_layer_sizes: List[int],
        encoder_activation_function: str,
        **kwargs,
    ):
        self.embedding_size = embedding_size
        self.edge_size = edge_size
        super(GraphEncoder, self).__init__(**kwargs)

        input_size = 2 * embedding_size + edge_size
        output_size = 3 * embedding_size
        activation_f = get_activation_function(encoder_activation_function)
        self.edge_encoder = sequential_from_layer_sizes(
            input_size, output_size, encoder_hidden_layer_sizes, activation_f
        )

    def forward(self, input_batch: Tensor) -> Tensor:
        """
        :param input_batch:
            Batch consisting of a tuple of diagonally represented graphs and their number of nodes.

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
        num_nodes_batch = input_batch[1]

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

            encoder_input = torch.cat(
                (embeddings_left, embeddings_right, current_diagonal), dim=2
            )

            encoder_output = self.edge_encoder(encoder_input)

            (embedding, mem_overwrite_ratio, embedding_ratio,) = torch.split(
                encoder_output,
                [
                    self.embedding_size,
                    self.embedding_size,
                    self.embedding_size,
                ],
                dim=2,
            )

            weighted_prev_embedding = weighted_average(
                embeddings_left, embeddings_right, embedding_ratio
            )
            prev_embedding = weighted_average(
                weighted_prev_embedding, embedding, mem_overwrite_ratio
            )

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
        parser.add_argument(
            "--encoder_hidden_layer_sizes",
            dest="encoder_hidden_layer_sizes",
            default=[256],
            type=parse_layer_sizes_list,
            metavar="DECODER_H_SIZES",
            help="list of the sizes of the decoder's hidden layers",
        )
        parser.add_argument(
            "--encoder_activation_function",
            dest="encoder_activation_function",
            default="ReLU",
            type=str,
            metavar="ACTIVATION_F_NAME",
            help="name of the activation function of hidden layers",
        )
        return parser


class GraphDecoder(BaseModel):
    def __init__(
        self,
        embedding_size: int,
        edge_size: int,
        max_number_of_nodes: int,
        decoder_hidden_layer_sizes: List[int],
        decoder_activation_function: str,
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

        input_size = self.internal_embedding_size * 2
        output_size = self.internal_embedding_size * 4 + edge_size
        activation_f = get_activation_function(decoder_activation_function)
        self.edge_decoder = sequential_from_layer_sizes(
            input_size, output_size, decoder_hidden_layer_sizes, activation_f
        )

    def forward(self, graph_encoding_batch: Tensor) -> Tensor:
        """
        :param graph_encoding_batch: batch of graph encodings (products of an encoder) of dimensions [batch_size, embedding_size]
        :return: graph adjacency matrices tensor of dimensions [batch_size, num_nodes, num_nodes, edge_size]
        """
        decoded_diagonals = []
        # The working embeddings batch has this shape: [graph_idx x embdedding_idx x embedding]
        prev_doubled_embeddings = graph_encoding_batch[:, None, :]

        original_indices = torch.IntTensor(list(range(graph_encoding_batch.shape[0])))
        indices_of_finished_graphs = []

        for _ in range(self.max_number_of_nodes):
            edge_with_embeddings = self.edge_decoder(prev_doubled_embeddings)

            (decoded_edges, doubled_embeddings, mem_overwrite_ratio,) = torch.split(
                edge_with_embeddings,
                [
                    self.edge_size,
                    self.internal_embedding_size * 2,
                    self.internal_embedding_size * 2,
                ],
                dim=2,
            )

            decoded_edges = torch.tanh(decoded_edges)

            decoded_edges_padded = decoded_edges
            for i in sorted(indices_of_finished_graphs):
                decoded_edges_padded = torch.cat(
                    [
                        decoded_edges_padded[:i],
                        torch.ones(
                            (1, decoded_edges_padded.shape[1], 1),
                            device=decoded_edges_padded.device,
                        )
                        * -1,
                        decoded_edges_padded[i:],
                    ],
                )

            decoded_diagonals.append(decoded_edges_padded)

            indices_graphs_still_generating = (
                torch.mean(decoded_edges[:, :, 0], dim=1) > -0.3
            )

            indices_of_finished_graphs.extend(
                original_indices[~indices_graphs_still_generating].tolist()
            )
            original_indices = original_indices[indices_graphs_still_generating]

            doubled_embeddings = doubled_embeddings[indices_graphs_still_generating]
            mem_overwrite_ratio = mem_overwrite_ratio[indices_graphs_still_generating]
            prev_doubled_embeddings = prev_doubled_embeddings[
                indices_graphs_still_generating
            ]

            if doubled_embeddings.shape[0] == 0:
                break

            doubled_embeddings = weighted_average(
                doubled_embeddings, prev_doubled_embeddings, mem_overwrite_ratio
            )

            embedding_1, embedding_2 = torch.split(
                doubled_embeddings,
                [self.internal_embedding_size, self.internal_embedding_size],
                dim=2,
            )

            # add zeroes to both sides - these are the empty embeddings of the far-out edges
            prev_embeddings_1 = torch.nn.functional.pad(embedding_1, (0, 0, 1, 0))
            prev_embeddings_2 = torch.nn.functional.pad(embedding_2, (0, 0, 0, 1))
            prev_doubled_embeddings = torch.cat(
                (prev_embeddings_1, prev_embeddings_2), dim=2
            )

        concatenated_diagonals = torch.cat(decoded_diagonals, dim=1)
        max_concatenated_diagonals_length = int(
            self.max_number_of_nodes * (1 + self.max_number_of_nodes) / 2
        )

        pad_length = max_concatenated_diagonals_length - concatenated_diagonals.shape[1]
        concatenated_diagonals = torch.nn.functional.pad(
            concatenated_diagonals, (0, 0, 0, pad_length), value=-1.0
        )

        return concatenated_diagonals

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
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
        parser.add_argument(
            "--decoder_hidden_layer_sizes",
            dest="decoder_hidden_layer_sizes",
            default=[256],
            type=parse_layer_sizes_list,
            metavar="DECODER_H_SIZES",
            help="list of the sizes of the decoder's hidden layers",
        )
        parser.add_argument(
            "--max-num-nodes",
            "--max-number-of-nodes",
            dest="max_number_of_nodes",
            default=50,
            type=int,
            metavar="NUM_NODES",
            help="max number of nodes of generated graphs",
        )
        parser.add_argument(
            "--decoder_activation_function",
            dest="decoder_activation_function",
            default="ReLU",
            type=str,
            metavar="ACTIVATION_F_NAME",
            help="name of the activation function of hidden layers",
        )
        return parser


def parse_layer_sizes_list(s: str) -> List[int]:
    if isinstance(s, str):
        if "," in s:
            return [int(v) for v in s.split(",")]
        if "|" in s:
            return [int(v) for v in s.split("|")]
        if ":" in s:
            return [int(v) for v in s.split(":")]
    return [int(s)]
