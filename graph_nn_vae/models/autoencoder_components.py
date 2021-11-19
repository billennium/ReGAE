from argparse import ArgumentParser, ArgumentError
from typing import List

import numpy as np
import torch
from torch import nn, Tensor
import pytorch_lightning as pl
from torch.nn import functional as F

from graph_nn_vae.models.base import BaseModel
from graph_nn_vae.models.utils.layers import sequential_from_layer_sizes
from graph_nn_vae.models.utils.calc import weighted_average


class GraphEncoder(BaseModel):
    def __init__(
        self,
        embedding_size: int,
        edge_size: int,
        encoder_hidden_layer_sizes: List[int],
        **kwargs,
    ):
        self.embedding_size = embedding_size
        self.edge_size = edge_size
        super(GraphEncoder, self).__init__(**kwargs)

        input_size = 2 * embedding_size + edge_size
        output_size = 3 * embedding_size
        self.edge_encoder = sequential_from_layer_sizes(
            input_size, output_size, encoder_hidden_layer_sizes
        )

    def forward(self, input_batch: Tensor) -> Tensor:
        """
        :param input_batch:
            Batch consisting of a tuple of diagonally represented graphs and their number of nodes.
            The batch has to be sorted by the number of nodes of the graphs.
        :return:
            Graph embedding Tensor of dimensions [batch_size x embedding_size]
        """
        diag_repr_graphs_batch = input_batch[0]
        diag_repr_graphs_batch.requires_grad = True
        num_nodes_batch = input_batch[1]

        max_num_nodes = num_nodes_batch[0]
        num_diagonals = max_num_nodes - 1
        first_diag_len = num_diagonals
        diag_right_pos = int((1 + num_diagonals) * num_diagonals / 2)
        num_nodes_to_graph_count = cdf_of_sorted_desc_list(num_nodes_batch)

        num_graphs_considered = num_nodes_to_graph_count[max_num_nodes]
        prev_embedding = torch.zeros(
            (
                num_graphs_considered * max_num_nodes,
                self.embedding_size,
            ),
            requires_grad=True,
            device=diag_repr_graphs_batch.device,
        )

        for diag_offset in range(num_diagonals):
            graphs_in_curr_diagonal = num_nodes_to_graph_count[
                num_diagonals - diag_offset + 1
            ]
            if num_graphs_considered != graphs_in_curr_diagonal:
                num_graps_not_yet_considered = (
                    graphs_in_curr_diagonal - num_graphs_considered
                )
                num_graphs_considered = graphs_in_curr_diagonal
                num_nodes_in_prev_diag = max_num_nodes - diag_offset
                pad_zero_embedding = torch.zeros(
                    (
                        num_graps_not_yet_considered * num_nodes_in_prev_diag,
                        self.embedding_size,
                    ),
                    requires_grad=True,
                    device=diag_repr_graphs_batch.device,
                )
                prev_embedding = torch.cat((prev_embedding, pad_zero_embedding), dim=0)

            diag_len = first_diag_len - diag_offset
            prev_diag_len = diag_len.item() + 1

            emb_left_indices = [
                i for i in range(len(prev_embedding)) if (i + 1) % prev_diag_len != 0
            ]
            embeddings_left = prev_embedding[emb_left_indices, :].requires_grad_()
            emb_right_indices = [i + 1 for i in emb_left_indices]
            embeddings_right = prev_embedding[emb_right_indices, :].requires_grad_()

            diag_left_pos = diag_right_pos - diag_len
            curr_diagonal = diag_repr_graphs_batch[
                :graphs_in_curr_diagonal, diag_left_pos:diag_right_pos, :
            ]
            curr_diagonal = torch.flatten(curr_diagonal, end_dim=1)

            encoder_input = torch.cat(
                (embeddings_left, embeddings_right, curr_diagonal), dim=1
            )
            encoder_output = self.edge_encoder(encoder_input)

            (embedding, mem_overwrite_ratio, embedding_ratio,) = torch.split(
                encoder_output,
                [
                    self.embedding_size,
                    self.embedding_size,
                    self.embedding_size,
                ],
                dim=1,
            )
            weighted_prev_embedding = weighted_average(
                embeddings_left, embeddings_right, embedding_ratio
            )
            new_embedding = weighted_average(
                weighted_prev_embedding, embedding, mem_overwrite_ratio
            )

            prev_embedding = new_embedding
            diag_right_pos = diag_left_pos

        return prev_embedding

        # returned_embeddings = []
        # for batch_idx, diagonal_repr_graph in enumerate(diag_repr_graphs_batch):
        #     num_nodes = num_nodes_batch[batch_idx]

        #     prev_embedding = torch.zeros(
        #         (num_nodes, self.embedding_size),
        #         requires_grad=True,
        #         device=diagonal_repr_graph.device,
        #     )

        #     """
        #     The graph is represented in the diagonal form with a shape like [summed_diagonal_length+padding, edge_size].
        #     For example, skipping the edge_size dimension for a graph of adjacency matrix:
        #      x 0 1 2 3
        #     y
        #     0  0 0 0 0
        #     1  1 0 0 0
        #     2  1 0 0 0
        #     3  0 1 1 0
        #         |
        #         V
        #     011101 + -1 padding
        #     """

        #     num_diagonals = num_nodes - 1
        #     first_diag_length = num_diagonals
        #     diag_right_pos = int((1 + num_diagonals) * num_diagonals / 2)
        #     for diagonal_offset in range(num_diagonals):
        #         embeddings_left = prev_embedding[:-1, :]
        #         embeddings_right = prev_embedding[1:, :]

        #         diag_length = first_diag_length - diagonal_offset
        #         diag_left_pos = diag_right_pos - diag_length
        #         curr_diagonal = diagonal_repr_graph[diag_left_pos:diag_right_pos, :]

        #         encoder_input = torch.cat(
        #             (embeddings_left, embeddings_right, curr_diagonal), dim=1
        #         )
        #         encoder_output = self.edge_encoder(encoder_input)

        #         (embedding, mem_overwrite_ratio, embedding_ratio,) = torch.split(
        #             encoder_output,
        #             [
        #                 self.embedding_size,
        #                 self.embedding_size,
        #                 self.embedding_size,
        #             ],
        #             dim=1,
        #         )
        #         weighted_prev_embedding = weighted_average(
        #             embeddings_left, embeddings_right, embedding_ratio
        #         )
        #         new_embedding = weighted_average(
        #             weighted_prev_embedding, embedding, mem_overwrite_ratio
        #         )

        #         prev_embedding = new_embedding
        #         diag_right_pos = diag_left_pos
        #     returned_embeddings.append(prev_embedding)

        # embeddings_batch = torch.cat(returned_embeddings)

        # return embeddings_batch

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
        return parser


def cdf_of_sorted_desc_list(l: list[int]) -> list[int]:
    out = [0] * (l[0] + 1)
    for i, v in enumerate(l):
        out[v] += 1
    curr_num = 0
    for i, v in reversed(list(enumerate(out))):
        curr_num += v
        out[i] = curr_num
    return out


class GraphDecoder(BaseModel):
    def __init__(
        self,
        embedding_size: int,
        edge_size: int,
        max_number_of_nodes: int,
        decoder_hidden_layer_sizes: List[int],
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
        self.edge_decoder = sequential_from_layer_sizes(
            input_size, output_size, decoder_hidden_layer_sizes
        )

    def forward(self, graph_encoding_batch: Tensor) -> Tensor:
        """
        :param graph_encoding_batch: batch of graph encodings (products of an encoder) of dimensions [batch_size, embedding_size]
        :return: graph adjacency matrices tensor of dimensions [batch_size, num_nodes, num_nodes, edge_size]
        """
        batch_concatenated_diagonals = []

        for batch_idx, graph_encoding in enumerate(graph_encoding_batch):
            prev_doubled_embeddings = graph_encoding[None, :]
            decoded_diagonals = []

            for _ in range(self.max_number_of_nodes):
                edge_with_embeddings = self.edge_decoder(prev_doubled_embeddings)
                (decoded_edges, doubled_embeddings, mem_overwrite_ratio,) = torch.split(
                    edge_with_embeddings,
                    [
                        self.edge_size,
                        self.internal_embedding_size * 2,
                        self.internal_embedding_size * 2,
                    ],
                    dim=1,
                )
                decoded_edges = torch.tanh(decoded_edges)
                decoded_diagonals.append(decoded_edges)
                if torch.mean(decoded_edges[:, 0]) < -0.3:
                    break

                doubled_embeddings = weighted_average(
                    doubled_embeddings, prev_doubled_embeddings, mem_overwrite_ratio
                )

                embedding_1, embedding_2 = torch.split(
                    doubled_embeddings,
                    [self.internal_embedding_size, self.internal_embedding_size],
                    dim=1,
                )

                # add zeroes to both sides - these are the empty embeddings of the far-out edges
                prev_embeddings_1 = torch.nn.functional.pad(embedding_1, (0, 0, 1, 0))
                prev_embeddings_2 = torch.nn.functional.pad(embedding_2, (0, 0, 0, 1))
                prev_doubled_embeddings = torch.cat(
                    (prev_embeddings_1, prev_embeddings_2), dim=1
                )

            concatenated_diagonals = torch.cat(decoded_diagonals, dim=0)
            max_concatenated_diagonals_length = int(
                self.max_number_of_nodes * (1 + self.max_number_of_nodes) / 2
            )
            pad_length = (
                max_concatenated_diagonals_length - concatenated_diagonals.shape[0]
            )
            concatenated_diagonals = torch.nn.functional.pad(
                concatenated_diagonals, (0, 0, 0, pad_length), value=-1.0
            )
            batch_concatenated_diagonals.append(concatenated_diagonals)

        return torch.stack(
            batch_concatenated_diagonals,
        )

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
