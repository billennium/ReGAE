from argparse import ArgumentParser, ArgumentError

import torch
from torch import nn, Tensor
import pytorch_lightning as pl
from torch.nn import functional as F

from graph_nn_vae.models.base import BaseModel


class GraphEncoder(BaseModel):
    def __init__(self, embedding_size: int, edge_size: int, **kwargs):
        self.embedding_size = embedding_size
        self.edge_size = edge_size
        super(GraphEncoder, self).__init__(**kwargs)
        self.edge_encoder = nn.Sequential(
            nn.Linear(2 * embedding_size + edge_size, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_size),
        )

    def forward(self, adjacency_matrices_batch: Tensor) -> Tensor:
        """
        :param adjacency matrix: a stripped adjacency matrix Tensor of dimensions [num_nodes-1, num_nodes-1, edge_size]
        :return: a graph embedding Tensor of dimensions [embedding_size]
        """
        returned_embeddings = []
        for batch_idx, adjacency_matrix in enumerate(adjacency_matrices_batch):
            num_nodes = 0
            for i in range(adjacency_matrix.shape[0]):
                diagonal = torch.diagonal(adjacency_matrix, offset=-i)
                if torch.sum(diagonal) != 0.0:
                    num_nodes = adjacency_matrix.shape[0] - i + 1
                    break

            prev_embedding = torch.zeros(
                (num_nodes, self.embedding_size),
                requires_grad=True,
                device=adjacency_matrices_batch.device,
            )

            """
            The adjacency matrix has now a shape like [y, x, edge_size].
            For example, skipping the edge_size dimension:
             x 0 1 2 3
            y
            0  0 0 0 0
            1  1 0 0 0
            2  1 0 0 0
            3  0 1 1 0
            """

            for diagonal_offset in reversed(range(1, num_nodes)):
                embeddings_left = prev_embedding[:-1, :]
                embeddings_right = prev_embedding[1:, :]
                current_diagonal = (
                    torch.diagonal(
                        adjacency_matrix,
                        offset=diagonal_offset - adjacency_matrix.shape[0],
                    )
                    .transpose(1, 0)
                    .requires_grad_()
                )

                encoder_input = torch.cat(
                    (embeddings_left, embeddings_right, current_diagonal), 1
                )
                prev_embedding = self.edge_encoder(encoder_input)

            returned_embeddings.append(prev_embedding)
        embeddings_batch = torch.cat(returned_embeddings)

        return embeddings_batch

    def step(self, batch: Tensor) -> Tensor:
        embeddings = self(batch)
        num_nodes = torch.zeros((len(batch), self.embedding_size), device=batch.device)
        for i, adjacency_matrix in enumerate(batch):
            num_nodes[i, 0] = torch.sum(adjacency_matrix)
        return F.mse_loss(embeddings, num_nodes)

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
        return parser


class GraphDecoder(BaseModel):
    def __init__(
        self, embedding_size: int, edge_size: int, max_number_of_nodes: int, **kwargs
    ):
        if embedding_size % 2 != 0:
            raise ValueError(
                "graph decoder's input graph embedding size must be divisible by 2"
            )
        self.internal_embedding_size = int(embedding_size / 2)
        self.edge_size = edge_size
        self.max_number_of_nodes = max_number_of_nodes
        super().__init__(**kwargs)
        self.edge_decoder = nn.Sequential(
            nn.Linear(self.internal_embedding_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.internal_embedding_size * 4 + edge_size),
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

            for diagonal_offset in range(1, self.max_number_of_nodes + 1):
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
                decoded_edges = nn.functional.tanh(decoded_edges)
                decoded_diagonals.append(decoded_edges)
                if torch.mean(decoded_edges[:, 0]) < -0.3:
                    break

                mem_overwrite_ratio = torch.sigmoid(mem_overwrite_ratio)
                doubled_embeddings = (
                    doubled_embeddings * mem_overwrite_ratio
                    + prev_doubled_embeddings
                    * (
                        torch.ones(
                            [self.internal_embedding_size * 2],
                            device=self.device,
                            requires_grad=True,
                        )
                        - mem_overwrite_ratio
                    )
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
            "--max-num-nodes",
            "--max-number-of-nodes",
            dest="max_number_of_nodes",
            default=50,
            type=int,
            metavar="NUM_NODES",
            help="max number of nodes of generated graphs",
        )
        return parser
