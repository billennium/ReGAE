from argparse import ArgumentParser

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
            nn.Linear(2 * embedding_size + edge_size, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_size),
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
        return parser


class GraphDecoder(BaseModel):
    def __init__(
        self, embedding_size: int, edge_size: int, max_number_of_nodes: int, **kwargs
    ):
        self.embedding_size = embedding_size
        self.edge_size = edge_size
        self.max_number_of_nodes = max_number_of_nodes
        super().__init__(**kwargs)
        self.edge_decoder = nn.Sequential(
            nn.Linear(2 * embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_size + edge_size),
            nn.ReLU(),
        )

    def forward(self, graph_encoding_batch: Tensor) -> Tensor:
        """
        :param graph_encoding: the encoding of a graph (product of an encoder) of dimensions [batch_size, embedding_size]
        :return: graph adjacency matrices tensor of dimensions [batch_size, num_nodes, num_nodes, edge_size]
        """
        batch_size = graph_encoding_batch.shape[0]
        adjacency_matrices = torch.zeros(
            (
                batch_size,
                self.max_number_of_nodes,
                self.max_number_of_nodes,
                self.edge_size,
            )
        )
        batch_concatenated_diagonals = []

        for batch_idx, graph_encoding in enumerate(graph_encoding_batch):
            prev_doubled_embeddings = graph_encoding
            decoded_diagonals = []

            for diagonal_offset in range(1, self.max_number_of_nodes):
                edge_with_embedding = self.edge_decoder(prev_doubled_embeddings)
                decoded_edges, embeddings = torch.split(
                    edge_with_embedding, [self.edge_size, self.embedding_size], dim=1
                )
                if torch.mean(decoded_edges[:, 0]) < -0.5:
                    break

                prev_doubled_embeddings = torch.cat((embeddings, embeddings), dim=0)
                # add zeroes to both sides - these are the empty embeddings of the far-out edges
                prev_doubled_embeddings = torch.functional.pad(
                    prev_doubled_embeddings, (1, 1, 0, 0)
                )

                decoded_diagonals.append(decoded_edges)

            concatenated_diagonals = torch.cat(decoded_diagonals, dim=0)
            max_concatenated_diagonals_length = (
                self.max_number_of_nodes * (1 + self.max_number_of_nodes) / 2
            )
            pad_length = (
                max_concatenated_diagonals_length - concatenated_diagonals.shape[0]
            )
            concatenated_diagonals = torch.functional.pad(
                concatenated_diagonals, (0, pad_length, 0, 0), value=0.0
            )
            batch_concatenated_diagonals.append(concatenated_diagonals)

        return torch.stack(
            batch_concatenated_diagonals,
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
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


class EdgeDecoder(pl.LightningModule):
    def __init__(self, embedding_size: int, edge_size: int = 1, **kwargs):
        self.edge_size = edge_size
        self.embedding_size = embedding_size
        super().__init__(**kwargs)
        self.layers = nn.Sequential(
            nn.Linear(2 * embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_size + edge_size),
            nn.ReLU(),
        )

    def forward(
        self, prev_embedding_1: Tensor, prev_embedding_2: Tensor
    ) -> tuple[Tensor, Tensor]:
        x = torch.cat((prev_embedding_1, prev_embedding_2), 0)
        y = self.layers(x)
        return torch.split(y, [self.edge_size, self.embedding_size])
