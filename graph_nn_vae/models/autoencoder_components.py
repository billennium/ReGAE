import torch
from torch import nn, Tensor
import pytorch_lightning as pl
from torch.nn import functional as F


class GraphEncoder(pl.LightningModule):
    def __init__(self, embedding_size: int, edge_size: int, **kwargs):
        self.embedding_size = embedding_size
        self.edge_size = edge_size
        super().__init__(**kwargs)
        self.edge_encoder = EdgeEncoder(
            embedding_size=embedding_size, edge_size=edge_size
        )

    def forward(self, adjacency_matrices_batch: Tensor) -> Tensor:
        """
        :param adjacency matrix: a stripped adjacency matrix Tensor of dimensions [num_nodes-1, num_nodes-1, edge_size]
        :return: a graph embedding Tensor of dimensions [embedding_size]
        """
        returned_embeddings = torch.zeros(
            (adjacency_matrices_batch.shape[0], self.embedding_size)
        )
        for batch_idx, adjacency_matrix in enumerate(adjacency_matrices_batch):
            num_nodes = 0
            for i in range(adjacency_matrix.shape[0]):
                diagonal = torch.diagonal(adjacency_matrix, offset=-i)
                if torch.sum(diagonal) != 0.0:
                    num_nodes = adjacency_matrix.shape[0] - i + 1
                    break
            embedding_matrix = torch.zeros(
                [num_nodes, num_nodes, self.embedding_size], dtype=torch.float32
            )
            emb_m_y_diff = adjacency_matrix.shape[0] - embedding_matrix.shape[0]

            for diagonal_offset in reversed(range(num_nodes - 1)):
                for edge_offset in range(diagonal_offset + 1):
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
                    edge_x = diagonal_offset - edge_offset
                    edge_y = adjacency_matrix.shape[0] - 1 - edge_offset
                    edge = adjacency_matrix[edge_y, edge_x, :]

                    embedding_1_x = edge_x + 1
                    embedding_1_y = edge_y - emb_m_y_diff
                    embedding_1 = embedding_matrix[embedding_1_y, embedding_1_x, :]

                    embedding_2_x = edge_x
                    embedding_2_y = edge_y - emb_m_y_diff - 1
                    embedding_2 = embedding_matrix[embedding_2_y, embedding_2_x, :]

                    embedding = self.edge_encoder(edge, embedding_1, embedding_2)
                    embedding_matrix[
                        edge_y - emb_m_y_diff : edge_y - emb_m_y_diff + 1,
                        edge_x : edge_x + 1,
                        :,
                    ] = embedding

            returned_embeddings[batch_idx, :] = embedding_matrix[
                embedding_matrix.shape[0] - 1, 0, :
            ]
        return returned_embeddings


class EdgeEncoder(pl.LightningModule):
    def __init__(self, embedding_size: int, edge_size: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.layers = nn.Sequential(
            nn.Linear(2 * embedding_size + edge_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_size),
            nn.ReLU(),
        )

    def forward(
        self, edge: Tensor, prev_embedding_1: Tensor, prev_embedding_2: Tensor
    ) -> Tensor:
        x = torch.cat((edge, prev_embedding_1, prev_embedding_2), 0)
        return self.layers(x)


class GraphDecoder(pl.LightningModule):
    def __init__(
        self, embedding_size: int, edge_size: int, max_number_of_nodes: int, **kwargs
    ):
        self.embedding_size = embedding_size
        self.edge_size = edge_size
        self.max_number_of_nodes = max_number_of_nodes
        super().__init__(**kwargs)
        self.edge_encoder = EdgeDecoder(
            embedding_size=embedding_size, edge_size=edge_size
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
        # for batch_idx, graph_encoding in enumerate(graph_encoding_batch):
        #     for diagonal_offset in range(1, self.max_number_of_nodes):
        #         for edge_offset in range(diagonal_offset + 1):
        #             """
        #             The adjacency matrix has now a shape like [y, x, edge_size].
        #             For example, skipping the edge_size dimension:
        #              x 0 1 2 3
        #             y
        #             0  0 0 0 0
        #             1  1 0 0 0
        #             2  1 0 0 0
        #             3  0 1 1 0
        #             """
        #             edge_x = diagonal_offset - edge_offset
        #             edge_y = adjacency_matrix.shape[0] - 1 - edge_offset
        #             edge = adjacency_matrix[edge_y, edge_x, :]

        #             embedding_1_x = edge_x + 1
        #             embedding_1_y = edge_y - emb_m_y_diff
        #             embedding_1 = embedding_matrix[embedding_1_y, embedding_1_x, :]

        #             embedding_2_x = edge_x
        #             embedding_2_y = edge_y - emb_m_y_diff - 1
        #             embedding_2 = embedding_matrix[embedding_2_y, embedding_2_x, :]

        #             embedding = self.edge_encoder(edge, embedding_1, embedding_2)
        #             embedding_matrix[
        #                 edge_y - emb_m_y_diff : edge_y - emb_m_y_diff + 1,
        #                 edge_x : edge_x + 1,
        #                 :,
        #             ] = embedding

        #     returned_embeddings[batch_idx, :] = embedding_matrix[
        #         embedding_matrix.shape[0] - 1, 0, :
        #     ]
        # return returned_embeddings


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
