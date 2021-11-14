from typing import List, Tuple
from argparse import ArgumentParser
import networkx as nx
import numpy as np

import torch

from graph_nn_vae.data.data_module import BaseDataModule
from graph_nn_vae.data.synthetic_graphs_create import create_synthetic_graphs
from graph_nn_vae.util import adjmatrix, split_dataset_train_val_test
from graph_nn_vae.util.graphs import max_number_of_nodes_in_graphs
from graph_nn_vae.util.adjmatrix.diagonal_representation import (
    adj_matrix_to_diagonal_representation,
)


class AdjMatrixDataModule(BaseDataModule):
    data_name = "AdjMatrix"

    def __init__(
        self, num_dataset_graph_permutations: int, bfs: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_dataset_graph_permutations = num_dataset_graph_permutations
        self.bfs = bfs
        self.prepare_data()

    def create_graphs(self) -> List[nx.Graph]:
        """
        Overload this function to specify graphs for the dataset.
        """
        return NotImplementedError

    def max_number_of_nodes_in_graphs(self, graphs: List[nx.Graph]) -> int:
        max_number_of_nodes = 0
        for graph in graphs:
            if graph.number_of_nodes() > max_number_of_nodes:
                max_number_of_nodes = graph.number_of_nodes()
        return max_number_of_nodes

    def nx_to_minimized_padded_adjacency_matrices(
        self, nx_graphs: List[nx.Graph]
    ) -> List[Tuple[torch.Tensor, int]]:
        """
        Returns tuples of adj matrices with number of nodes.
        """
        max_number_of_nodes = max_number_of_nodes_in_graphs(nx_graphs)

        adjacency_matrices = []
        for nx_graph in nx_graphs:
            np_adj_matrix = nx.to_numpy_array(nx_graph, dtype=np.float32)
            for i in range(self.num_dataset_graph_permutations):
                if i != 0:
                    adj_matrix = adjmatrix.random_permute(np_adj_matrix)
                else:
                    adj_matrix = np_adj_matrix

                if self.bfs:
                    adj_matrix = adjmatrix.bfs_ordering(adj_matrix)

                reshaped_matrix = adjmatrix.minimize_and_pad(
                    adj_matrix, max_number_of_nodes
                )
                adjacency_matrices.append((reshaped_matrix, nx_graph.number_of_nodes()))
        return adjacency_matrices

    def prepare_data(self, *args, **kwargs):
        super().prepare_data(*args, **kwargs)

        nx_graphs = self.create_graphs()
        adj_matrices = self.nx_to_minimized_padded_adjacency_matrices(nx_graphs)
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = split_dataset_train_val_test(adj_matrices, [0.7, 0.2, 0.1])

        if len(self.val_dataset) == 0 or len(self.train_dataset) == 0:
            self.train_dataset = adj_matrices
            self.val_dataset = adj_matrices
            self.test_dataset = adj_matrices

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = BaseDataModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--num_dataset_graph_permutations",
            dest="num_dataset_graph_permutations",
            default=10,
            type=int,
            help="number of permuted copies of the same graphs to generate in the dataset",
        )
        parser.add_argument(
            "--bfs",
            dest="bfs",
            action="store_true",
            help="reorder nodes in graphs by using BFS",
        )
        return parser


class DiagonalRepresentationGraphDataModule(AdjMatrixDataModule):
    data_name = "DiagRepr"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _adj_batch_to_diagonal(
        self, batch: List[Tuple[torch.Tensor, int]]
    ) -> List[Tuple[torch.Tensor, int]]:
        max_num_nodes = 0
        for m in batch:
            num_nodes = m[1]
            if num_nodes > max_num_nodes:
                max_num_nodes = num_nodes

        diag_represented_batch = []
        for m in batch:
            matrix = m[0]
            num_nodes = m[1]
            diag_represented_matrix = adj_matrix_to_diagonal_representation(
                matrix, num_nodes, max_num_nodes, -1.0
            )
            diag_represented_batch.append((diag_represented_matrix, num_nodes))

        return diag_represented_batch

    def prepare_data(self, *args, **kwargs):
        super().prepare_data(*args, **kwargs)
        self.train_dataset = self._adj_batch_to_diagonal(self.train_dataset)
        self.val_dataset = self._adj_batch_to_diagonal(self.val_dataset)
        self.test_dataset = self._adj_batch_to_diagonal(self.test_dataset)


class SyntheticGraphsDataModule(DiagonalRepresentationGraphDataModule):
    data_name = "synthetic"

    def __init__(self, graph_type: str = "", **kwargs):
        self.graph_type = graph_type
        self.data_name += "_" + graph_type
        super().__init__(**kwargs)

    def create_graphs(self) -> List[nx.Graph]:
        return create_synthetic_graphs(self.graph_type)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = AdjMatrixDataModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--graph_type",
            dest="graph_type",
            default="grid_small",
            type=str,
            help="Type of synthethic graphs",
        )
        return parser
