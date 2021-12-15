from typing import List, Tuple
from argparse import ArgumentParser
import networkx as nx
import numpy as np

import torch

from graph_nn_vae.data.data_module import BaseDataModule
from graph_nn_vae.data.graph_loaders import GraphLoaderBase
from graph_nn_vae.util import adjmatrix, split_dataset_train_val_test
from graph_nn_vae.util.graphs import max_number_of_nodes_in_graphs


class AdjMatrixDataModule(BaseDataModule):
    data_name = "AdjMatrix"

    def __init__(
        self,
        data_loader: GraphLoaderBase,
        num_dataset_graph_permutations: int,
        bfs: bool = False,
        use_labels: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_dataset_graph_permutations = num_dataset_graph_permutations
        self.bfs = bfs
        self.data_loader = data_loader
        self.use_labels = use_labels

        self.prepare_data()

    def create_graphs(self) -> Tuple[List[nx.Graph], List[int]]:
        return self.data_loader.load_graphs()

    def max_number_of_nodes_in_graphs(self, graphs: List[nx.Graph]) -> int:
        max_number_of_nodes = 0
        for graph in graphs:
            if graph.number_of_nodes() > max_number_of_nodes:
                max_number_of_nodes = graph.number_of_nodes()
        return max_number_of_nodes

    def nx_to_minimized_padded_adjacency_matrices(
        self,
        nx_graphs: List[nx.Graph],
        remove_duplicates: bool = True,
        labels: List[int] = None,
    ) -> List[Tuple[torch.Tensor, int]]:
        """
        Returns tuples of adj matrices with number of nodes.
        """
        max_number_of_nodes = max_number_of_nodes_in_graphs(nx_graphs)

        adjacency_matrices = []
        adjacency_matrices_labels = [] if labels is not None else None

        for index, nx_graph in enumerate(nx_graphs):
            np_adj_matrix = nx.to_numpy_array(
                nx_graph, dtype=np.float32
            )  # TODO move to_numpy_array to create graphs
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
                if labels is not None:
                    adjacency_matrices_labels.append(labels[index])

        if remove_duplicates:
            return adjmatrix.remove_duplicates(
                adjacency_matrices, adjacency_matrices_labels
            )
        else:
            return adjacency_matrices, adjacency_matrices_labels

    def prepare_data(self, *args, **kwargs):
        super().prepare_data(*args, **kwargs)

        nx_graphs, graph_labels = self.create_graphs()
        adj_matrices, graph_labels = self.nx_to_minimized_padded_adjacency_matrices(
            nx_graphs, labels=graph_labels
        )

        if self.use_labels and graph_labels is None:
            raise RuntimeError(
                f"If you want to use labels (flag --use_labels), provide them."
            )

        graph_data = (
            list(zip(adj_matrices, graph_labels)) if self.use_labels else adj_matrices
        )

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = split_dataset_train_val_test(graph_data, [0.7, 0.2, 0.1])

        if len(self.val_dataset) == 0 or len(self.train_dataset) == 0:
            self.train_dataset = graph_data
            self.val_dataset = graph_data
            self.test_dataset = graph_data

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
        parser.add_argument(
            "--use_labels",
            dest="use_labels",
            action="store_true",
            help="use graph labels",
        )

        return parser
