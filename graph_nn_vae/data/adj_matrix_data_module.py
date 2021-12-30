from typing import List, Tuple, Dict
from argparse import ArgumentParser
import networkx as nx
import numpy as np
import pickle
import os
import torch

from tqdm.auto import tqdm

from graph_nn_vae.data.data_module import BaseDataModule
from graph_nn_vae.data.graph_loaders import GraphLoaderBase
from graph_nn_vae.util import adjmatrix, split_dataset_train_val_test
from graph_nn_vae.util.graphs import max_number_of_nodes_in_graphs
from graph_nn_vae.util.convert_size import convert_size
from graph_nn_vae.data.util.print_dataset_statistics import print_dataset_statistics


class AdjMatrixDataModule(BaseDataModule):
    data_name = "AdjMatrix"

    def __init__(
        self,
        data_loader: GraphLoaderBase,
        num_dataset_graph_permutations: int,
        bfs: bool = False,
        use_labels: bool = False,
        save_dataset_to_pickle: str = None,
        pickled_dataset_path: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_dataset_graph_permutations = num_dataset_graph_permutations
        self.bfs = bfs
        self.data_loader = data_loader
        self.use_labels = use_labels
        self.save_dataset_to_pickle = save_dataset_to_pickle
        self.pickled_dataset_path = pickled_dataset_path

        self.prepare_data()

    def create_graphs(self) -> Dict:
        data = self.data_loader.load_graphs()
        return data["graphs"], data.get("labels", None)

    def max_number_of_nodes_in_graphs(self, graphs: List[nx.Graph]) -> int:
        max_number_of_nodes = 0
        for graph in graphs:
            if graph.number_of_nodes() > max_number_of_nodes:
                max_number_of_nodes = graph.number_of_nodes()
        return max_number_of_nodes

    def multiplicate_graphs(
        self,
        graph_data,
        remove_duplicates: bool = True,
    ):
        graphs = [el[0] for el in graph_data] if self.use_labels else graph_data
        labels = [el[1] for el in graph_data] if self.use_labels else None

        multipliacted_graphs = []
        multipliacted_labels = [] if labels else None

        for index, graph in enumerate(graphs):
            for i in range(self.num_dataset_graph_permutations):
                if i != 0:
                    adj_matrix = adjmatrix.random_permute(graph)
                else:
                    adj_matrix = graph

                multipliacted_graphs.append(adj_matrix)
                if labels is not None:
                    multipliacted_labels.append(labels[index])

        if remove_duplicates:
            multipliacted_graphs, multipliacted_labels = adjmatrix.remove_duplicates(
                multipliacted_graphs, multipliacted_labels
            )

        return (
            list(zip(multipliacted_graphs, multipliacted_labels))
            if labels is not None
            else multipliacted_graphs
        )

    def process_adjacency_matrices(
        self,
        graph_data,
        data_set_name: str = "",
    ) -> List[Tuple[torch.Tensor, int]]:
        """
        Returns tuples of adj matrices with number of nodes.
        """
        graphs = [el[0] for el in graph_data] if self.use_labels else graph_data
        labels = [el[1] for el in graph_data] if self.use_labels else None

        adjacency_matrices = []
        adjacency_matrices_labels = [] if labels is not None else None

        for index, graph in enumerate(
            tqdm(graphs, desc="preprocessing " + data_set_name)
        ):
            adj_matrix = graph

            if self.bfs:
                adj_matrix = adjmatrix.bfs_ordering(adj_matrix)

            reshaped_matrix = adjmatrix.minimize_adj_matrix(adj_matrix)
            adjacency_matrices.append((reshaped_matrix, graph.shape[0]))
            if labels is not None:
                adjacency_matrices_labels.append(labels[index])

        return (
            list(zip(adjacency_matrices, adjacency_matrices_labels))
            if adjacency_matrices_labels is not None
            else adjacency_matrices
        )

    def prepare_data(self, *args, **kwargs):
        super().prepare_data(*args, **kwargs)

        if self.pickled_dataset_path:
            self.load_pickled_data()
        else:
            graphs, graph_labels = self.create_graphs()

            if self.use_labels and graph_labels is None:
                raise RuntimeError(
                    f"If you want to use labels (flag --use_labels), provide them."
                )

            graph_data = list(zip(graphs, graph_labels)) if self.use_labels else graphs

            if self.num_dataset_graph_permutations > 1:
                graph_data = self.multiplicate_graphs(graph_data)

            (
                self.train_dataset,
                self.val_dataset,
                self.test_dataset,
            ) = split_dataset_train_val_test(graph_data, [0.7, 0.15, 0.15])

            if len(self.val_dataset) == 0 or len(self.train_dataset) == 0:
                self.train_dataset = graph_data
                self.val_dataset = graph_data
                self.test_dataset = graph_data

            if self.save_dataset_to_pickle:
                self.pickle_dataset()

        print_dataset_statistics(self.train_dataset, "Train dataset", self.use_labels)
        print_dataset_statistics(
            self.val_dataset, "Validation dataset", self.use_labels
        )
        print_dataset_statistics(self.test_dataset, "Test dataset", self.use_labels)

        self.train_dataset = self.process_adjacency_matrices(
            self.train_dataset,
            data_set_name="train set",
        )
        self.val_dataset = self.process_adjacency_matrices(
            self.val_dataset, data_set_name="val set"
        )
        self.test_dataset = self.process_adjacency_matrices(
            self.test_dataset, data_set_name="test set"
        )

    def load_pickled_data(self):
        with open(self.pickled_dataset_path, "rb") as input:
            (
                train_graphs,
                val_graphs,
                test_graphs,
                train_labels,
                val_labels,
                test_labels,
            ) = pickle.load(input)
        if self.use_labels:
            self.train_dataset = list(zip(train_graphs, train_labels))
            self.val_dataset = list(zip(val_graphs, val_labels))
            self.test_dataset = list(zip(test_graphs, test_labels))
        else:
            self.train_dataset = train_graphs
            self.val_dataset = val_graphs
            self.test_dataset = test_graphs

        print("Dataset successfully loaded!")
        print("File path:", self.pickled_dataset_path)

    def pickle_dataset(self):
        with open(self.save_dataset_to_pickle, "wb") as output:
            train_graphs = (
                [el[0] for el in self.train_dataset]
                if self.use_labels
                else self.train_dataset
            )
            train_labels = (
                [el[1] for el in self.train_dataset] if self.use_labels else None
            )
            val_graphs = (
                [el[0] for el in self.val_dataset]
                if self.use_labels
                else self.val_dataset
            )
            val_labels = [el[1] for el in self.val_dataset] if self.use_labels else None
            test_graphs = (
                [el[0] for el in self.test_dataset]
                if self.use_labels
                else self.test_dataset
            )
            test_labels = (
                [el[1] for el in self.test_dataset] if self.use_labels else None
            )

            pickle.dump(
                (
                    train_graphs,
                    val_graphs,
                    test_graphs,
                    train_labels,
                    val_labels,
                    test_labels,
                ),
                output,
            )
        print("Dataset successfully pickled!")
        print("File path:", self.save_dataset_to_pickle)
        print("File size:", convert_size(os.path.getsize(self.save_dataset_to_pickle)))

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
        parser.add_argument(
            "--save_dataset_to_pickle",
            dest="save_dataset_to_pickle",
            default=None,
            type=str,
            help="save dataset to pickle files",
        )
        parser.add_argument(
            "--pickled_dataset_path",
            dest="pickled_dataset_path",
            default=None,
            type=str,
            help="save dataset to pickle files",
        )
        return parser
