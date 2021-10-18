from argparse import ArgumentParser
import networkx as nx
import numpy as np

from graph_nn_vae.data.data_module import BaseDataModule
from graph_nn_vae.data.synthetic_graphs_create import create_synthetic_graphs
from graph_nn_vae.util import adjmatrix, split_dataset_train_val_test


class SyntheticGraphsDataModule(BaseDataModule):
    data_name = "synthethic"
    pad_sequence = False
    adjacency_matrices = []

    def __init__(self, graph_type: str, num_dataset_graph_permutations: int, **kwargs):
        super().__init__(**kwargs)
        self.graph_type = graph_type
        self.num_dataset_graph_permutations = num_dataset_graph_permutations
        self.data_name += "_" + graph_type
        self.prepare_data()

    def prepare_data(self, *args, **kwargs):
        super().prepare_data(*args, **kwargs)

        nx_graphs = create_synthetic_graphs(self.graph_type)
        max_number_of_nodes = 0
        for graph in nx_graphs:
            if graph.number_of_nodes() > max_number_of_nodes:
                max_number_of_nodes = graph.number_of_nodes()

        self.adjacency_matrices = []
        for nx_graph in nx_graphs:
            np_adj_matrix = nx.to_numpy_array(nx_graph, dtype=np.float32)
            for _ in range(self.num_dataset_graph_permutations):
                adj_matrix = adjmatrix.random_permute(np_adj_matrix)
                reshaped_matrix = adjmatrix.minimize_and_pad(
                    adj_matrix, max_number_of_nodes
                )
                self.adjacency_matrices.append(reshaped_matrix)

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = split_dataset_train_val_test(self.adjacency_matrices, [0.8, 0.1, 0.1])

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = BaseDataModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--graph_type",
            dest="graph_type",
            default="grid_small",
            type=str,
            help="Type of synthethic graphs",
        )
        parser.add_argument(
            "--num_dataset_graph_permutations",
            dest="num_dataset_graph_permutations",
            default=200,
            type=int,
            help="number of permuted copies of the same graphs to generate in the dataset",
        )

        return parser
