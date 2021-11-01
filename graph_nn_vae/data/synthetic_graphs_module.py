from typing import List
from argparse import ArgumentParser
import networkx as nx
import numpy as np

from graph_nn_vae.data.data_module import BaseDataModule
from graph_nn_vae.data.synthetic_graphs_create import create_synthetic_graphs
from graph_nn_vae.util import adjmatrix, split_dataset_train_val_test


class SyntheticGraphsDataModule(BaseDataModule):
    data_name = "synthethic"
    pad_sequence = False

    def __init__(
        self,
        num_dataset_graph_permutations: int,
        preprepared_graphs: List[nx.Graph] = None,
        graph_type: str = "",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.graph_type = graph_type
        self.num_dataset_graph_permutations = num_dataset_graph_permutations
        if preprepared_graphs is None:
            self.data_name += "_" + graph_type
        else:
            self.data_name = "preprepared"
        self.prepare_data(preprepared_graphs=preprepared_graphs)

    def prepare_data(self, preprepared_graphs: List[nx.Graph] = None, *args, **kwargs):
        super().prepare_data(*args, **kwargs)

        nx_graphs = preprepared_graphs
        if nx_graphs is None:
            nx_graphs = create_synthetic_graphs(self.graph_type)
        max_number_of_nodes = 0
        for graph in nx_graphs:
            if graph.number_of_nodes() > max_number_of_nodes:
                max_number_of_nodes = graph.number_of_nodes()

        adjacency_matrices = []
        for nx_graph in nx_graphs:
            np_adj_matrix = nx.to_numpy_array(nx_graph, dtype=np.float32)
            for _ in range(self.num_dataset_graph_permutations):
                adj_matrix = adjmatrix.random_permute(np_adj_matrix)
                reshaped_matrix = adjmatrix.minimize_and_pad(
                    adj_matrix, max_number_of_nodes
                )
                adjacency_matrices.append(reshaped_matrix)

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = split_dataset_train_val_test(adjacency_matrices, [0.8, 0.1, 0.1])
        if len(self.val_dataset) == 0 or len(self.train_dataset) == 0:
            self.train_dataset = adjacency_matrices
            self.val_dataset = adjacency_matrices
            self.test_dataset = adjacency_matrices

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
            default=10,
            type=int,
            help="number of permuted copies of the same graphs to generate in the dataset",
        )

        return parser
