from argparse import ArgumentParser
from typing import Optional
import networkx as nx
import numpy as np

import torch
from torch._C import dtype
from torch.utils.data import TensorDataset

from graph_nn_vae.data.data_module import BaseDataModule
from graph_nn_vae.data.synthetic_graphs_create import create_synthetic_graphs


class SyntheticGraphsDataModule(BaseDataModule):
    data_name = "synthethic"
    pad_sequence = False
    adjacency_matrices = []

    def __init__(self, graph_type: str, **kwargs):
        super().__init__(**kwargs)
        self.graph_type = graph_type
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
            np_adj_matrix = np.tril(np_adj_matrix)
            padding_size = max_number_of_nodes - np_adj_matrix.shape[0]
            padded_matrix = np.pad(
                np_adj_matrix,
                [(padding_size, 0), (0, padding_size)],
                "constant",
                constant_values=0.0,
            )
            # # invert the matrix in the y dim, so that the triangle is in the upper left corner
            # flipped_matrix = np.flip(padded_matrix, 0)
            # torch_matrix = torch.Tensor(flipped_matrix.copy())
            torch_matrix = torch.Tensor(padded_matrix)
            extended_matrix = torch_matrix[:, :, None]
            self.adjacency_matrices.append(extended_matrix)

        self.train_dataset = self.adjacency_matrices
        self.val_dataset = self.adjacency_matrices
        self.test_dataset = self.adjacency_matrices

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

        return parser
