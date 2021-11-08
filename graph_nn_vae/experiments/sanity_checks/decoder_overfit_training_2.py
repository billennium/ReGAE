from argparse import ArgumentParser
import networkx as nx
import numpy as np

import torch
from torch import Tensor

from graph_nn_vae.data import SyntheticGraphsDataModule
from graph_nn_vae.experiments.experiment import Experiment
from graph_nn_vae.models.autoencoder_components import GraphDecoder
from graph_nn_vae import util
from graph_nn_vae.util import adjmatrix, split_dataset_train_val_test
from graph_nn_vae.data.data_module import BaseDataModule
from graph_nn_vae.data.synthetic_graphs_create import create_synthetic_graphs


class OverfitDecoder(GraphDecoder):
    def __init__(self, embedding_size: int, **kwargs):
        super().__init__(embedding_size=embedding_size, **kwargs)
        self.input_adapter_layer = torch.nn.Sequential(
            torch.nn.Linear(1, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, embedding_size),
        )

    def step(self, adj_with_codes_batch: Tensor) -> Tensor:
        adj_batch = adj_with_codes_batch[0]
        graph_codes = adj_with_codes_batch[1]
        graph_embeddings = self.input_adapter_layer(graph_codes)
        reconstructed_graph_diagonals = self.forward(graph_embeddings)
        loss_f = torch.nn.MSELoss()
        loss = util.get_reconstruction_loss(
            adj_batch,
            reconstructed_graph_diagonals,
            loss_f,
        )
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = GraphDecoder.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="MSE",
            optimizer="Adam",
            batch_size=32,
            learning_rate=0.0005,
            gradient_clip_val=0.01,
            max_epochs=100000,
            check_val_every_n_epoch=100,
            embedding_size=64,
            max_number_of_nodes=17,
        )
        return parser


class SyntheticGraphsCodedDataModule(BaseDataModule):
    data_name = "synthethic"
    pad_sequence = False
    adjacency_matrices = []

    def __init__(self, graph_type: str, num_dataset_graph_permutations: int, **kwargs):
        super().__init__(**kwargs)
        self.graph_type = graph_type
        self.num_dataset_graph_permutations = num_dataset_graph_permutations
        self.data_name += "_coded_" + graph_type
        self._prepare_data()

    def prepare_data(self, *args, **kwargs):
        super().prepare_data(*args, **kwargs)

    def _prepare_data(self):
        # nx_graphs = create_synthetic_graphs(self.graph_type)
        nx_graphs = [
            nx.grid_2d_graph(2, 3),
            nx.grid_2d_graph(3, 2),
            nx.grid_2d_graph(2, 2),
            nx.grid_2d_graph(3, 3),
            nx.grid_2d_graph(4, 3),
            nx.grid_2d_graph(3, 4),
            nx.grid_2d_graph(4, 4),
        ]
        max_number_of_nodes = 0
        for graph in nx_graphs:
            if graph.number_of_nodes() > max_number_of_nodes:
                max_number_of_nodes = graph.number_of_nodes()

        self.adjacency_matrices = []
        for graph_idx, nx_graph in enumerate(nx_graphs):
            np_adj_matrix = nx.to_numpy_array(nx_graph, dtype=np.float32)
            for permutation_idx in range(self.num_dataset_graph_permutations):
                adj_matrix = adjmatrix.random_permute(np_adj_matrix)
                reshaped_matrix = adjmatrix.minimize_and_pad(
                    adj_matrix, max_number_of_nodes
                )
                graph_code = torch.FloatTensor(
                    [graph_idx * self.num_dataset_graph_permutations + permutation_idx]
                )
                self.adjacency_matrices.append((reshaped_matrix, graph_code))

        # self.train_dataset = self.adjacency_matrices
        # self.val_dataset = self.adjacency_matrices
        # self.test_dataset = self.adjacency_matrices
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = split_dataset_train_val_test(self.adjacency_matrices, [0.8, 0.1, 0.1])

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = BaseDataModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--num_dataset_graph_permutations",
            dest="num_dataset_graph_permutations",
            default=4,
            type=int,
            help="number of permuted copies of the same graphs to generate in the dataset",
        )
        parser.set_defaults(
            graph_type="grid_small",
            num_dataset_graph_permutations=4,
        )
        return parser


if __name__ == "__main__":
    Experiment(OverfitDecoder, SyntheticGraphsCodedDataModule).run()
