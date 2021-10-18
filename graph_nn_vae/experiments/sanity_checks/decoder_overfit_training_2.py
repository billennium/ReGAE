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
        embedding_size = 2
        super().__init__(embedding_size=embedding_size, **kwargs)

    def step(self, adj_with_codes_batch: Tensor) -> Tensor:
        adj_batch = adj_with_codes_batch[0]
        graph_embdeddings = adj_with_codes_batch[1]
        reconstructed_graph_diagonals = self.forward(graph_embdeddings)
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
            max_number_of_nodes=20,
            learning_rate=0.0001,
            check_val_every_n_epoch=100,
        )
        return parser


class SyntheticGraphsCodedDataModule(BaseDataModule):
    data_name = "synthethic"
    pad_sequence = False
    adjacency_matrices = []

    def __init__(self, graph_type: str, **kwargs):
        super().__init__(**kwargs)
        self.graph_type = graph_type
        self.data_name += "_coded_" + graph_type
        self._prepare_data()

    def prepare_data(self, *args, **kwargs):
        super().prepare_data(*args, **kwargs)

    def _prepare_data(self):
        nx_graphs = create_synthetic_graphs(self.graph_type)
        max_number_of_nodes = 0
        for graph in nx_graphs:
            if graph.number_of_nodes() > max_number_of_nodes:
                max_number_of_nodes = graph.number_of_nodes()

        self.adjacency_matrices = []
        for code_idx, nx_graph in enumerate(nx_graphs):
            adj_matrix = nx.to_numpy_array(nx_graph, dtype=np.float32)
            reshaped_matrix = adjmatrix.minimize_and_pad(
                adj_matrix, max_number_of_nodes
            )
            # graph embedding has to have at least 2 values, hence the 0.0
            graph_code = torch.FloatTensor([code_idx, 0.0])
            self.adjacency_matrices.append((reshaped_matrix, graph_code))

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = split_dataset_train_val_test(self.adjacency_matrices, [0.8, 0.1, 0.1])

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = BaseDataModule.add_model_specific_args(parent_parser)
        parser.set_defaults(
            graph_type="grid_small",
        )
        return parser


if __name__ == "__main__":
    Experiment(OverfitDecoder, SyntheticGraphsCodedDataModule).run()
