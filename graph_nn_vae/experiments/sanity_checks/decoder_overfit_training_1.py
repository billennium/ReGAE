from argparse import ArgumentParser
from random import gammavariate

import torch
from torch import Tensor

from graph_nn_vae.data import SyntheticGraphsDataModule
from graph_nn_vae.experiments.experiment import Experiment
from graph_nn_vae.models.autoencoder_components import GraphDecoder
from graph_nn_vae import util


class OverfitDecoder(GraphDecoder):
    def __init__(self, embedding_size: int, max_number_of_nodes: int, **kwargs):
        self.max_number_of_nodes = max_number_of_nodes
        embedding_size = 2 * max_number_of_nodes
        super().__init__(
            embedding_size=embedding_size,
            max_number_of_nodes=max_number_of_nodes,
            **kwargs
        )

    def step(self, adjacency_matrices_batch: Tensor) -> Tensor:
        graph_embdeddings = self._hash_adjacency_matrices(adjacency_matrices_batch)
        reconstructed_graph_diagonals = self.forward(graph_embdeddings)
        loss_f = torch.nn.MSELoss()
        loss = util.get_reconstruction_loss(
            adjacency_matrices_batch,
            reconstructed_graph_diagonals,
            loss_f,
        )
        return loss

    def _hash_adjacency_matrices(self, adjacency_matrices_batch: Tensor):
        graph_hashes = []
        for adjacency_matrix in adjacency_matrices_batch:
            y_edge_num_sums = [torch.sum(row) for row in adjacency_matrix]
            x_edge_num_sums = [
                torch.sum(col) for col in torch.rot90(adjacency_matrix, -1, [0, 1])
            ]
            graph_hash = torch.stack((*y_edge_num_sums, *x_edge_num_sums))
            pad_length = self.max_number_of_nodes * 2 - graph_hash.shape[0]
            graph_hash = torch.nn.functional.pad(graph_hash, [0, pad_length])
            graph_hashes.append(graph_hash)
        return torch.stack(graph_hashes)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = GraphDecoder.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="MSE",
            optimizer="Adam",
            batch_size=32,
            max_number_of_nodes=20,
            learning_rate=0.002,
            check_val_every_n_epoch=10,
        )
        return parser


class NonPermutatedGridGraphsDataModule(SyntheticGraphsDataModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = SyntheticGraphsDataModule.add_model_specific_args(parent_parser)
        parser.set_defaults(
            num_dataset_graph_permutations="4",
            graph_type="grid_small",
        )
        return parser


if __name__ == "__main__":
    Experiment(OverfitDecoder, NonPermutatedGridGraphsDataModule).run()
