import argparse
import networkx as nx
from argparse import ArgumentParser

from graph_nn_vae.experiments.experiment import Experiment
from graph_nn_vae.data import SyntheticGraphsDataModule
from graph_nn_vae.models.autoencoder_base import RecurrentGraphAutoencoder


class GraphAutoencoder(RecurrentGraphAutoencoder):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = RecurrentGraphAutoencoder.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="MSE",
            optimizer="Adam",
            embedding_size=256,
            batch_size=32,
            learning_rate=0.00006,
            gradient_clip_val=0.005,
            max_number_of_nodes=20,
            max_epochs=100000,
            check_val_every_n_epoch=100,
        )
        return parser


class PreparedGraphsDataModule(SyntheticGraphsDataModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = SyntheticGraphsDataModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--preprepared_graphs",
            dest="preprepared_graphs",
            default=[
                nx.grid_2d_graph(2, 2),
                nx.grid_2d_graph(3, 2),
                nx.grid_2d_graph(3, 3),
                nx.grid_2d_graph(4, 3),
                nx.grid_2d_graph(4, 4),
            ],
            help=argparse.SUPPRESS,
        )
        parser.set_defaults(
            num_dataset_graph_permutations=4,
        )
        return parser


if __name__ == "__main__":
    Experiment(GraphAutoencoder, PreparedGraphsDataModule).run()
