from argparse import ArgumentParser

from graph_nn_vae.data import SyntheticGraphsDataModule
from graph_nn_vae.experiments.experiment import Experiment
from graph_nn_vae.models.autoencoder_components import GraphEncoder


class NumEdgesEncoder(GraphEncoder):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = GraphEncoder.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="MSE",
            optimizer="Adam",
            embedding_size=1,
            batch_size=32,
        )
        return parser


if __name__ == "__main__":
    Experiment(NumEdgesEncoder, SyntheticGraphsDataModule).run()
