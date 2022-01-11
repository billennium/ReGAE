from argparse import ArgumentParser

from rga.data import SyntheticGraphsDataModule
from rga.experiments.experiment import Experiment
from rga.models.autoencoder_components import GraphEncoder


class NumEdgesEncoder(GraphEncoder):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = GraphEncoder.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="MSE",
            optimizer="Adam",
            embedding_size=1,
            learning_rate=0.0001,
            gradient_clip_val=0.1,
            batch_size=32,
            max_epochs=10000,
            check_val_every_n_epoch=100,
        )
        return parser


if __name__ == "__main__":
    Experiment(NumEdgesEncoder, SyntheticGraphsDataModule).run()
