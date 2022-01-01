from argparse import ArgumentParser

from graph_nn_vae.experiments.experiment import Experiment
from graph_nn_vae.experiments.decorators import add_graphloader_args
from graph_nn_vae.data import (
    DiagonalRepresentationGraphDataModule,
    SyntheticGraphLoader,
)
from graph_nn_vae.models.autoencoder_base import RecurrentGraphAutoencoder


class ExperimentModel(RecurrentGraphAutoencoder):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = RecurrentGraphAutoencoder.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="BCEWithLogits",
            mask_loss_function="BCEWithLogits",
            mask_loss_weight=0.5,
            diagonal_embeddings_loss_weight=0.2,
            optimizer="AdamWAMSGrad",
            lr_monitor=True,
            lr_scheduler_name="NoSched",
            lr_scheduler_metric="loss/train_avg",
            learning_rate=0.0025,
            gradient_clip_val=2.0,
            batch_size=32,
            embedding_size=128,
            encoder_hidden_layer_sizes=[1024],
            encoder_activation_function="ELU",
            decoder_hidden_layer_sizes=[1024],
            decoder_activation_function="ELU",
            block_size=3,
            metrics=[
                "EdgePrecision",
                "EdgeRecall",
                "MaskPrecision",
                "MaskRecall",
            ],
            max_number_of_nodes=16,
            max_epochs=10000,
            check_val_every_n_epoch=20,
            metric_update_interval=20,
            early_stopping=False,
            bfs=True,
            num_dataset_graph_permutations=10,
            graph_type="grid_small",
        )
        return parser


@add_graphloader_args
class ExperimentDataModule(DiagonalRepresentationGraphDataModule):
    graphloader_class = SyntheticGraphLoader


if __name__ == "__main__":
    Experiment(ExperimentModel, ExperimentDataModule).run()
