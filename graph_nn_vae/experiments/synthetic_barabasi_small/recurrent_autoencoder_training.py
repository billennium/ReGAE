from argparse import ArgumentParser

from graph_nn_vae.experiments.experiment import Experiment
from graph_nn_vae.experiments.decorators import add_dataloader_args
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
            learning_rate=0.001,
            gradient_clip_val=1.0,
            batch_size=32,
            embedding_size=128,
            encoder_hidden_layer_sizes=[2048],
            encoder_activation_function="ELU",
            decoder_hidden_layer_sizes=[2048],
            decoder_activation_function="ELU",
            metrics=[
                "EdgePrecision",
                "EdgeRecall",
                "MaskPrecision",
                "MaskRecall",
            ],
            max_number_of_nodes=21,
            max_epochs=30,
            check_val_every_n_epoch=2,
            metric_update_interval=2,
            early_stopping=False,
            bfs=True,
            num_dataset_graph_permutations=10,
            graph_type="barabasi_small",
        )
        return parser


@add_dataloader_args
class ExperimentDataModule(DiagonalRepresentationGraphDataModule):
    dataloader_class = SyntheticGraphLoader


if __name__ == "__main__":
    Experiment(ExperimentModel, ExperimentDataModule).run()
