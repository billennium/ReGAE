from argparse import ArgumentParser

from rga.experiments.experiment import Experiment
from rga.experiments.decorators import add_graphloader_args
from rga.data import (
    DiagonalRepresentationGraphDataModule,
    SyntheticGraphLoader,
)
from rga.models.autoencoder_base import RecursiveGraphAutoencoder


class ExperimentModel(RecursiveGraphAutoencoder):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = RecursiveGraphAutoencoder.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="BCEWithLogits",
            mask_loss_function="BCEWithLogits",
            mask_loss_weight=0.5,
            diagonal_embeddings_loss_weight=0.2,
            recall_to_precision_bias=0.3,
            optimizer="AdamWAMSGrad",
            lr_monitor=True,
            lr_scheduler_name="NoSched",
            lr_scheduler_metric="loss/train_avg",
            learning_rate=0.0003,
            gradient_clip_val=1.0,
            batch_size=32,
            embedding_size=256,
            encoder_hidden_layer_sizes=[2048],
            encoder_activation_function="ELU",
            decoder_hidden_layer_sizes=[2048],
            decoder_activation_function="ELU",
            block_size=4,
            metrics=[
                "EdgeAccuracy",
                "EdgePrecision",
                "EdgeRecall",
                "EdgeF1",
                "MaskPrecision",
                "MaskRecall",
                "MeanReconstructionLoss",
                "MeanEmbeddingsLoss",
            ],
            max_epochs=10000,
            check_val_every_n_epoch=3,
            metric_update_interval=1,
            early_stopping=True,
            bfs=True,
            num_dataset_graph_permutations=10,
            graph_type="grid_medium",
        )
        return parser


@add_graphloader_args
class ExperimentDataModule(DiagonalRepresentationGraphDataModule):
    graphloader_class = SyntheticGraphLoader


if __name__ == "__main__":
    Experiment(ExperimentModel, ExperimentDataModule).run()
