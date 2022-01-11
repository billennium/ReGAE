from argparse import ArgumentParser

from rga.experiments.experiment import Experiment
from rga.experiments.decorators import add_graphloader_args
from rga.data import (
    DiagonalRepresentationGraphDataModule,
    RealGraphLoader,
)
from rga.models.autoencoder_base import RecursiveGraphAutoencoder
from rga.util.early_stopping import TimeBasedEarlyStopping


class ExperimentModel(RecursiveGraphAutoencoder):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = RecursiveGraphAutoencoder.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="BCEWithLogits",
            mask_loss_function="BCEWithLogits",
            mask_loss_weight=0.5,
            diagonal_embeddings_loss_weight=0.13,
            recall_to_precision_bias=0.45,
            optimizer="AdamWAMSGrad",
            lr_monitor=True,
            lr_scheduler_name="none",
            lr_scheduler_metric="loss/train_avg",
            learning_rate=0.00068,
            gradient_clip_val=1.0,
            batch_size=64,
            embedding_size=160,
            block_size=4,
            encoder_hidden_layer_sizes=[1024, 768],
            encoder_activation_function="ELU",
            decoder_hidden_layer_sizes=[768, 1024],
            decoder_activation_function="ELU",
            metrics=[
                "EdgePrecision",
                "EdgeRecall",
                "EdgeF1",
                "MaskPrecision",
                "MaskRecall",
                "MaxGraphSize",
                "MeanReconstructionLoss",
                "MeanEmbeddingsLoss",
            ],
            max_epochs=10000,
            check_val_every_n_epoch=3,
            metric_update_interval=3,
            early_stopping=False,
            bfs=True,
            num_dataset_graph_permutations=10,
            train_val_test_permutation_split=[1.0, 0.0, 0.0],
            dataset_name="IMDB-BINARY",
        )
        return parser


@add_graphloader_args
class ExperimentDataModule(DiagonalRepresentationGraphDataModule):
    graphloader_class = RealGraphLoader


if __name__ == "__main__":
    Experiment(ExperimentModel, ExperimentDataModule).run()
