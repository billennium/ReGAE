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
            diagonal_embeddings_loss_weight=0.19,
            diagonal_embeddings_loss_weight=0.30,
            optimizer="AdamWAMSGrad",
            lr_monitor=True,
            lr_scheduler_name="NoSched",
            lr_scheduler_metric="loss/train_avg",
            learning_rate=0.00023,
            gradient_clip_val=1.0,
            batch_size=16,
            embedding_size=256,
            encoder_hidden_layer_sizes=[2048, 1536],
            encoder_activation_function="ELU",
            decoder_hidden_layer_sizes=[768, 1024],
            decoder_activation_function="ELU",
            block_size=6,
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
            max_number_of_nodes=16,
            max_epochs=10000,
            check_val_every_n_epoch=5,
            metric_update_interval=1,
            early_stopping=False,
            bfs=True,
            num_dataset_graph_permutations=10,
            train_val_test_permutation_split=[1.0, 0.0, 0.0],
            graph_type="grid_medium",
            gpus="0,",
            workers=0,
        )
        return parser


@add_graphloader_args
class ExperimentDataModule(DiagonalRepresentationGraphDataModule):
    graphloader_class = SyntheticGraphLoader


if __name__ == "__main__":
    Experiment(ExperimentModel, ExperimentDataModule).run()
