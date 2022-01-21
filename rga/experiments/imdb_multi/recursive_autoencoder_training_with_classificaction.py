from argparse import ArgumentParser

from rga.experiments.experiment import Experiment
from rga.experiments.decorators import add_graphloader_args
from rga.data import (
    DiagonalRepresentationGraphDataModule,
    RealGraphLoader,
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
            recall_to_precision_bias=0.5,
            optimizer="AdamWAMSGrad",
            lr_monitor=True,
            lr_scheduler_name="NoSched",
            lr_scheduler_metric="loss/train_avg",
            learning_rate=0.0005,
            gradient_clip_val=1.0,
            batch_size=64,
            embedding_size=104,
            block_size=8,
            encoder_hidden_layer_sizes=[1024],
            encoder_activation_function="ELU",
            decoder_hidden_layer_sizes=[2048],
            decoder_activation_function="ELU",
            metrics=[
                # "EdgeAccuracy",
                "EdgePrecision",
                "EdgeRecall",
                "EdgeF1",
                "MaskPrecision",
                "MaskRecall",
                "MeanReconstructionLoss",
                "MeanEmbeddingsLoss",
                "MeanClassificationLoss",
                "Accuracy",
            ],
            max_epochs=10000,
            check_val_every_n_epoch=3,
            metric_update_interval=3,
            early_stopping=True,
            bfs=True,
            num_dataset_graph_permutations=1,
            train_val_test_permutation_split=[1.0, 0.0, 0.0],
            dataset_name="IMDB-MULTI",
            use_labels=True,
            class_count=3,
            classifier_hidden_layer_sizes=[],
            classifier_activation_function="ReLU",
            classifier_dropout=0,
            gpus="0,",
            workers=0,
            pickled_dataset_path="datasets/imdb_multi_labels.pkl",
        )
        return parser


@add_graphloader_args
class ExperimentDataModule(DiagonalRepresentationGraphDataModule):
    graphloader_class = RealGraphLoader


if __name__ == "__main__":
    Experiment(ExperimentModel, ExperimentDataModule).run()
