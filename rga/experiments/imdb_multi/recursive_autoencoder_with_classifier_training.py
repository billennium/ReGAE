from argparse import ArgumentParser

from rga.experiments.experiment import Experiment
from rga.experiments.decorators import add_graphloader_args
from rga.data import (
    DiagonalRepresentationGraphDataModule,
    RealGraphLoader,
)
from rga.models.autoencoder_with_classifier import (
    RecursiveGraphAutoencoderWithClassifier,
)
from rga.models.autoencoder_base import RecursiveGraphAutoencoder
from rga.models.vae import RecursiveGraphVAE


class ExperimentModel(RecursiveGraphAutoencoderWithClassifier):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = RecursiveGraphAutoencoderWithClassifier.add_model_specific_args(
            parent_parser
        )
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
                # "MaskPrecision",
                # "MaskRecall",
                "MeanReconstructionLoss",
                "MeanEmbeddingsLoss",
                "MeanClassificationLoss",
                # "MeanKLDLoss",
                "Accuracy",
            ],
            max_epochs=10000,
            check_val_every_n_epoch=1,
            metric_update_interval=1,
            early_stopping=True,
            bfs=True,
            num_dataset_graph_permutations=1,
            dataset_name="IMDB-MULTI",
            use_labels=True,
            class_count=3,
            classifier_hidden_layer_sizes=[],
            classifier_activation_function="ReLU",
            classifier_dropout=0,
            gpus="0,",
            workers=0,
            # pickled_dataset_path="datasets/imdb_multi_labels.pkl",
            checkpoint_monitor="loss_classifiaction/val",
            # kld_loss_weight=0.5,
            # load_from_checkpoint_path="./best_checkpoints/IMDB-MULTI/0.ckpt",
            # load_with_hparams_path="./best_checkpoints/IMDB-MULTI/0_hparams.yaml",
        )
        return parser


@add_graphloader_args
class ExperimentDataModule(DiagonalRepresentationGraphDataModule):
    graphloader_class = RealGraphLoader


if __name__ == "__main__":
    Experiment(ExperimentModel, ExperimentDataModule).run()
