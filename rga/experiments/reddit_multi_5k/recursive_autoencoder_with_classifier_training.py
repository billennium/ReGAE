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
            recall_to_precision_bias=0.03,
            gradient_clip_val=0.5,
            optimizer="AdamWAMSGrad",
            learning_rate=0.00001,
            lr_monitor=True,
            lr_scheduler_name="NoSched",
            lr_scheduler_metric="loss/train_avg",
            # minimal_subgraph_size=4,
            # subgraph_stride=0.5,
            # subgraph_scheduler_name="edge_metrics_based",
            # subgraph_scheduler_params={
            #     "subgraph_size_initial": 0.05,
            #     "metrics_treshold": 0.1,
            #     "step": 0.1,
            # },
            batch_size=4,
            accumulate_grad_batches=4,
            embedding_size=2036,
            block_size=64,
            encoder_hidden_layer_sizes=[4096],
            encoder_activation_function="ELU",
            decoder_hidden_layer_sizes=[6144],
            decoder_activation_function="ELU",
            metrics=[
                "Accuracy",
                "MeanClassificationLoss",
                "MeanReconstructionLoss",
                "MeanEmbeddingsLoss",
                # "EdgeAccuracy",
                "EdgePrecision",
                "EdgeRecall",
                "EdgeF1",
                # "MaskPrecision",
                # "MaskRecall",
                # "MaxGraphSize",
            ],
            max_steps=10000,
            max_epochs=10000,
            check_val_every_n_epoch=1,
            metric_update_interval=1,
            early_stopping=False,
            bfs=True,
            num_dataset_graph_permutations=1,
            dataset_name="REDDIT-MULTI-5K",
            use_labels=True,
            class_count=5,
            classifier_hidden_layer_sizes=[4096,2048,16],
            classifier_activation_function="ReLU",
            classifier_dropout=0,
            # pickled_dataset_path="datasets/imdb_multi_labels.pkl",
            checkpoint_monitor="Accuracy/val",
            precision=16,
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
