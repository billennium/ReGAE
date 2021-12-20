from argparse import ArgumentParser

from graph_nn_vae.experiments.experiment import Experiment
from graph_nn_vae.data import (
    DiagonalRepresentationGraphDataModule,
    RealGraphLoader,
)
from graph_nn_vae.models.classifier_base import RecurrentEncoderGraphClassifier


class GraphClassifier(RecurrentEncoderGraphClassifier):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = RecurrentEncoderGraphClassifier.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="BCE",
            optimizer="AdamWAMSGrad",
            lr_monitor=True,
            lr_scheduler_name="NoSched",
            lr_scheduler_metric="loss/train_avg",
            learning_rate=0.0002,
            gradient_clip_val=1.0,
            batch_size=32,
            embedding_size=64,
            edge_size=1,
            encoder_hidden_layer_sizes=[256, 128],
            encoder_activation_function="ELU",
            classifier_hidden_layer_sizes=[128],
            classifier_activation_function="ELU",
            class_count=2,
            classifier_dropout=0.6,
            metrics=[
                "Accuracy",
                "F1",
                # "Precision",
                # "Recall",
            ],
            max_epochs=200,
            check_val_every_n_epoch=1,
            metric_update_interval=1,
            early_stopping=True,
            early_stopping_patience=20,
            bfs=True,
            num_dataset_graph_permutations=1,
            datasets_dir="/home/adam/phd/recurrent-graph-autoencoder/data",
            dataset_name="IMDB-BINARY",
            use_labels=True,
            workers=0,
            gpus=1,
            freeze_encoder=True,
            checkpoint_path="/home/adam/phd/recurrent-graph-autoencoder/tb_logs_old/RecurrentGraphAutoencoder/IMDB-BINARY/version_2/epoch=119-step=2039.ckpt",
        )
        return parser


if __name__ == "__main__":
    Experiment(
        GraphClassifier, DiagonalRepresentationGraphDataModule, RealGraphLoader
    ).run()
