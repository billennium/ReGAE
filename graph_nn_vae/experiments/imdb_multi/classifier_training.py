from argparse import ArgumentParser

from graph_nn_vae.experiments.experiment import Experiment
from graph_nn_vae.experiments.decorators import add_graphloader_args
from graph_nn_vae.data import (
    DiagonalRepresentationGraphDataModule,
    RealGraphLoader,
)
from graph_nn_vae.models.classifier_base import RecurrentEncoderGraphClassifier


class ExperimentModel(RecurrentEncoderGraphClassifier):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = RecurrentEncoderGraphClassifier.add_model_specific_args(parent_parser)
        parser.set_defaults(
            loss_function="CrossEntropy",
            optimizer="AdamWAMSGrad",
            lr_monitor=True,
            lr_scheduler_name="NoSched",
            lr_scheduler_metric="loss/train_avg",
            learning_rate=0.0002,
            gradient_clip_val=1.0,
            batch_size=32,
            embedding_size=104,
            edge_size=1,
            block_size=6,
            encoder_hidden_layer_sizes=[1024, 768],
            encoder_activation_function="ELU",
            classifier_hidden_layer_sizes=[1024, 512, 256, 128],
            classifier_activation_function="ELU",
            class_count=3,
            classifier_dropout=0.6,
            metrics=[
                "Accuracy",
                "F1",
                # "Precision",
                # "Recall",
            ],
            max_epochs=1000,
            check_val_every_n_epoch=1,
            metric_update_interval=1,
            early_stopping=False,
            early_stopping_patience=20,
            bfs=True,
            num_dataset_graph_permutations=1,
            dataset_name="IMDB-MULTI",
            use_labels=True,
            workers=0,
            gpus=1,
            freeze_encoder=True,
            checkpoint_path="/home/adam/phd/recurrent-graph-autoencoder/tb_logs/RecurrentGraphAutoencoder/IMDB-MULTI/version_0/checkpoints/epoch=329-step=14189-v1.ckpt",
        )
        return parser


@add_graphloader_args
class ExperimentDataModule(DiagonalRepresentationGraphDataModule):
    graphloader_class = RealGraphLoader


if __name__ == "__main__":
    Experiment(ExperimentModel, ExperimentDataModule).run()
