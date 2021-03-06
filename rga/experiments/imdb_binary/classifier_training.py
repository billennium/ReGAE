from argparse import ArgumentParser

from rga.experiments.experiment import Experiment
from rga.experiments.decorators import add_graphloader_args
from rga.data import (
    DiagonalRepresentationGraphDataModule,
    RealGraphLoader,
)
from rga.models.classifier_base import RecursiveEncoderGraphClassifier


class ExperimentModel(RecursiveEncoderGraphClassifier):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = RecursiveEncoderGraphClassifier.add_model_specific_args(parent_parser)
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
            datasets_dir="",
            dataset_name="IMDB-BINARY",
            use_labels=True,
            workers=0,
            gpus=1,
            freeze_encoder=True,
            checkpoint_path="",
        )
        return parser


@add_graphloader_args
class ExperimentDataModule(DiagonalRepresentationGraphDataModule):
    graphloader_class = RealGraphLoader


if __name__ == "__main__":
    Experiment(ExperimentModel, ExperimentDataModule).run()
