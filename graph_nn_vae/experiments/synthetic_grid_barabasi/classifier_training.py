import networkx as nx
from graph_nn_vae.data.synthetic_graphs_create import create_synthetic_graphs


from argparse import ArgumentParser

from graph_nn_vae.experiments.experiment import Experiment
from graph_nn_vae.data import (
    GraphLoaderBase,
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
            learning_rate=0.0001,
            gradient_clip_val=1.0,
            batch_size=32,
            embedding_size=16,
            edge_size=1,
            encoder_hidden_layer_sizes=[1024, 768],
            encoder_activation_function="ELU",
            classifier_hidden_layer_sizes=[512, 128],
            classifier_activation_function="ELU",
            class_count=2,
            # classifier_dropout=0.25,
            metrics=[
                "Accuracy",
                "F1",
                # "Precision",
                # "Recall",
            ],
            max_epochs=10000,
            check_val_every_n_epoch=5,
            metric_update_interval=5,
            early_stopping=True,
            bfs=True,
            num_dataset_graph_permutations=1,
            datasets_dir="/home/adam/phd/recurrent-graph-autoencoder/data",
            dataset_name="IMDB-BINARY",
            use_labels=True,
            use_catche=True,
            workers=0,
        )
        return parser


class GridBarabasiClassification(GraphLoaderBase):
    data_name = "grid_barabasi_classification"

    def load_graphs(self):
        grids = create_synthetic_graphs("grid_small")
        barabasi = create_synthetic_graphs("barabasi_small")

        return grids + barabasi, [0] * len(grids) + [1] * len(barabasi)


if __name__ == "__main__":
    Experiment(
        GraphClassifier,
        DiagonalRepresentationGraphDataModule,
        GridBarabasiClassification,
    ).run()
