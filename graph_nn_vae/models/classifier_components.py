from argparse import ArgumentParser, ArgumentError
from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from graph_nn_vae.models.base import BaseModel
from graph_nn_vae.models.utils.layers import (
    parse_layer_sizes_list,
    sequential_from_layer_sizes,
)
from graph_nn_vae.models.utils.getters import get_activation_function


class MLPClassifier(BaseModel):
    def __init__(
        self,
        embedding_size: int,
        class_count: int,
        classifier_hidden_layer_sizes: List[int],
        classifier_activation_function: str,
        classifier_dropout: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        input_size = embedding_size

        activation_f = get_activation_function(classifier_activation_function)

        output_function = nn.Sigmoid if class_count == 2 else nn.Softmax

        self.nn = sequential_from_layer_sizes(
            input_size,
            class_count if class_count != 2 else 1,
            classifier_hidden_layer_sizes,
            activation_f,
            output_function=output_function,
            dropout=classifier_dropout,
        )

    def forward(self, graphs: Tensor) -> Tensor:
        return self.nn(graphs)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--classifier_hidden_layer_sizes",
            dest="classifier_hidden_layer_sizes",
            default=[256],
            type=parse_layer_sizes_list,
            metavar="CLASSIFIER_H_SIZES",
            help="list of the sizes of the clasifier's hidden layers",
        )
        parser.add_argument(
            "--classifier_activation_function",
            dest="classifier_activation_function",
            default="ReLU",
            type=str,
            metavar="ACTIVATION_F_NAME",
            help="name of the activation function of hidden layers",
        )
        parser.add_argument(
            "--classifier_dropout",
            dest="classifier_dropout",
            default=0.25,
            type=float,
            metavar="DROPOUT",
            help="value of dropout between classifier hidden layers",
        )
        return parser