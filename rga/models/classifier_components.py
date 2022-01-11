from argparse import ArgumentParser, ArgumentError
from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from rga.models.base import BaseModel
from rga.models.utils.layers import (
    parse_layer_sizes_list,
    sequential_from_layer_sizes,
)
from rga.models.utils.getters import get_activation_function


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

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        try:
            parent_parser = BaseModel.add_model_specific_args(parent_parser)
        except ArgumentError:
            pass
        parser = parent_parser.add_argument_group(cls.__name__)
        try:  # these may collide with an encoder module, but that's fine
            parser = BaseModel.add_model_specific_args(parent_parser=parser)
            parser.add_argument(
                "--embedding_size",
                dest="embedding_size",
                default=32,
                type=int,
                metavar="EMB_SIZE",
                help="size of the encoder output graph embedding",
            )
            parser.add_argument(
                "--edge_size",
                dest="edge_size",
                default=1,
                type=int,
                metavar="EDGE_SIZE",
                help="number of dimensions of a graph's edge",
            )
        except ArgumentError:
            pass
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
        return parent_parser
