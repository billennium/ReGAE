from argparse import ArgumentParser
from typing import Callable, List, Tuple

import torch
from torch import Tensor

from graph_nn_vae.models.base import BaseModel
from graph_nn_vae.models.autoencoder_components import GraphEncoder
from graph_nn_vae.models.classifier_components import MLPClassifier
from graph_nn_vae.models.edge_encoders.memory_standard import MemoryEdgeEncoder


class GraphClassifierBase(BaseModel):
    def __init__(
        self,
        class_count: int,
        **kwargs,
    ):
        super(GraphClassifierBase, self).__init__(**kwargs)
        self.class_count = class_count

    def step(self, batch, metrics: List[Callable] = []) -> Tensor:
        y_pred = self(batch[:-1])
        labels = batch[-1]

        if self.class_count == 2:
            loss = self.loss_function(y_pred[:, 0], labels.float())
            y_pred_labels = torch.round(y_pred[:, 0]).int()
        else:
            loss = self.loss_function(y_pred, labels)
            y_pred_labels = torch.argmax(y_pred, dim=1)

        for metric in metrics:
            metric(y_pred_labels, labels)

        return loss

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = BaseModel.add_model_specific_args(parent_parser=parser)
        return parser


class RecurrentEncoderGraphClassifier(GraphClassifierBase):
    model_name = "RecurrentEncoderGraphClassifier"

    graph_encoder_class = GraphEncoder
    edge_encoder_class = MemoryEdgeEncoder
    classifier_network_class = MLPClassifier

    def __init__(
        self, freeze_encoder: bool = False, checkpoint_path: str = "", **kwargs
    ):
        super(RecurrentEncoderGraphClassifier, self).__init__(**kwargs)

        self.encoder = self.graph_encoder_class(self.edge_encoder_class, **kwargs)
        self.classifier_network = self.classifier_network_class(**kwargs)
        self.freeze_encoder = freeze_encoder

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            encoder_checkpoint = {
                k.replace("encoder.edge_encoder.", "edge_encoder."): v
                for (k, v) in checkpoint["state_dict"].items()
                if "encoder" in k
            }
            self.encoder.load_state_dict(encoder_checkpoint)

    def forward(self, batch: Tensor) -> Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                graph_embdeddings = self.encoder(batch)
        else:
            graph_embdeddings = self.encoder(batch)
        predictions = self.classifier_network(graph_embdeddings)

        return predictions

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parent_parser = GraphClassifierBase.add_model_specific_args(
            parent_parser=parent_parser
        )
        parser = cls.classifier_network_class.add_model_specific_args(
            parent_parser=parent_parser
        )
        parser = cls.graph_encoder_class.add_model_specific_args(
            parent_parser=parent_parser
        )
        parser = cls.edge_encoder_class.add_model_specific_args(
            parent_parser=parent_parser
        )
        parser = parent_parser.add_argument_group(cls.__name__)

        parser.add_argument(
            "--freeze_encoder",
            dest="freeze_encoder",
            action="store_true",
            help="freeze encoder part",
        )
        parser.add_argument(
            "--checkpoint_path",
            dest="checkpoint_path",
            default="",
            type=str,
            help="path to encoder checkpoint",
        )
        return parent_parser
