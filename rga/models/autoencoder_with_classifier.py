from argparse import ArgumentParser
from typing import IO, Callable, Dict, List, Optional, Union

import torch
from torch import Tensor
import torchmetrics
from rga.models.autoencoder_base import RecursiveGraphAutoencoder
from rga.models.classifier_components import MLPClassifier


class RecursiveGraphAutoencoderWithClassifier(RecursiveGraphAutoencoder):
    model_name = "RecursiveGraphAutoencoderWithClassifier"

    classifier_class = MLPClassifier

    def __init__(self, class_count: int, **kwargs):
        super().__init__(**kwargs)
        self.classifier = self.classifier_class(class_count=class_count, **kwargs)
        if class_count == 2:
            self.classification_loss = torch.nn.BCELoss()
        else:
            self.classification_loss = torch.nn.CrossEntropyLoss()
        self.classification_loss_weight = 0.0
        self.class_count = class_count

    def step(self, batch, metrics: List[Callable] = []) -> Tensor:
        y_pred, diagonal_embeddings_norm, prediction_labels = self(batch)

        y_edge, y_mask, y_pred_edge, y_pred_mask = self.adjust_y_to_prediction(
            batch, y_pred
        )

        labels = batch[-1] - min(batch[-1])
        loss_classification = self.calc_classification_loss(prediction_labels, labels)

        if self.class_count == 2:
            prediction_labels = torch.round(prediction_labels[:, 0]).int()
        else:
            prediction_labels = torch.argmax(prediction_labels, dim=1)

        loss_reconstruction = self.calc_reconstruction_loss(
            y_edge, y_mask, y_pred_edge, y_pred_mask, batch[2]
        )
        loss_embeddings = (
            diagonal_embeddings_norm * self.diagonal_embeddings_loss_weight
        )
        loss = loss_reconstruction + loss_embeddings + loss_classification

        shared_metric_state = {}
        for metric in metrics:
            if isinstance(metric, torchmetrics.Accuracy):
                metric.update(prediction_labels, labels)
            else:
                metric.update(
                    edges_predicted=y_pred_edge,
                    edges_target=y_edge,
                    mask_predicted=y_pred_mask,
                    mask_target=y_mask,
                    num_nodes=batch[2],
                    loss_reconstruction=loss_reconstruction,
                    loss_embeddings=loss_embeddings,
                    loss_classification=loss_classification,
                    shared_metric_state=shared_metric_state,
                )

        return loss

    def calc_classification_loss(self, predictions, targets) -> Tensor:
        if self.class_count == 2:
            loss = self.classification_loss(predictions[:, 0], targets.float())
        else:
            loss = self.classification_loss(predictions, targets)

        loss *= self.classification_loss_weight
        self.classification_loss_weight += 0.00001
        return loss

    def forward(self, batch: Tensor) -> Tensor:
        num_nodes_batch = batch[2]
        max_num_nodes_in_graph_batch = max(num_nodes_batch)

        graph_embeddings = self.encoder(batch)

        labels = self.classifier(graph_embeddings)

        reconstructed_graph_diagonals, diagonal_embeddings_norm = self.decoder(
            graph_encoding_batch=graph_embeddings,
            max_number_of_nodes=max_num_nodes_in_graph_batch,
        )

        return reconstructed_graph_diagonals, diagonal_embeddings_norm, labels

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, IO],
        map_location: Optional[
            Union[Dict[str, str], str, torch.device, int, Callable]
        ] = None,
        hparams_file: Optional[str] = None,
        **kwargs,
    ):
        classifier_exception = None
        try:
            return super().load_from_checkpoint(
                checkpoint_path, map_location, hparams_file, True, **kwargs
            )
        except Exception as e:
            classifier_exception = e
        try:
            ae_model = RecursiveGraphAutoencoder.load_from_checkpoint(
                checkpoint_path, map_location, hparams_file, True, **kwargs
            )
            m = cls(**kwargs)
            m.encoder = ae_model.encoder
            m.decoder = ae_model.decoder
            return m
        except Exception as e:
            raise Exception(
                f"neither the {cls.__name__} nor the transfer learining {RecursiveGraphAutoencoder.__name__} could be loaded: "
                f"{cls.__name__} exception: {classifier_exception}"
            ) from e
