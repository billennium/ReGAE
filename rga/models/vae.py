from typing import List, Tuple, Callable
from argparse import ArgumentParser

import torch
from torch import Tensor
from torch import nn
import torchmetrics

from rga.models.autoencoder_base import RecursiveGraphAutoencoder


class RecursiveGraphVAE(RecursiveGraphAutoencoder):
    model_name = "RecursiveGraphVAE"

    def __init__(self, kld_loss_weight: float, **kwargs):
        super(RecursiveGraphVAE, self).__init__(**kwargs)

        self.nn_log_var = nn.Linear(
            self.encoder.embedding_size, self.encoder.embedding_size
        )
        self.kld_loss_weight = kld_loss_weight

    def step(self, batch, metrics: List[Callable] = []) -> Tensor:
        y_pred, mu, log_var, diagonal_embeddings_norm, prediction_labels = self(batch)
        y_edge, y_mask, y_pred_edge, y_pred_mask = self.adjust_y_to_prediction(
            batch, y_pred
        )

        labels = batch[-1] - 1
        loss_classification = torch.nn.CrossEntropyLoss()(prediction_labels, labels)

        loss_reconstruction = self.calc_reconstruction_loss(
            y_edge, y_mask, y_pred_edge, y_pred_mask, batch[2]
        )
        loss_embeddings = (
            diagonal_embeddings_norm * self.diagonal_embeddings_loss_weight
        )
        loss_kld = self.calc_kld_loss(mu, log_var)
        # if self.trainer.current_epoch > 650:
        #     self.loss_kld_weight = min(0.004, self.loss_kld_weight * 1.003)

        loss = (
            loss_reconstruction
            + loss_embeddings
            + loss_kld * self.kld_loss_weight
            + loss_classification * 1
        )

        for metric in metrics:
            if isinstance(metric, torchmetrics.Accuracy):
                metric.update(prediction_labels, labels)
            else:
                metric(
                    edges_predicted=y_pred_edge,
                    edges_target=y_edge,
                    mask_predicted=y_pred_mask,
                    mask_target=y_mask,
                    num_nodes=batch[2],
                    loss_reconstruction=loss_reconstruction,
                    loss_embeddings=loss_embeddings,
                    loss_classification=loss_classification,
                    loss_kld=loss_kld,
                )
        self.log("loss_kld_weight", self.kld_loss_weight, on_step=False, on_epoch=True)

        return loss

    def forward(self, batch: Tensor) -> Tensor:
        num_nodes_batch = batch[2]
        max_num_nodes_in_graph_batch = max(num_nodes_batch)

        raw_graph_embdeddings = self.encoder(batch)

        labels = self.classifier(raw_graph_embdeddings)

        log_var = self.nn_log_var(raw_graph_embdeddings)
        graph_embeddings = self.reparameterize(
            mu=raw_graph_embdeddings, log_var=log_var
        )

        reconstructed_graph_diagonals, diagonal_embeddings_norm = self.decoder(
            graph_encoding_batch=graph_embeddings,
            max_number_of_nodes=max_num_nodes_in_graph_batch,
        )

        return (
            reconstructed_graph_diagonals,
            raw_graph_embdeddings,
            log_var,
            diagonal_embeddings_norm,
            labels,
        )

    def distribute_latent_gaussian(self, x_hat: Tensor) -> Tuple[Tensor, Tensor]:
        return self.nn_mu(x_hat), self.nn_log_var(x_hat)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: Mean of the latent Gaussian
        :param log_var: Standard deviation of the latent Gaussian
        :return: Reparametrized Tensor of original shape.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def calc_kld_loss(self, mu: Tensor, log_var: Tensor) -> Tensor:
        in_batch_dims = tuple(range(1, mu.ndim))
        return torch.mean(
            0.5 * torch.sum(log_var.exp() + mu ** 2 - 1 - log_var, dim=in_batch_dims),
            dim=0,
        )

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parent_parser = RecursiveGraphAutoencoder.add_model_specific_args(
            parent_parser=parent_parser
        )
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--kld_loss_weight",
            dest="kld_loss_weight",
            default=0.0001,
            type=float,
            metavar="LOSS_WEIGHT",
            help="weight of the Kullback-Leibler divergence loss",
        )
        return parent_parser
