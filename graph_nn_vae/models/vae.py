from typing import List, Tuple, Callable
from argparse import ArgumentParser

import torch
from torch import Tensor
from torch import nn

from graph_nn_vae.models.autoencoder_base import RecurrentGraphAutoencoder


class RecurrentGraphVAE(RecurrentGraphAutoencoder):
    model_name = ""

    def __init__(self, **kwargs):
        super(RecurrentGraphVAE, self).__init__(**kwargs)

        # latent_size = 1024
        # self.nn_mu = nn.Linear(self.encoder.embedding_size, self.encoder.embedding_size)
        self.nn_log_var = nn.Linear(
            self.encoder.embedding_size, self.encoder.embedding_size
        )
        # self.nn_log_var.bias.data.fill_(-5)
        # self.nn_log_var.weight.data.fill_(0)
        # self.nn_latent_to_embeddings = nn.Linear(
        #     latent_size, self.encoder.embedding_size
        # )
        self.loss_kld_weight = 0.000000001
        # self.loss_kld_weight_step = 0.00001

    def step(self, batch, metrics: List[Callable] = []) -> Tensor:
        if not self.is_with_graph_mask:
            return super().step(batch, metrics)

        # y_pred, diagonal_embeddings_norm = self(batch)
        y_pred, mu, log_var, diagonal_embeddings_norm = self(batch)
        y_edge, y_mask, y_pred_edge, y_pred_mask = self.adjust_y_to_prediction(
            batch, y_pred
        )

        loss_reconstruction = self.calc_reconstruction_loss(
            y_edge, y_mask, y_pred_edge, y_pred_mask
        )
        loss_embeddings = (
            diagonal_embeddings_norm * self.diagonal_embeddings_loss_weight
        )

        loss_kld = self.calc_kld_loss(mu, log_var)
        # if self.trainer.current_epoch > 650:
        self.loss_kld_weight = min(0.004, self.loss_kld_weight * 1.003)

        self.log("loss_kld_weight", self.loss_kld_weight, on_step=False, on_epoch=True)

        loss = loss_reconstruction + loss_embeddings + loss_kld * self.loss_kld_weight

        for metric in metrics:
            metric(
                edges_predicted=y_pred_edge,
                edges_target=y_edge,
                mask_predicted=y_pred_mask,
                mask_target=y_mask,
                num_nodes=batch[2],
                loss_reconstruction=loss_reconstruction,
                loss_embeddings=loss_embeddings,
                loss_kld=loss_kld,
            )

        return loss

    def forward(self, batch: Tensor) -> Tensor:
        num_nodes_batch = batch[2]
        max_num_nodes_in_graph_batch = max(num_nodes_batch)

        raw_graph_embdeddings = self.encoder(batch)

        # if self.trainer.current_epoch < 650:
        #     graph_embeddings = raw_graph_embdeddings
        #     log_var = None
        # else:
        log_var = self.nn_log_var(raw_graph_embdeddings)
        graph_embeddings = self.reparameterize(
            mu=raw_graph_embdeddings, log_var=log_var
        )

        # graph_embeddings = self.nn_latent_to_embeddings(latent_space_embeddings)

        reconstructed_graph_diagonals, diagonal_embeddings_norm = self.decoder(
            graph_encoding_batch=graph_embeddings,
            max_number_of_nodes=max_num_nodes_in_graph_batch,
        )

        return (
            reconstructed_graph_diagonals,
            raw_graph_embdeddings,
            log_var,
            diagonal_embeddings_norm,
        )
        # return reconstructed_graph_diagonals, diagonal_embeddings_norm

    def distribute_latent_gaussian(self, x_hat: Tensor) -> Tuple[Tensor, Tensor]:
        return self.nn_mu(x_hat), self.nn_log_var(x_hat)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: Mean of the latent Gaussian
        :param log_var: Standard deviation of the latent Gaussian
        :return: Reparametrized Tensor of original shape.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def calc_kld_loss(self, mu: Tensor, log_var: Tensor) -> Tensor:
        in_batch_dims = tuple(range(1, mu.ndim))
        return torch.mean(
            0.5 * torch.sum(log_var.exp() + mu ** 2 - 1 - log_var, dim=in_batch_dims),
            dim=0,
        )

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = RecurrentGraphAutoencoder.add_model_specific_args(parent_parser=parser)
        return parser
