import torch
import torchmetrics


class MeanReconstructionLoss(torchmetrics.MeanMetric):
    label = "loss_reconstruction"

    def update(
        self,
        loss_reconstruction: torch.Tensor,
        **kwargs,
    ):
        super().update(loss_reconstruction.item())


class MeanEmbeddingsLoss(torchmetrics.MeanMetric):
    label = "loss_embeddings"

    def update(
        self,
        loss_embeddings: torch.Tensor,
        **kwargs,
    ):
        super().update(loss_embeddings.item())


class MeanKLDLoss(torchmetrics.MeanMetric):
    label = "loss_kld"

    def update(
        self,
        loss_kld: torch.Tensor,
        **kwargs,
    ):
        super().update(loss_kld.item())


class MeanClassificationLoss(torchmetrics.MeanMetric):
    label = "loss_classifiaction"

    def update(
        self,
        loss_classification: torch.Tensor,
        **kwargs,
    ):
        super().update(loss_classification.item())
