import torch
import torchmetrics


class MeanReconstructionLoss(torchmetrics.MeanMetric):
    label = "loss_reconstruction"

    def update(
        self,
        loss_reconstruction: torch.Tensor,
        **kwargs,
    ):
        super().update(loss_reconstruction)


class MeanEmbeddingsLoss(torchmetrics.MeanMetric):
    label = "loss_embeddings"

    def update(
        self,
        loss_embeddings: torch.Tensor,
        **kwargs,
    ):
        super().update(loss_embeddings)


class MeanKLDLoss(torchmetrics.MeanMetric):
    label = "loss_kld"

    def update(
        self,
        loss_kld: torch.Tensor,
        **kwargs,
    ):
        super().update(loss_kld)


class MeanClassificationLoss(torchmetrics.MeanMetric):
    label = "loss_clasifiaction"

    def update(
        self,
        loss_classification: torch.Tensor,
        **kwargs,
    ):
        super().update(loss_classification)
