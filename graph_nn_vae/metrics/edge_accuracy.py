import torch
import torchmetrics


class EdgeAccuracy(torchmetrics.Accuracy):
    label = "edge_accuracy"

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def update(
        self,
        edges_predicted: torch.Tensor,
        edges_target: torch.Tensor,
        mask_predicted: torch.Tensor,
        mask_target: torch.Tensor,
        **kwargs,
    ):
        edges_predicted = edges_predicted.sigmoid().round().int()
        edges_target = torch.clamp(edges_target.int(), min=0)

        mask_predicted = mask_predicted.sigmoid().round().int()
        mask_target = mask_target.int()

        mask = (mask_target == 1) + (mask_predicted == 1)

        super().update(
            edges_predicted[mask] + mask_predicted[mask] * 2,
            edges_target[mask] + mask_target[mask] * 2,
        )


class MaskAccuracy(torchmetrics.Accuracy):
    label = "mask_accuracy"

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def update(
        self,
        mask_predicted: torch.Tensor,
        mask_target: torch.Tensor,
        **kwargs,
    ):
        mask_predicted = torch.sigmoid(mask_predicted)
        mask_target = torch.clamp(mask_target.int(), min=0)
        mask = mask_target == 1
        mask_predicted = torch.round(mask_predicted[mask]).int()
        super().update(mask_predicted, mask_target[mask])
