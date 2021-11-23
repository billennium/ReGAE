import torch
import torchmetrics


class EdgeAccuracy(torchmetrics.Accuracy):
    label = "edge_accuracy"

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def update(
        self,
        edges_predicted: torch.Tensor, edges_target: torch.Tensor,
        mask_predicted: torch.Tensor, mask_target: torch.Tensor
    ):
        edges_predicted = edges_predicted.sigmoid().round().int()
        edges_target = edges_target.int()

        mask_predicted = mask_predicted.sigmoid().round().int()
        mask_target = mask_target.int()

        mask = (mask_target == 1) + (mask_predicted == 1)

        super().update(edges_predicted[mask] + mask_predicted[mask] * 2, edges_target[mask] + mask_target[mask] * 2)

