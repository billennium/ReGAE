import torch
import torchmetrics


class EdgeAccuracy(torchmetrics.Accuracy):
    label = "edge_accuracy"

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def update(
        self,
        edges_predicted: torch.Tensor, edges_target: torch.Tensor,
        mask_predicted: torch.Tensor = None, mask_target: torch.Tensor = None
    ):
        edges_predicted = torch.sigmoid(edges_predicted)
        edges_predicted = torch.round(edges_predicted).int()
        edges_target = edges_target.int()

        if mask_target is not None:
            mask = mask_target == 1
            super().update(edges_predicted[mask], edges_target[mask])
        else:
            super().update(edges_predicted, edges_target)
