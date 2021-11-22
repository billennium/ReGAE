import torch
import torchmetrics


class EdgePrecision(torchmetrics.Precision):
    label = "edge_precision"

    def __init__(self, **kwargs):
        super().__init__(num_classes=2, average=None, **kwargs)

    def update(
        self,
        edges_predicted: torch.Tensor, edges_target: torch.Tensor,
        mask_predicted: torch.Tensor = None, mask_target: torch.Tensor = None
    ):
        edges_predicted = torch.sigmoid(edges_predicted).int().flatten()
        edges_target = edges_target.int().flatten()

        super().update(edges_predicted, edges_target)

    def compute(self) -> torch.Tensor:
        return super().compute()[1]


class EdgeRecall(torchmetrics.Recall):
    label = "edge_recall"

    def __init__(self, **kwargs):
        super().__init__(num_classes=2, average=None, **kwargs)

    def update(
        self,
        edges_predicted: torch.Tensor, edges_target: torch.Tensor,
        mask_predicted: torch.Tensor = None, mask_target: torch.Tensor = None
    ):
        edges_predicted = torch.sigmoid(edges_predicted).int().flatten()
        edges_target = edges_target.int().flatten()

        super().update(edges_predicted, edges_target)

    def compute(self) -> torch.Tensor:
        return super().compute()[1]


class MaskPrecision(torchmetrics.Precision):
    label = "mask_precision"

    def __init__(self, **kwargs):
        super().__init__(num_classes=2, average=None, **kwargs)

    def update(
        self,
        mask_predicted: torch.Tensor, mask_target: torch.Tensor,
        edges_predicted: torch.Tensor = None, edges_target: torch.Tensor = None
    ):
        mask_predicted = torch.sigmoid(mask_predicted).int().flatten()
        mask_target = mask_target.int().flatten()

        super().update(mask_predicted, mask_target)

    def compute(self) -> torch.Tensor:
        return super().compute()[1]


class MaskRecall(torchmetrics.Recall):
    label = "mask_recall"

    def __init__(self, **kwargs):
        super().__init__(num_classes=2, average=None, **kwargs)

    def update(
        self,
        mask_predicted: torch.Tensor, mask_target: torch.Tensor,
        edges_predicted: torch.Tensor = None, edges_target: torch.Tensor = None
    ):
        mask_predicted = torch.sigmoid(mask_predicted).int().flatten()
        mask_target = mask_target.int().flatten()

        super().update(mask_predicted, mask_target)

    def compute(self) -> torch.Tensor:
        return super().compute()[1]
