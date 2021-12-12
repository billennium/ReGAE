import torch
import torchmetrics


class MaxGraphSize(torchmetrics.MaxMetric):
    label = "max_graph_size"

    def __init__(self, **kwargs):
        super().__init__(nan_strategy="error", **kwargs)

    def update(
        self,
        num_nodes: torch.Tensor,
        **kwargs,
    ):
        max_num_nodes = torch.max(num_nodes)
        super().update(max_num_nodes)

    @property
    def is_differentiable(self) -> bool:
        return True
