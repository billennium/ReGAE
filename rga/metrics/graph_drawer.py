from typing import Tuple
import math

import torch
from torch import Tensor
import torchmetrics

from rga.util.draw import draw_diag_repr_graph


class GraphDrawer(torchmetrics.Metric):
    label = "graph_drawer"
    alt_logging = True
    log_every_epochs = 100
    num_graphs = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_predicted_edges = None
        self.last_predicted_mask = None
        self.last_target_edges = None
        self.last_target_num_nodes = None
        self.next_epoch_to_log = 100

    def update(
        self,
        edges_predicted: Tensor,
        edges_target: Tensor,
        mask_predicted: Tensor,
        num_nodes: int,
        **kwargs,
    ):
        self.last_predicted_edges = edges_predicted[-self.num_graphs :]
        self.last_predicted_mask = mask_predicted[-self.num_graphs :]
        self.last_target_edges = edges_target[-self.num_graphs :]
        self.last_target_num_nodes = num_nodes[-self.num_graphs :]

    @property
    def is_differentiable(self) -> bool:
        return False

    def compute(self) -> None:
        return

    def alt_log(self, name: str, epoch: int) -> None:
        if epoch < self.next_epoch_to_log:
            return
        self.next_epoch_to_log = (
            epoch - (epoch % self.log_every_epochs) + self.log_every_epochs
        )
        self.draw_graph(name, epoch)

    def draw_graph(self, name: str, epoch: int):
        for i in range(self.num_graphs):
            predicted_edges, predicted_mask = self.remove_padding(
                torch.sigmoid(self.last_predicted_edges[i]),
                torch.sigmoid(self.last_predicted_mask[i]),
                0.0,
            )
            pred_g, pred_num_nodes = self.clean_raw_diag_repr_graph(
                predicted_edges, predicted_mask
            )
            draw_diag_repr_graph(pred_g, pred_num_nodes, f"{name}_{epoch}_{i}_pred")
            draw_diag_repr_graph(
                self.last_target_edges[i].cpu().detach(),
                self.last_target_num_nodes[i].cpu().detach(),
                f"{name}_{epoch}_{i}_target",
            )

    def remove_padding(self, edges, mask, padding_v) -> Tuple[Tensor, Tensor]:
        edges = edges.cpu().detach()
        mask = mask.cpu().detach()
        proper_mask_indices = mask != padding_v
        mask_shape = mask.shape[1:]
        mask = mask[proper_mask_indices]
        mask = torch.reshape(mask, (-1, *mask_shape))
        edges = edges[proper_mask_indices]
        edges = torch.reshape(edges, (-1, *mask_shape))
        return edges, mask

    def clean_raw_diag_repr_graph(
        self, edges: Tensor, mask: Tensor
    ) -> Tuple[Tensor, int]:
        num_block_diagonals = int((math.sqrt(mask.shape[0] * 8 + 1) - 1) / 2)
        last_mask_diag = mask[mask.shape[0] - num_block_diagonals :, ...]
        last_edge_diag = edges[edges.shape[0] - num_block_diagonals :, ...]
        block_size = edges.shape[-2]

        num_diagonals_in_block = 2 * block_size - 1
        center_diag_offset = int(num_diagonals_in_block / 2)
        num_diagonals = num_block_diagonals * block_size + block_size - 1
        for diag_offset in range(num_diagonals_in_block):
            t_diag_offset = center_diag_offset - diag_offset
            diag = torch.diagonal(last_mask_diag, offset=t_diag_offset, dim1=1, dim2=2)
            diag_mean = torch.mean(diag)
            if diag_mean < 0.5:
                num_diagonals -= 1
                edge_diag = torch.diagonal(
                    last_edge_diag, offset=t_diag_offset, dim1=1, dim2=2
                )
                edge_diag.fill_(0.0)
            else:
                break
        num_nodes = num_diagonals + 1
        edges = (edges > 0.5).float() * 1
        return edges, num_nodes
