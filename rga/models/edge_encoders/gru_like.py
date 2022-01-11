from argparse import ArgumentParser

import torch
from torch import nn, Tensor

from rga.models.utils.calc import weighted_average


class GRULikeEdgeEncoder(nn.Module):
    def __init__(self, embedding_size: int, edge_size: int, **kwargs):
        self.embedding_size = embedding_size
        self.edge_size = edge_size
        self.input_size = 2 * embedding_size + edge_size
        super().__init__()
        self.r = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ELU(),
            nn.Linear(512, embedding_size * 2),
            nn.Sigmoid(),
        )
        self.z = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ELU(),
            nn.Linear(512, embedding_size * 2),
            nn.Sigmoid(),
        )
        self.h = nn.Sequential(
            nn.Linear(self.input_size, 512), nn.ELU(), nn.Linear(512, embedding_size)
        )

    def forward(
        self, diagonal_x: Tensor, embedding_l: Tensor, embedding_r: Tensor
    ) -> Tensor:
        x = torch.cat((embedding_l, embedding_r, diagonal_x), dim=-1)
        embeddings = torch.cat((embedding_l, embedding_r), dim=-1)

        z = self.z(x)
        embedding_ratio, mem_overwrite_ratio = torch.split(
            z, (self.embedding_size, self.embedding_size), dim=-1
        )
        weighted_prev_emb = weighted_average(embedding_l, embedding_r, embedding_ratio)

        r = self.r(x)
        altered_embeddings = r * embeddings

        altered_prev_embs_with_diag_input = torch.cat(
            (altered_embeddings, diagonal_x), dim=-1
        )
        h = self.h(altered_prev_embs_with_diag_input)

        y = weighted_average(weighted_prev_emb, h, mem_overwrite_ratio)
        return y

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        return parent_parser
