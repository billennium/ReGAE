from typing import List
from argparse import ArgumentParser
import networkx as nx

from graph_nn_vae.data.diag_repr_graph_data_module import (
    DiagonalRepresentationGraphDataModule,
)
from graph_nn_vae.data.synthetic_graphs_create import create_synthetic_graphs


class SyntheticGraphsDataModule(DiagonalRepresentationGraphDataModule):
    data_name = "synthetic"

    def __init__(self, graph_type: str = "", **kwargs):
        self.graph_type = graph_type
        self.data_name += "_" + graph_type
        super().__init__(**kwargs)

    def create_graphs(self) -> List[nx.Graph]:
        return create_synthetic_graphs(self.graph_type)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = DiagonalRepresentationGraphDataModule.add_model_specific_args(
            parent_parser
        )
        parser.add_argument(
            "--graph_type",
            dest="graph_type",
            default="grid_small",
            type=str,
            help="Type of synthethic graphs",
        )
        return parser
