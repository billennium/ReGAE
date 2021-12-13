from typing import List
from argparse import ArgumentParser
from pathlib import Path

import networkx as nx

from networkx.readwrite.gml import parse_gml_lines

from graph_nn_vae.data.data_module import BaseDataModule
from graph_nn_vae.data.synthetic_graphs_create import create_synthetic_graphs


class GraphLoaderBase:
    data_name = "graph_loader"

    def __init__(self, **kwargs):
        pass

    def load_graphs(self) -> List[nx.Graph]:
        """
        Overload this function to specify graphs for the dataset.
        """
        return NotImplementedError

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser


class SyntheticGraphLoader(GraphLoaderBase):
    data_name = "synthetic"

    def __init__(self, graph_type: str = "", **kwargs):
        self.graph_type = graph_type
        self.data_name += "_" + graph_type
        super().__init__(**kwargs)

    def load_graphs(self) -> List[nx.Graph]:
        return create_synthetic_graphs(self.graph_type)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = GraphLoaderBase.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--graph_type",
            dest="graph_type",
            default="grid_small",
            type=str,
            help="Type of synthethic graphs",
        )
        return parser


class RealGraphLoader(GraphLoaderBase):
    data_name = "real"

    def __init__(self, datasets_dir: str = "", dataset_name: str = "", **kwargs):
        self.dataset_dir = Path(datasets_dir)
        self.dataset_name = dataset_name
        self.dataset_folder = self.dataset_dir / Path(dataset_name)
        self.data_name += dataset_name
        super().__init__(**kwargs)

    def load_graphs(self) -> List[nx.Graph]:
        # TODO PICKLE FROM BACKUP

        graph_with_all_edges = nx.read_edgelist(
            self.dataset_folder / Path(self.dataset_name + "_A.txt"),
            delimiter=", ",
            data=int,
        )

        graphs = [
            graph_with_all_edges.subgraph(c)
            for c in nx.connected_components(graph_with_all_edges)
        ]

        # TODO SAVE PICKLE BACKUP

        return graphs

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = GraphLoaderBase.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--datasets_dir",
            dest="datasets_dir",
            default="",
            type=str,
            help="dir to folder of datasets (imdb, reddit, collab)",
        )
        parser.add_argument(
            "--dataset_name",
            dest="dataset_name",
            default="",
            type=str,
            help="name of dataset (IMDB_BINARY, IMDB_MULTI, COLLAB, REDDIT-BINARY, REDDIT-MULTI-5K, REDDIT-MULTI-12K)",
        )
        parser.add_argument(
            "--use_catche",
            dest="use_catche",
            action="store_true",
            help="catche subgraphs into pickle file, if file exist, read insted of loading from txt files",
        )
        return parser
