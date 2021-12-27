from typing import List, Tuple, Dict
from argparse import ArgumentParser
from pathlib import Path
from tqdm.auto import tqdm

import networkx as nx
import numpy as np

from networkx.readwrite.gml import parse_gml_lines

from graph_nn_vae.data.data_module import BaseDataModule
from graph_nn_vae.data.synthetic_graphs_create import create_synthetic_graphs
from graph_nn_vae.util.adjmatrix import filter_out_big_graphs


class GraphLoaderBase:
    data_name = "graph_loader"

    def __init__(self, **kwargs):
        pass

    def load_graphs(self) -> Dict:
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

    def load_graphs(self) -> Dict:
        return {
            "graphs": [
                nx.to_numpy_array(nx_graph, dtype=np.float32)
                for nx_graph in create_synthetic_graphs(self.graph_type)
            ]
        }

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

    def __init__(
        self,
        datasets_dir: str = "",
        dataset_name: str = "",
        use_labels: bool = False,
        max_graph_size: int = None,
        **kwargs
    ):
        self.dataset_dir = Path(datasets_dir)
        self.dataset_name = dataset_name
        self.dataset_folder = self.dataset_dir / Path(dataset_name)
        self.data_name = dataset_name
        self.use_labels = use_labels
        self.max_graph_size = max_graph_size
        super().__init__(**kwargs)

    def load_graphs(self) -> Dict:
        with open(
            self.dataset_folder / Path(self.dataset_name + "_graph_indicator.txt")
        ) as file:
            graph_indicator = file.read().splitlines()
            graph_indicator = np.array([int(el) for el in graph_indicator])

        graphs_index, graphs_sizes = np.unique(graph_indicator, return_counts=True)
        graph_size_cumulative = [sum(graphs_sizes[: el - 1]) for el in graphs_index]

        adj_matrices = []
        for i in graphs_sizes:
            adj_matrices.append(np.zeros((i, i)))

        with open(self.dataset_folder / Path(self.dataset_name + "_A.txt")) as file:
            for line in tqdm(file, desc="reading edges"):
                edge_first_node, edge_second_node = line.strip().split(",")
                edge_first_node = int(edge_first_node)
                edge_second_node = int(edge_second_node)
                current_graph = graph_indicator[edge_first_node - 1]
                edge_first_node = (
                    edge_first_node - graph_size_cumulative[current_graph - 1]
                )
                edge_second_node = (
                    edge_second_node - graph_size_cumulative[current_graph - 1]
                )

                adj_matrices[current_graph - 1][
                    edge_first_node - 1, edge_second_node - 1
                ] = 1

        if self.use_labels:
            with open(
                self.dataset_folder / Path(self.dataset_name + "_graph_labels.txt")
            ) as file:
                graphs_labels = [int(el) for el in file.read().splitlines()]
        else:
            graphs_labels = None

        if self.max_graph_size:
            filtered_adj_matrices, filtered_graph_labels = filter_out_big_graphs(
                adj_matrices, graphs_labels, self.max_graph_size
            )

        return {"graphs": filtered_adj_matrices, "labels": filtered_graph_labels}

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = GraphLoaderBase.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--max_graph_size",
            dest="max_graph_size",
            type=int,
            help="ignore graphs which has more nodes",
        )
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
