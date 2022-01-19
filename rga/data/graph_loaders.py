from typing import List, Tuple, Dict
from argparse import ArgumentParser
from pathlib import Path
from tqdm.auto import tqdm

import networkx as nx
import numpy as np

from networkx.readwrite.gml import parse_gml_lines

from rga.data.data_module import BaseDataModule
from rga.data.synthetic_graphs_create import create_synthetic_graphs
from rga.util.adjmatrix import filter_out_big_graphs


class BaseGraphLoader:
    data_name = "graph_loader"

    def __init__(self, **kwargs):
        pass

    def load_graphs(self) -> Dict:
        """
        Overload this function to specify graphs for the dataset.
        """
        return NotImplementedError

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser):
        return parent_parser


class SyntheticGraphLoader(BaseGraphLoader):
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

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser):
        parent_parser = BaseGraphLoader.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--graph_type",
            dest="graph_type",
            default="grid_small",
            type=str,
            help="Type of synthethic graphs",
        )
        return parent_parser


class RealGraphLoader(BaseGraphLoader):
    data_name = "real"

    def __init__(
        self,
        datasets_dir: str = "",
        dataset_name: str = "",
        use_labels: bool = False,
        max_graph_size: int = None,
        **kwargs,
    ):
        self.dataset_dir = Path(datasets_dir)
        self.dataset_name = dataset_name
        self.dataset_folder = self.dataset_dir / Path(dataset_name)
        self.data_name = dataset_name
        self.use_labels = use_labels
        self.max_graph_size = max_graph_size
        super().__init__(**kwargs)

    def load_graphs(self) -> Dict:
        print(f"Loading graphs from {self.dataset_folder / Path(self.dataset_name)}")
        data_adj = np.loadtxt(
            self.dataset_folder / Path(self.dataset_name + "_A.txt"), delimiter=","
        ).astype(int)
        data_graph_indicator = np.loadtxt(
            self.dataset_folder / Path(self.dataset_name + "_graph_indicator.txt"),
            delimiter=",",
        ).astype(int)
        data_tuple = list(map(tuple, data_adj))

        if self.use_labels:
            with open(
                self.dataset_folder / Path(self.dataset_name + "_graph_labels.txt")
            ) as file:
                graphs_labels = [int(el) for el in file.read().splitlines()]
        else:
            graphs_labels = None

        print("Processing loaded graph edges")
        G = nx.Graph()
        G.add_edges_from(data_tuple)
        G.remove_nodes_from(list(nx.isolates(G)))
        graph_num = data_graph_indicator.max()
        node_list = np.arange(data_graph_indicator.shape[0]) + 1
        graphs = []
        for i in range(graph_num):
            # find the nodes for each graph
            nodes = node_list[data_graph_indicator == i + 1]
            G_sub: nx.Graph = G.subgraph(nodes)
            graphs.append(nx.to_scipy_sparse_matrix(G_sub, dtype=np.int32))

        if self.max_graph_size:
            adj_matrices, graphs_labels = filter_out_big_graphs(
                graphs, graphs_labels, self.max_graph_size
            )

        return {"graphs": graphs, "labels": graphs_labels}

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser):
        parent_parser = BaseGraphLoader.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--max_graph_size",
            dest="max_graph_size",
            type=int,
            help="ignore graphs which has more nodes",
        )
        parser.add_argument(
            "--datasets_dir",
            dest="datasets_dir",
            default="datasets",
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
        return parent_parser
