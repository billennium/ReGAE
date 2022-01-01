from typing import List, Tuple, Dict, Optional, Type
from argparse import ArgumentParser
import pickle
import os
from tqdm.auto import tqdm
import json

import torch

from graph_nn_vae.data.data_module import BaseDataModule
from graph_nn_vae.data.graph_loaders import GraphLoaderBase
from graph_nn_vae.util import adjmatrix, split_dataset_train_val_test, flatten, errors
from graph_nn_vae.util.convert_size import convert_size
from graph_nn_vae.data.util.print_dataset_statistics import print_dataset_statistics


class AdjMatrixDataModule(BaseDataModule):
    graphloader_class: Type[GraphLoaderBase] = None  # override in experiment

    def __init__(
        self,
        num_dataset_graph_permutations: int,
        train_val_test_split: list,
        train_val_test_permutation_split: Optional[list],
        bfs: bool = False,
        deduplicate_train: bool = False,
        deduplicate_val_test: bool = False,
        use_labels: bool = False,
        save_dataset_to_pickle: str = None,
        pickled_dataset_path: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.initialize_graphloader(**kwargs)
        self.num_dataset_graph_permutations = num_dataset_graph_permutations
        self.bfs = bfs
        self.deduplicate_train = deduplicate_train
        self.deduplicate_val_test = deduplicate_val_test
        self.train_val_test_split = train_val_test_split
        self.train_val_test_permutation_split = train_val_test_permutation_split
        self.use_labels = use_labels
        self.save_dataset_to_pickle = save_dataset_to_pickle
        self.pickled_dataset_path = pickled_dataset_path

        self.prepare_data()

    def initialize_graphloader(self, **kwargs):
        if self.graphloader_class is None:
            raise errors.MisconfigurationException(
                "the graphloader_class attribute of the DataModule was not specified"
            )
        self.graphloader: GraphLoaderBase = self.graphloader_class(**kwargs)
        self.data_name = self.graphloader.data_name

    def prepare_data(self, *args, **kwargs):
        super().prepare_data(*args, **kwargs)

        if self.pickled_dataset_path:
            self.load_pickled_data()
        else:
            graphs, graph_labels = self.create_graphs()

            if self.use_labels and graph_labels is None:
                raise RuntimeError(
                    f"the flag --use_labels is set, but the dataset contains no labels"
                )

            graph_data = list(zip(graphs, graph_labels)) if self.use_labels else graphs
            print_dataset_statistics(
                graph_data, "Full original dataset", self.use_labels
            )

            train_graphs, val_graphs, test_graphs = split_dataset_train_val_test(
                graph_data, self.train_val_test_split
            )

            if len(val_graphs) == 0 or len(train_graphs) == 0:
                val_graphs = graph_data
                train_graphs = graph_data
                test_graphs = graph_data

            train_graph_permutations = self.permute_adj_matrices(
                train_graphs, self.num_dataset_graph_permutations
            )
            val_graphs = flatten(
                self.permute_adj_matrices(
                    val_graphs, self.num_dataset_graph_permutations
                )
            )
            test_graphs = flatten(
                self.permute_adj_matrices(
                    test_graphs, self.num_dataset_graph_permutations
                )
            )

            self.train_dataset = flatten(train_graph_permutations)
            self.val_datasets = [val_graphs]
            self.test_datasets = [test_graphs]

            train_graph_permutations_train = []
            train_graph_permutations_val = []
            train_graph_permutations_test = []
            if isinstance(self.train_val_test_permutation_split, list):
                for g in train_graph_permutations:
                    train, val, test = split_dataset_train_val_test(
                        g, self.train_val_test_permutation_split
                    )
                    train_graph_permutations_train.extend(train)
                    train_graph_permutations_val.extend(val)
                    train_graph_permutations_test.extend(test)
                    if len(test) > 0:
                        pass

            if len(train_graph_permutations_train) != 0:
                self.train_dataset = train_graph_permutations_train
            if len(train_graph_permutations_val) != 0:
                self.val_datasets.append(train_graph_permutations_val)
            if len(train_graph_permutations_test) != 0:
                self.val_datasets.append(train_graph_permutations_test)

            if self.save_dataset_to_pickle:
                self.pickle_dataset()

        print_dataset_statistics(self.train_dataset, "Train dataset", self.use_labels)
        for i, d in enumerate(self.val_datasets):
            print_dataset_statistics(d, f"Validation dataset {i}", self.use_labels)
        for i, d in enumerate(self.test_datasets):
            print_dataset_statistics(d, f"Test dataset {i}", self.use_labels)

        self.train_dataset = self.prepare_dataset_for_autoencoder(
            self.train_dataset,
            self.deduplicate_train,
            dataset_name="train",
        )
        self.val_datasets = [
            self.prepare_dataset_for_autoencoder(
                d, self.deduplicate_val_test, dataset_name=f"val {i}"
            )
            for i, d in enumerate(self.val_datasets)
        ]
        self.test_datasets = [
            self.prepare_dataset_for_autoencoder(
                d, self.deduplicate_val_test, dataset_name=f"test {i}"
            )
            for i, d in enumerate(self.test_datasets)
        ]

    def create_graphs(self) -> Dict:
        data = self.graphloader.load_graphs()
        return data["graphs"], data.get("labels", None)

    def prepare_dataset_for_autoencoder(
        self,
        graph_data,
        deduplicate: bool,
        dataset_name: str = "",
    ) -> List[Tuple[torch.Tensor, int]]:
        """
        Reorders adj matrices with bfs, removes duplicates, minimizes the matrices and appends num_nodes info.
        """
        graphs = [el[0] for el in graph_data] if self.use_labels else graph_data
        labels = [el[1] for el in graph_data] if self.use_labels else None

        adj_matrices = []
        adj_matrix_labels = [] if labels is not None else None

        for index, adj_matrix in enumerate(
            tqdm(graphs, desc=f"preparing dataset {dataset_name} for autoencoder")
        ):
            adj_matrix = adjmatrix.bfs_ordering(adj_matrix)
            adj_matrix = adjmatrix.minimize_adj_matrix(adj_matrix)
            adj_matrices.append((adj_matrix, adj_matrix.shape[0]))
            if labels is not None:
                adj_matrix_labels.append(labels[index])

        if deduplicate:
            adj_matrices, adj_matrix_labels = self.deduplicate_graph_batch(
                adj_matrices, adj_matrix_labels
            )

        return (
            list(zip(adj_matrices, adj_matrix_labels))
            if adj_matrix_labels is not None
            else adj_matrices
        )

    def deduplicate_graph_batch(self, adj_matrices, adj_matrix_labels):
        unique_matrix_indices = adjmatrix.get_unique_indices(
            [m[0] for m in adj_matrices]
        )
        adj_matrices = [adj_matrices[i] for i in unique_matrix_indices]
        if adj_matrix_labels is not None:
            adj_matrix_labels = [adj_matrix_labels[i] for i in unique_matrix_indices]
        return unique_matrix_indices, adj_matrix_labels

    def permute_adj_matrices(self, graph_data, num_permutations: int):
        graphs = [el[0] for el in graph_data] if self.use_labels else graph_data
        labels = [el[1] for el in graph_data] if self.use_labels else None

        permuted_graphs = []

        for i, graph in enumerate(graphs):
            graph_permutations = [graph] + [
                adjmatrix.random_permute(graph) for _ in range(num_permutations - 1)
            ]
            graph_permutations = adjmatrix.remove_duplicates(graph_permutations)
            if self.use_labels:
                multiplied_labels = [labels[i] for _ in range(len(graph_permutations))]
                permuted_graphs.append(list(zip(graph_permutations, multiplied_labels)))
            else:
                permuted_graphs.append(graph_permutations)

        return permuted_graphs

    def flatten(self, l: list[list]) -> list:
        return [item for sublist in l for item in sublist]

    def load_pickled_data(self):
        with open(self.pickled_dataset_path, "rb") as input:
            (
                train_graph_dataset,
                val_graph_datasets,
                test_graph_datasets,
                train_label_dataset,
                val_label_datasets,
                test_label_datasets,
            ) = pickle.load(input)
        if self.use_labels:
            self.train_dataset = list(zip(train_graph_dataset, train_label_dataset))
            self.val_datasets = [
                list(zip(val_graph_datasets[i], val_label_datasets[i]))
                for i in len(val_graph_datasets)
            ]
            self.test_datasets = [
                list(zip(test_graph_datasets[i], test_label_datasets[i]))
                for i in len(test_graph_datasets)
            ]
        else:
            self.train_dataset = train_graph_dataset
            self.val_datasets = val_graph_datasets
            self.test_datasets = test_graph_datasets

        print("Dataset successfully loaded!")
        print("File path:", self.pickled_dataset_path)

    def pickle_dataset(self):
        val_graph_datasets, val_label_datasets = [], []
        test_graph_datasets, test_label_datasets = [], []

        train_graphs = (
            [el[0] for el in self.train_dataset]
            if self.use_labels
            else self.train_dataset
        )
        train_labels = [el[1] for el in self.train_dataset] if self.use_labels else None

        for dataset in self.val_datasets:
            val_graphs = [el[0] for el in dataset] if self.use_labels else dataset
            val_graph_datasets.append(val_graphs)
            val_labels = [el[1] for el in dataset] if self.use_labels else None
            val_label_datasets.append(val_labels)

        for dataset in self.test_datasets:
            test_graphs = [el[0] for el in dataset] if self.use_labels else dataset
            test_graph_datasets.append(test_graphs)
            test_labels = [el[1] for el in dataset] if self.use_labels else None
            test_label_datasets.append(test_labels)

        with open(self.save_dataset_to_pickle, "wb") as output:
            pickle.dump(
                (
                    train_graphs,
                    val_graph_datasets,
                    test_graph_datasets,
                    train_labels,
                    val_label_datasets,
                    test_label_datasets,
                ),
                output,
            )
        print("Dataset pickled successfully")
        print("File path:", self.save_dataset_to_pickle)
        print("File size:", convert_size(os.path.getsize(self.save_dataset_to_pickle)))

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser):
        parent_parser = BaseDataModule.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--num_dataset_graph_permutations",
            dest="num_dataset_graph_permutations",
            default=10,
            type=int,
            help="number of permuted copies of the same graphs to generate in the dataset",
        )
        parser.add_argument(
            "--bfs",
            dest="bfs",
            type=bool,
            help="reorder nodes in graphs by using BFS",
        )
        parser.add_argument(
            "--train_val_test_split",
            dest="train_val_test_split",
            default=[0.7, 0.15, 0.15],
            metavar="JSON_LIST",
            type=json.loads,
            help="list of 3 floats specifying the dataset train/val/test split",
        )
        parser.add_argument(
            "--train_val_test_permutation_split",
            dest="train_val_test_permutation_split",
            default=[0.8, 0.2, 0.0],
            metavar="JSON_LIST",
            type=json.loads,
            help="""
                list of 3 floats specifying the train dataset permutation split. \
                Use this to check how well the network generalizes to unknown train dataset permutations. \
                The metrics for the permutation val test datasets will be marked with "_1". \
                Passing anything but a list will disable permutation splitting.""",
        )
        parser.add_argument(
            "--deduplicate_train",
            dest="deduplicate_train",
            action="store_true",
            help="remove duplicates from the train dataset after applying bfs ordering",
        )
        parser.add_argument(
            "--deduplicate_val_test",
            dest="deduplicate_val_test",
            action="store_true",
            help="remove duplicates from val and test datasets after applying bfs ordering",
        )
        parser.add_argument(
            "--use_labels",
            dest="use_labels",
            action="store_true",
            help="use graph labels",
        )
        parser.add_argument(
            "--save_dataset_to_pickle",
            dest="save_dataset_to_pickle",
            default=None,
            type=str,
            help="save dataset to pickle files",
        )
        parser.add_argument(
            "--pickled_dataset_path",
            dest="pickled_dataset_path",
            default=None,
            type=str,
            help="save dataset to pickle files",
        )
        return parent_parser
