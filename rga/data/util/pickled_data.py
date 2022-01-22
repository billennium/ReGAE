import pickle
from typing import Tuple


def load_pickled_data(path: str, expect_labels: bool) -> Tuple:
    with open(path, "rb") as input:
        (
            train_graph_dataset,
            val_graph_datasets,
            test_graph_datasets,
            train_label_dataset,
            val_label_datasets,
            test_label_datasets,
        ) = pickle.load(input)
    if expect_labels:
        train_dataset = list(zip(train_graph_dataset, train_label_dataset))
        val_datasets = [
            list(zip(val_graph_datasets[i], val_label_datasets[i]))
            for i in range(len(val_graph_datasets))
        ]
        test_datasets = [
            list(zip(test_graph_datasets[i], test_label_datasets[i]))
            for i in range(len(test_graph_datasets))
        ]
    else:
        train_dataset = train_graph_dataset
        val_datasets = val_graph_datasets
        test_datasets = test_graph_datasets

    print(f"Dataset {path} successfully loaded!")
    return (train_dataset, val_datasets, test_datasets)
