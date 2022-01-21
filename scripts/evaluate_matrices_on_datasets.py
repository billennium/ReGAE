import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from rga.data.util.pickled_data import load_pickled_data
from rga import util
from rga.util import adjmatrix
from rga.metrics.adjency_matrices_metrics import calculate_metrics


def print_separator(name: str = "", size: int = 60):
    print("")
    print("\u2500" * 3, name, "\u2500" * (size - 3 - len(name)), sep="")


if __name__ == "__main__":
    NUM_DATASETS = 5
    DATASET_NAMES = ["IMDB-BINARY"]
    DATASETS_PATH = "/usr/local/datasets"
    PREDICTIONS_PATH = "./ae_pred/"

    aggregated_metrics = {}
    for dataset in DATASET_NAMES:
        all_metrics = []
        for i in range(NUM_DATASETS):
            dataset_path = f"{DATASETS_PATH}/{dataset}/{i}.pkl"
            predictions_path = f"{PREDICTIONS_PATH}/{dataset}/test_predictions_{i}.pkl"

            _, _, targets = load_pickled_data(dataset_path, False)
            if isinstance(targets, list):
                targets = targets[0]
            with open(predictions_path, "rb") as input:
                predictions = pickle.load(input)

            if len(targets) != len(predictions):
                print(
                    f"Number of predictions does not match the number of targets in the test dataset;"
                    f"{len(predictions) = }, {len(targets) = }"
                )

            diag_targets = []
            diag_predictions = []
            for i in tqdm(
                range(len(targets)), desc=f"processing target and prediction matrices"
            ):
                target = targets[i]
                if isinstance(target, (np.ndarray, np.generic)):
                    target = torch.from_numpy(target)
                target = util.to_dense_if_not(target)[..., None]

                num_nodes = target.shape[0]

                prediction = predictions[i]
                if isinstance(prediction, (np.ndarray, np.generic)):
                    prediction = torch.from_numpy(prediction)
                prediction = util.to_dense_if_not(prediction)[..., None]
                if prediction.shape[0] > num_nodes:
                    prediction = prediction[:num_nodes, :num_nodes, :]

                target = adjmatrix.adj_matrix_to_diagonal_representation(
                    target, num_nodes
                )[..., 0].int()
                prediction = adjmatrix.adj_matrix_to_diagonal_representation(
                    prediction, num_nodes
                )[..., 0]

                if ((prediction == 0.0).sum() + (prediction == 1.0).sum()) != len(
                    prediction
                ):
                    prediction = torch.sigmoid(prediction)
                    prediction = prediction.round_().int()

                diag_targets.append(target)
                diag_predictions.append(prediction)

            metrics = calculate_metrics(diag_targets, diag_predictions)
            all_metrics.append(metrics)

            for k, v in metrics.items():
                print(f"{k} {v.item():.4f}")

        dataset_agg_metrics = {}
        for k in all_metrics[0]:
            dataset_agg_metrics[k] = [m[k].item() for m in all_metrics]
        df = pd.DataFrame.from_dict(dataset_agg_metrics, orient="index").round(4)
        df_mean = df.mean(numeric_only=True, axis=1)
        df_std = df.std(numeric_only=True, axis=1)
        df["mean"] = df_mean
        df["std"] = df_std
        aggregated_metrics[dataset] = df

    for k in aggregated_metrics:
        print(aggregated_metrics[k])
    df = pd.concat(aggregated_metrics, axis=1)

    print_separator("latex_print")
    print(df.to_latex())

    print_separator("html_print")
    print(df.to_html())

    print_separator("normal_print")
    print(df)
