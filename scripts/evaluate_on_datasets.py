from pathlib import Path
import pandas as pd

from .generate_graphs import GraphGenerator


def print_separator(name: str = "", size: int = 60):
    print("")
    print("\u2500" * 3, name, "\u2500" * (size - 3 - len(name)), sep="")


if __name__ == "__main__":
    NUM_DATASETS = 5
    DATASET_NAMES = ["IMDB-BINARY"]
    DATASETS_PATH = "/usr/local/datasets/"
    CHECKPOINTS_PATH = "./best_checkpoints/"

    aggregated_metrics = {}
    for dataset in DATASET_NAMES:
        all_metrics = []
        for i in range(NUM_DATASETS):
            dataset_path = f"{DATASETS_PATH}/{dataset}/{i}.pkl"
            checkpoint_path = f"{CHECKPOINTS_PATH}/{dataset}/{i}.ckpt"

            generator = GraphGenerator()
            metrics = generator.run(
                dataset_pickle_path=dataset_path,
                checkpoint_path=checkpoint_path,
            )
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
