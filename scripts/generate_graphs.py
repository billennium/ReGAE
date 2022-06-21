import argparse
import pickle
import pytorch_lightning as pl

from rga.data.diag_repr_graph_data_module import DiagonalRepresentationGraphDataModule
from rga.models.utils.load import load_hparams, load_model
from rga.models.autoencoder_base import RecursiveGraphAutoencoder
from rga.util import adjmatrix
from rga.util.generate_graphs import *
from rga.metrics.adjency_matrices_metrics import calculate_metrics


class GraphGenerator:
    def run(
        self,
        checkpoint_path: str,
        dataset_pickle_path: str,
        output_graphs_path: str = None,
        gpu: int = None,
        evaluate: bool = True,
        **kwargs,
    ):
        pl.seed_everything(0)

        hparams_path = checkpoint_path.removesuffix(".ckpt") + "_hparams.yaml"
        hparams = load_hparams(hparams_path)
        model = load_model(hparams_path, checkpoint_path, RecursiveGraphAutoencoder)

        print("Model loaded.")
        # model.summarize()
        # pl.utilities.model_summary.summarize(model)

        hparams["pickled_dataset_path"] = dataset_pickle_path
        data_module = DiagonalRepresentationGraphDataModule(**hparams)

        gpus = [gpu] if gpu not in ["none", "None", None] else None
        predictor = pl.Trainer(gpus=gpus)
        model_output = predictor.predict(
            model, dataloaders=data_module.test_dataloader()
        )
        diag_block_predictions = convert_model_output_to_diag_block(model_output)

        predictions = diag_block_graphs_to_tril_adj_matrices(diag_block_predictions)
        test_dataset = data_module.test_datasets[0]
        targets = diag_block_graphs_to_tril_adj_matrices(test_dataset)

        for i, g in enumerate(predictions):
            predictions[i] = adjmatrix.adj_matrix_to_diagonal_representation(
                g[..., None], diag_block_predictions[i][2]
            )[..., 0]
        for i, g in enumerate(targets):
            targets[i] = adjmatrix.adj_matrix_to_diagonal_representation(
                g[..., None], test_dataset[i][2]
            )[..., 0]

        if output_graphs_path is not None:
            with open(output_graphs_path, "wb") as output:
                pickle.dump((predictions, targets), output)

        if evaluate:
            return calculate_metrics(targets, predictions)
        return None

    def add_argparse_arguments(
        self, parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        parser.add_argument("--checkpoint_path", type=str)
        parser.add_argument("--dataset_pickle_path", type=str)
        parser.add_argument("--output_graphs_path", type=str, default=None)
        parser.add_argument("--gpu", type=int, default=None)
        parser.add_argument("--evaluate", type=bool, default=True)
        return parser


if __name__ == "__main__":
    generator = GraphGenerator()

    parser = argparse.ArgumentParser()
    parser = generator.add_argparse_arguments(parser)
    args = parser.parse_args()

    metrics = generator.run(**vars(args))
    for k, v in metrics.items():
        print(f"{k}:{v.item():.4f}")
