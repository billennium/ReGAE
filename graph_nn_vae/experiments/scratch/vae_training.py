from graph_nn_vae.data import SyntheticGraphsDataModule

from graph_nn_vae.experiments.experiment import Experiment
from graph_nn_vae.models.autoencoder_components import GraphEncoder


if __name__ == "__main__":
    Experiment(GraphEncoder, SyntheticGraphsDataModule).run()
