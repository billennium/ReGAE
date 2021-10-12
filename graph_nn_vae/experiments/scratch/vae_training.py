from graph_nn_vae.data import SyntheticGraphsDataModule

from graph_nn_vae.experiments.experiment import Experiment


if __name__ == "__main__":
    Experiment(SyntheticGraphsDataModule).run()
