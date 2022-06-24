# R-GAE

Source code for paper ["R-GAE: Graph autoencoder based on recursiveneural networks"](https://arxiv.org/abs/2201.12165).

The paper was developed in collaboration with [Institute of Computer Science (FEIT/WUT)](http://www.ii.pw.edu.pl).

## Environment installation
**Note : Use Python 3.7 or newer**

Install the dependencies using `pip`
```
python -m pip install -r requirements.txt
```
The `requirements.txt` lists a standard "CPU" version on `pytorch`. In order to use a GPU, a version of `pytorch` specific
to the device will be required.

Install R-GAE as a python module:
```
pip install . 
```
in a folder of the cloned repository.

## Training models
The specific combinations of model-dataset-default_hyperparameters are defined as *experiments* in the `experiments/` subdirectory. There, a generic `Experiment` class is defined in `experiment.py` that describes the general training procedure. Apart from this class, individual experiment configurations are defined inside the subdirectories specifying a dataset. Each one runs the generic `Experiment` with the model-dataset-default_hyperparameters individual to the file. These files are runnable and are the intended way to run any training.

Example experiment run:
```
python -m rga.experiments.synthetic_grid_medium.recursive_autoencoder_training
```

As may be seen from the experiment definitions, all hyperparameters and run configurations are passed with run argument flags. Each module involved in the run (data, model, schedulers) define their flags, together with their descriptions and default values. Their full summary for a given experiment is displayed with the use of the `--help`/`-h` flag:
```
python -m rga.experiments.synthetic_grid_medium.recursive_autoencoder_training -h
```

To use own data:
1. Create a new experiment
2. Write new class inherited from BaseGraphLoader (rga.data.graph_loaders)
3. Assign a new data loader to graphloader_class
4. Customize parameters to your needs

## Use trained models 

To use trained autoencoder model install R-GAE as a python module:
```
pip install .
```
and import RGAE class from rga.models.rgae

```
from rga.models.rgae import RGAE

model = RGAE(path_hparams='...', path_ckpt='...')

#Show model parameters

print(model.hparams)

#Graph to embeddings

graphs = ... (list of torch.FloatTensor which represent adjacency matrices and (N, N) shape)
embeds = model.encode(graphs)

#Embeddings to graphs

reconstructed_graphs = model.decode(embeds)
```

## Running experiments with guild
[Guild AI](https://guild.ai/) is a toolset for running machine learning experiments. It provides a unified way to run hyperparameter searches,
analyze the network's performance and compare search results.

A rough configuration of the available experiments is defined in `guild.yml`. Each experiment is run by invoking it by the model's
name and the dataset, for example, `recursive_autoencoder:synthetic_grid_medium`. In the beginning, Guild may display a variety of warnings
related to either modules it cannot find or hyperparameter flags it cannot parse. The first is caused by the automatic definition of dataset-model
pairs that don't exist in the files (they were never needed). The flag-related warnings are caused mainly by guild's lack of support for some
flag types (ex. `list`, `dict`, `json`). These are harmless.

To run a generic experiment with the default parameters:
```
guild run recursive_autoencoder:synthetic_grid_medium
```

The `--force-flags` argument may be needed, as most of the model's hyperparameters are not defined in guild's configurations file. That's because
there are too many optional arguments, they often change throughout development, and some are only valid for specific experiment configurations.
A simple run with a custom `--seed` and `--learing_rate`:
```
guild run recursive_autoencoder:synthetic_grid_medium --force-flags seed=0 lr=0.001
```

To run a hyperparameter grid-search:
```
guild run recursive_autoencoder:synthetic_grid_medium --force-flags seed=0 lr=[0.001,0.003,0.005,0.01] batch_size=[32,64]
```

## Metrics and logging
The solution utilizes TensorBoard for logging the various metrics that help assess the training characteristics. `torchmetrics` is used to calculate the different metrics from model outputs. Logs are stored in the `tb_logs/` directory, which can be used as the `--logdir` directory of TensorBoard:
```
tensorboard --logdir=tb_logs
```

## Project structure
The primary model and training implementation are included in a Python module under the `rga/` directory.

All neural network-related processing is done using the `pytorch` set of libraries. The Pytorch Lightning framework is used to maintain an organized structure and utilize existing generic training methods. Pytorch Lightning generally takes care of the training procedure but requires an explicit split between the neural network model and the data loading functionalities. With this in mind, the codebase consists of the `models/` and `data/` directories. On top of this, the aforementioned general framework for specifying model-data-hyperparameter sets is implemented, defined in the `experiments/` directory. Other than these three, various other directories provide supporting functionalities.
### data/
Data loading and transforming. This includes loading-in graph datasets, transforming them into formats compatible with NN processing, and permuting adjacency matrices. Most of these are performed inside `DataModule`s, which provide a unified architecture for loading, processing, and passing the data to Pytorch Lightning's training methods.

### experiments/
As described above, the main `Experiment` class and the training run definitions (`experiments`) are defined there.

### models/
Neural network model definitions and the closely related data processing, loss calculating, and training forward passes. All models conform to the `BaseModule` class interface, which is derived from a `LightningModule`. This kind of interface allows for relatively clean definitions of various models compatible with the `Experiment` procedure.

### lr_schedulers/
Custom learning rate schedulers. The `FactorDecreasingOnMetricChange` one is instrumental when paired with the progressive subgraph training procedure for learning large graphs. Modules access all classes of this kind through the functions defined in `models/utils/getters.py`.

### metrics/
Custom metrics. The metrics for calculating graph edge reconstruction performance are noteworthy, such as `EdgeAccuracy`, `EdgePrecision`, `EdgeF1`.

### util/
An assortment of one-off calculation functions, training helpers, model weight loaders, and such. Of note is the `utils/adjmatrix/` submodule that defines graph adjacency matrix format conversions, permutations, and orderings.

Apart from the `rga` module, the following are included:
- `tests/`, in which unit tests for some essential functionalities are defined. Tests use the `pytest` testing framework.
- `scipts/` and `notebooks/`, which include scripts for batch training and evaluating, as well as generating data splits for benchmarking.

A `dockerfile` is provided to build an image with a Python environment compatible with running all experiments. The containerization is optional but handy for separating Python environments or avoiding the installation of pip modules.

## Datasets
The experiments presented in the paper, and for which the default, tested hyperparameters are provided, are:
- GRID-MEDIUM
- IMDB-BINARY
- IMDB-MULTI
- COLLAB
- REDDIT-BINARY
- REDDIT-MULTI-5K
- REDDIT-MULTI-12K

The GRID-MEDIUM dataset is a synthetic dataset that doesn't require any outside data, as it is programmatically generated. The rest can be downloaded, among other sources, from the [Benchmark Data Sets for Graph Kernels](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets) database. These should be extracted to a `datasets/` directory in the repository's root directory (by default). The resulting file tree should be the following:
```
.
└── datasets/
    └── "dataset_name"/
        ├── "dataset_name"_A.txt
        ├── "dataset_name"_graph_indicator.txt
        └── "dataset_name"_graph_labels.txt
```

## Cite
Please cite our paper if you use this code in your own work:

```
@article{malkowski2022graph,
  title={Graph autoencoder with constant dimensional latent space},
  author={Ma{\l}kowski, Adam and Grzechoci{\'n}ski, Jakub and Wawrzy{\'n}ski, Pawe{\l}},
  journal={arXiv preprint arXiv:2201.12165},
  year={2022}
}
```
