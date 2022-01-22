from argparse import ArgumentParser
from typing import Optional

import pytorch_lightning as pl
from torch import Tensor
from torch.utils import data


class BaseDataModule(pl.LightningDataModule):
    data_name = ""

    collate_fn_train = None
    collate_fn_val = None
    collate_fn_test = None

    def __init__(
        self,
        batch_size: int,
        batch_size_val: int,
        batch_size_test: int,
        workers: int,
        persistent_workers: bool = False,
        **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val if batch_size_val > 0 else batch_size
        self.batch_size_test = batch_size_test if batch_size_test > 0 else batch_size
        self.workers = workers
        self.persistent_workers = persistent_workers
        self.train_dataset = None
        self.val_datasets = []
        self.test_datasets = []

    def train_dataloader(self, **kwargs):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn_train,
            **kwargs
        )

    def val_dataloader(self, **kwargs):
        return [
            data.DataLoader(
                dataset,
                batch_size=self.batch_size_val,
                num_workers=self.workers,
                persistent_workers=self.persistent_workers,
                collate_fn=self.collate_fn_val,
                **kwargs
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self, **kwargs):
        return [
            data.DataLoader(
                dataset,
                batch_size=self.batch_size_test,
                num_workers=self.workers,
                persistent_workers=self.persistent_workers,
                collate_fn=self.collate_fn_test,
                **kwargs
            )
            for dataset in self.test_datasets
        ]

    def input_size(self) -> int:
        raise NotImplementedError

    def output_size(self) -> Optional[int]:
        raise NotImplementedError

    def num_train_dataloaders(self) -> int:
        return 1

    def num_val_dataloaders(self) -> int:
        return len(self.val_datasets)

    def num_test_dataloaders(self) -> int:
        return len(self.test_datasets)

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--batch_size",
            dest="batch_size",
            default=64,
            type=int,
            metavar="SIZE",
            help="batch size of training data.",
        )
        parser.add_argument(
            "--batch_size_val",
            dest="batch_size_val",
            default=-1,
            type=int,
            metavar="SIZE",
            help="batch size of validation data.",
        )
        parser.add_argument(
            "--batch_size_test",
            dest="batch_size_test",
            default=-1,
            type=int,
            metavar="SIZE",
            help="batch size of test data.",
        )
        parser.add_argument(
            "--workers",
            default=0,
            type=int,
            metavar="W",
            help="""Number of data loading workers. \
                Depending on the OS and data module used, multiple workers may not work due to the way Pytorch \
                handles some collate_fn dataloder output types used in this project. For graph data multiple workers \
                are not needed and only slow down the training.""",
        )
        parser.add_argument(
            "--persistent_workers",
            dest="persistent_workers",
            action="store_true",
            help="turn on pytorch's data loader persistent workers",
        )

        return parent_parser
