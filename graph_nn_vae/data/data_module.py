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
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.workers = workers
        self.persistent_workers = persistent_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

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
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_val,
            num_workers=self.workers,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn_val,
            **kwargs
        )

    def test_dataloader(self, **kwargs):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_test,
            num_workers=self.workers,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn_test,
            **kwargs
        )

    def input_size(self) -> int:
        raise NotImplementedError

    def output_size(self) -> Optional[int]:
        raise NotImplementedError

    def loss_weight(self) -> Optional[Tensor]:
        return None

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
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
            default=1,
            type=int,
            metavar="SIZE",
            help="batch size of validation data.",
        )
        parser.add_argument(
            "--batch_size_test",
            dest="batch_size_test",
            default=1,
            type=int,
            metavar="SIZE",
            help="batch size of test data.",
        )
        parser.add_argument(
            "--workers",
            default=1,
            type=int,
            metavar="W",
            help="number of data loading workers",
        )
        parser.add_argument(
            "--persistent_workers",
            dest="persistent_workers",
            action="store_true",
            help="turn on pytorch's data loader persistent workers",
        )

        return parser
