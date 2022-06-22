import torch
from typing import List, Tuple


def split_dataset_train_val_test(
    dataset: List, size_fractions: List[float]
) -> Tuple[List]:
    if len(size_fractions) != 3:
        raise ValueError("the size_fractions argument has to be a list of length 3")

    val_dataset_size = int(size_fractions[1] * len(dataset))
    test_dataset_size = int(size_fractions[2] * len(dataset))
    train_dataset_size = len(dataset) - val_dataset_size - test_dataset_size

    return torch.utils.data.random_split(
        dataset,
        [train_dataset_size, val_dataset_size, test_dataset_size],
    )
