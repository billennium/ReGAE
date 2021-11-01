import torch


def split_dataset_train_val_test(
    dataset: list, size_fractions: list[float]
) -> tuple[list]:
    if len(size_fractions) != 3:
        raise ValueError("the size_fractions argument has to be a list of length 3")

    train_dataset_size = int(0.8 * len(dataset))
    val_dataset_size = int(0.1 * len(dataset))
    test_dataset_size = len(dataset) - train_dataset_size - val_dataset_size

    return torch.utils.data.random_split(
        dataset,
        [train_dataset_size, val_dataset_size, test_dataset_size],
    )
