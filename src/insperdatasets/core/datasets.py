from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Subset


class ListDataset(Dataset):
    def __init__(
        self,
        X: list,
        y: list,
    ):
        assert len(X) == len(y), 'X and y must have the same length'
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FileLoadingDataset(Dataset):
    def __init__(
        self,
        file_paths: list[str | Path],
        labels: list,
        loader_func: Callable[[str | Path], Any],
    ):
        assert len(file_paths) == len(labels), (
            'file_paths and labels must have the same length'
        )
        self.file_paths = file_paths
        self.labels = labels
        self.loader_func = loader_func

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = self.loader_func(self.file_paths[idx])
        label = self.labels[idx]
        return data, label


def get_random_subset(dataset: Dataset, n_samples: int) -> Dataset:
    """Return a random subset of the given dataset with n_samples."""
    assert hasattr(dataset, '__len__'), 'Dataset must have a __len__ method'
    total_samples = len(dataset)  # type: ignore

    random_indices = torch.randint(0, total_samples, (n_samples,))
    return Subset(dataset, random_indices.tolist())
