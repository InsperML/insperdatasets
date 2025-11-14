from torch.utils.data import Dataset
from collections.abc import Callable


class ListDataset(Dataset):

    def __init__(
        self,
        X: list,
        y: list,
    ):
        assert len(self.X) == len(self.y), "X and y must have the same length"
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FileLoadingDataset(Dataset):

    def __init__(
        self,
        file_paths: list,
        labels: list,
        loader_func: Callable,
    ):
        assert len(file_paths) == len(
            labels), "file_paths and labels must have the same length"
        self.file_paths = file_paths
        self.labels = labels
        self.loader_func = loader_func

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = self.loader_func(self.file_paths[idx])
        label = self.labels[idx]
        return data, label
