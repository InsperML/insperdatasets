from torch.utils.data import Dataset
from collections.abc import Callable
from typing import Any
from pathlib import Path


class FMADataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        loader_func: Callable[[Path], Any],        
    ):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None, None
