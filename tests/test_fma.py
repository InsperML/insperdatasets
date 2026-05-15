import pytest

from insperdatasets.audio.fma import FMADataset

def test_fma_dataset():
    dataset = FMADataset(data_dir="/mnt/data3/fma/fma", loader_func=lambda x: x)
    assert len(dataset) > 0
