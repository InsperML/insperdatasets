from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import Dataset


def _read_metadata(data_dir: str | Path):
    data_dir = Path(data_dir)
    metadata_file = data_dir / 'fma_metadata' / 'tracks.csv'

    # Using Pandas to read the CSV file. The first 3 rows are header information.
    # The final lines of the CSV file do not conform to the expected format and
    # are ignored by Pandas. Setting low_memory=False to avoid dtype inference issues.
    data = pd.read_csv(
        metadata_file, header=None, skiprows=3, index_col=None, low_memory=False
    )

    # Take the first 3 rows as header information, concatenate the cell values
    # to form proper column names, and set them as the DataFrame's columns.
    header_data = pd.read_csv(metadata_file, header=None, nrows=3)
    column_names = (
        header_data.fillna('').astype(str).agg('_'.join, axis=0).str.strip('_').values
    )
    data.columns = column_names

    # Construct the file paths for each track based on the 'track_id' column and the directory structure.
    data['file_path'] = data['track_id'].map(
        lambda x: data_dir / 'fma_wav16k' / f'{x // 1000:03d}' / f'{x:06d}.wav'
    )

    return data


class FMADataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        loader_func: Callable[[Path], Any],
    ):
        self.data = _read_metadata(data_dir)
        self.loader_func = loader_func

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        track = self.data.iloc[idx]
        audio_path = track['file_path']
        try:
            audio_data = self.loader_func(audio_path)
        except Exception as e:
            print(f'Error occurred while loading audio file {audio_path}: {e}')
            audio_data = None
        return audio_data, track['track_genre_top']
