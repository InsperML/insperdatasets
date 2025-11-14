import toml
from datasets import load_dataset
from torch.utils.data import Dataset
from .core.datasets import ListDataset
from pathlib import Path
from typing import Dict

def _get_text_dataset_from_huggingface(
    dataset_info,
    cache_dir=None,
):
    
    path = dataset_info['path']
    dataset = load_dataset(path, cache_dir=cache_dir)
    

    output_datasets = {}
    input_key = dataset_info['input']
    target_key = dataset_info['target']
    for split in ['split_train', 'split_validation']:
        data_split = dataset[dataset_info[split]]
        X = list(data_split[input_key])
        y = list(data_split[target_key])
        output_datasets[split.split('_')[1]] = ListDataset(X, y)
    return output_datasets



def get_dataset(
    dataset_name,
    cache_dir=None,
) -> Dict[str, Dataset]:
    try:
        info_path = Path(__file__).parent / 'resources' / 'datasets.toml'
        with open(info_path, 'r') as f:
            datasets = toml.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("The datasets.toml file was not found.")

    try:
        dataset_info = datasets[dataset_name]
    except KeyError:
        available_datasets = ', '.join(datasets.keys())
        raise KeyError(
            f"Dataset '{dataset_name}' not found in datasets.toml. Available datasets are: {available_datasets}"
        )

    if dataset_info['engine'] == 'huggingface':
        if dataset_info['type'] == 'text':
            return _get_text_dataset_from_huggingface(
                dataset_info,
                cache_dir=cache_dir,
            )
        else:
            raise ValueError(
                f"Unsupported dataset type '{dataset_info['type']}' for engine 'huggingface'."
            )