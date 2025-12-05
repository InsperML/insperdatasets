import toml
from datasets import load_dataset
from torch.utils.data import Dataset
from .core.datasets import ListDataset
from pathlib import Path
from typing import Dict
from sklearn.model_selection import train_test_split


def _get_dataset_split(data_split, input_key, target_key):
    X = list(data_split[input_key])
    y = list(data_split[target_key])
    return X, y


def _get_text_dataset_from_huggingface(
    dataset_info,
    cache_dir=None,
):

    path = dataset_info['path']
    dataset = load_dataset(path, cache_dir=cache_dir)

    output_datasets = {}
    input_key = dataset_info['input']
    target_key = dataset_info['target']

    X, y = _get_dataset_split(dataset[dataset_info['split_train']], input_key,
                              target_key)

    if dataset_info['split_train'] == dataset_info['split_validation']:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=dataset_info['split_train_size'],
            random_state=dataset_info['split_random_seed'],
        )
        X, y = X_train, y_train
        
    else:
        X_test, y_test = _get_dataset_split(
            dataset[dataset_info['split_validation']],
            input_key,
            target_key,
        )
    output_datasets['train'] = ListDataset(X, y)
    output_datasets['validation'] = ListDataset(X_test, y_test)
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
