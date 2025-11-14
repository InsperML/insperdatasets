import toml
from datasets import load_dataset
from .core.datasets import ListDataset
from pathlib import Path

def _get_text_dataset_from_huggingface(
    dataset_info,
    cache_dir=None,
):
    
    path = dataset_info['path']
    dataset = load_dataset(path, cache_dir=cache_dir)
    
    for split in dataset_info['splits']:
        if split not in dataset:
            raise ValueError(
                f"Split '{split}' not found in dataset '{path}'. Available splits are: {list(dataset.keys())}"
            )
    
    datasets = {}
    input_key = dataset_info['input']
    target_key = dataset_info['target']
    for split in dataset_info['splits']:
        data_split = dataset[split]
        X = list(data_split[input_key])
        y = list(data_split[target_key])
        datasets[split] = ListDataset(X, y)

    return datasets



def get_dataset(
    dataset_name,
    cache_dir=None,
):
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