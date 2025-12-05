from insperdatasets.get_dataset import get_dataset

def test_get_dataset():
    dataset = get_dataset('sst2')
    assert 'train' in dataset
    assert 'validation' in dataset
    assert len(dataset['train']) > 0
    assert len(dataset['validation']) > 0

def test_get_dataset_consistency():
    dataset_sst2 = get_dataset('sst2')
    dataset_imdb = get_dataset('imdb')
    assert dataset_sst2.keys() == dataset_imdb.keys()
    
def test_arxiv_dataset():
    dataset = get_dataset('arxiv2025')
    assert 'train' in dataset
    assert 'validation' in dataset
    assert len(dataset['train']) > 0
    assert len(dataset['validation']) > 0