import pytest
import torch 


def test_load_audio():
    import insperdatasets.audio.utils as utils
    audio_path = "/mnt/data2/fma/fma_full/001/001236.mp3"
    tensor, sr = utils.load_audio(audio_path)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(sr, int)
    assert tensor.ndim == 1
    assert sr == 16000