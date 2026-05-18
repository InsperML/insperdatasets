from pathlib import Path
from typing import Tuple
from torchaudio.transforms import Resample, Spectrogram, MelScale

import soundfile as sf
import torch


def load_audio(
    file_path: str | Path,
    t_start=None,
    t_end=None,
) -> Tuple[torch.Tensor, int]:
    """Load an audio file and return a PyTorch tensor and sample rate.

	- Reads the file with `soundfile` (supports many formats).
	- Returns a tensor with shape `(channels, samples)` and dtype `torch.float32`.
	- Also returns the sample rate as an `int`.

	Args:
		file_path: Path to an audio file.

	Returns:
		A tuple `(audio_tensor, sample_rate)`.
	"""
    path = Path(file_path)
    data, sr = sf.read(
        str(path),
        dtype="float32",
        start=t_start,
        stop=t_end,
    )

    # data shape: (samples,) for mono or (samples, channels) for multi-channel
    if data.ndim == 1:
        data = data[None, :]
    else:
        # transpose to (channels, samples)
        data = data.T

    tensor = torch.from_numpy(data)
    return tensor, int(sr)


class PreprocessingPipeline(torch.nn.Module):

    def __init__(
        self,
        input_freq=44100,
        resample_freq=16000,
        n_fft=1024,
        hop_length=512,
        n_mel=256,
    ):
        super().__init__()
        self.resample = Resample(orig_freq=input_freq, new_freq=resample_freq)

        self.spec = Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=1,
        )

        self.mel_scale = MelScale(n_mels=n_mel,
                                  f_min=20,
                                  f_max=resample_freq // 2,
                                  sample_rate=resample_freq,
                                  n_stft=n_fft // 2 + 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Resample the input
        resampled = self.resample(waveform)

        # Convert to power spectrogram
        spec = self.spec(resampled)

        # Convert to mel-scale
        mel = self.mel_scale(spec)

        return mel


def collate_audio(batch):
    """Collate audio tensors of variable length by zero-padding to the longest in the batch.

    Returns:
        audio: (B, C, T) tensor, zero-padded
        mask: (B, T) bool tensor, True where the signal is real (not padding)
        labels: list of label strings
    """
    audio_list, label_list = zip(*batch)

    # Filter out None entries from failed loads; unpack (tensor, sr) tuples if needed
    valid = [(a, l) for a, l in zip(audio_list, label_list) if a is not None]
    if not valid:
        return None, None, []
    audio_list, label_list = zip(*valid)
    audio_list = [a[0] if isinstance(a, (tuple, list)) else a for a in audio_list]

    max_len = max(a.shape[-1] for a in audio_list)
    padded = []
    mask = []
    for a in audio_list:
        pad_len = max_len - a.shape[-1]
        padded.append(torch.nn.functional.pad(a, (0, pad_len)))
        m = torch.ones(max_len, dtype=torch.bool)
        m[a.shape[-1]:] = False
        mask.append(m)

    return torch.stack(padded), torch.stack(mask), list(label_list)