import insperdatasets.audio.utils as audio_utils
from insperdatasets.core.datasets import FileLoadingDataset
import insperdatasets.audio.fma as fma
from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial

dur = 3

def main():
    dataset = fma.FMADataset(data_dir='/mnt/data3/fma/fma',
                             loader_func=partial(audio_utils.load_audio, crop='random', t_len=dur*16000))

    print(f'We have {len(dataset)} tracks in the dataset.')

    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=40,
        prefetch_factor=2,
        collate_fn=audio_utils.collate_audio,
    )

    for batch in tqdm(dataloader):
        #print(batch)
        audio_data, mask, label = batch
        audio_data = audio_data.to('cuda')
        mask = mask.to('cuda')
        #label = [l.to('cuda') for l in label]
        # Here you can add code to process the audio_data and label as needed.
        # For example, you could print the shape of the audio data and the label:
        #print(f'Track {i}: Audio data shape: {audio_data.shape}, Label: {label}')
    print(audio_data.shape, mask.shape, label[:10])

if __name__ == "__main__":
    main()
