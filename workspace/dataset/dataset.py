import re
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from utils import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim

ROOT = Path(__file__).parent


def prepare(audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    audio = audio.to(torch.float32)
    mel = log_mel_spectrogram(audio, n_mels=80, padding=N_SAMPLES)
    content_frames = mel.shape[-1] - N_FRAMES
    mel_segment = mel[:, 0 : 0 + N_FRAMES]
    mel_segment = pad_or_trim(mel_segment, N_FRAMES)
    return mel_segment

class AudioDataset(Dataset):
    def __init__(self, split: str = "train"):
        super().__init__()
        self.X_dir = ROOT / "data" / split / "input"
        self.Y_dir = ROOT / "data" / split / "output"
        self.X = list(self.X_dir.iterdir())
        self.regex = re.compile(r"\d+\.\d+_\w+")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = self.X[idx]
        clean_filename = self.regex.search(item.stem).group()
        X, _ = torchaudio.load(item)
        Y, _ = torchaudio.load(self.Y_dir / f"{clean_filename}.wav")
        return prepare(X).squeeze(), prepare(Y).squeeze()

# class AudioDataset(datasets.GeneratorBasedBuilder):
#     BUILDER_CONFIGS = [
#         datasets.BuilderConfig(
#             name="noisy",
#             version=datasets.Version("1.0.0"),
#             description=""
#         )
#     ]

#     def _info(self):
#         return datasets.DatasetInfo(
#             features=datasets.Features(
#                 {
#                     "input": datasets.Value("string"),
#                     "output": datasets.Value("string")
#                 }
#             ),
#             supervised_keys=None,
#         )

#     def _split_generators(self, dl_manager):
#         return [
#             datasets.SplitGenerator(
#                 name=datasets.Split.TRAIN,
#                 gen_kwargs={"files": dl_manager.iter_files("dataset/data/train/")}
#             ),
#             datasets.SplitGenerator(
#                 name=datasets.Split.VALIDATION,
#                 gen_kwargs={"files": dl_manager.iter_files("dataset/data/val/")}
#             )
#         ]

#     def _generate_examples(self, files):
#         number_regex = re.compile(r"\d+\.\d+_\w+")
#         for file in files:
#             file = Path(file)
#             noisy_filename = file.stem
#             clean_filename = number_regex.search(noisy_filename).group()
#             split_dir = file.parent.parent
#             yield noisy_filename, {
#                 "input": str(file),
#                 "output": f"{split_dir}/output/{clean_filename}.wav"
#             }


# if __name__ == "__main__":
#     dataset = load_dataset(
#         __file__,
#     )

#     print(dataset['validation'][0])
