from typing import Union

import numpy as np
import torch
from utils import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim


def prepare(audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    audio = audio.to(torch.float32)
    mel = log_mel_spectrogram(audio, n_mels=80, padding=N_SAMPLES)
    mel_segment = mel[:, 0 : 0 + N_FRAMES]
    mel_segment = pad_or_trim(mel_segment, N_FRAMES)
    return mel_segment
