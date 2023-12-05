from typing import Any, Dict, Union

import numpy as np
import torch
from utils import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim


def prepare_audio(audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    audio = audio.to(torch.float32)
    mel = log_mel_spectrogram(audio, n_mels=80, padding=N_SAMPLES)
    mel_segment = mel[:, 0 : 0 + N_FRAMES]
    mel_segment = pad_or_trim(mel_segment, N_FRAMES)
    return mel_segment


def prepare(audios: Dict[str, Any]) -> torch.Tensor:
    inputs = []
    outputs = []
    for audio_noisy, audio_clean in zip(audios['input'], audios['output']):
        audio_noisy = audio_noisy['array']
        audio_clean = audio_clean['array']
        mel_segment_noisy = prepare_audio(audio_noisy)
        mel_segment_clean = prepare_audio(audio_clean)
        inputs.append(mel_segment_noisy)
        outputs.append(mel_segment_clean)
    return {"input": inputs, "output": outputs}
