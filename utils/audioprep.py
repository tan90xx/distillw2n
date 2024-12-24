import torch
import torchaudio
import torch.nn.functional as F

# Resampling if necessary
def resample_if_needed(signal, orig_sr, target_sr):
    if orig_sr != target_sr:
        return torchaudio.functional.resample(signal, orig_sr, target_sr)
    return signal
# Squeeze and normalize
def squeeze_and_normalize(signal):
    signal = torch.squeeze(signal)
    return signal * (0.95 / torch.max(signal))
# Pad if necessary
def pad_if_needed(signal, length):
    if signal.shape[0] < length:
        return F.pad(signal, [0, length - signal.shape[0]], "constant")
    return signal

def process_signal(signal, orig_sr, target_sr, target_len, segment_len):
    signal = resample_if_needed(signal, orig_sr, target_sr)
    signal = squeeze_and_normalize(signal)
    signal = signal[:target_len]
    signal = pad_if_needed(signal, segment_len)
    return signal