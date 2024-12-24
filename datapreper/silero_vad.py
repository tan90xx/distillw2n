import torch
torch.set_num_threads(1)
from typing import List
USE_ONNX = True # change this to True if you want to test onnx model
  
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=USE_ONNX)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

def replace_chunks(tss: List[dict],
                wav: torch.Tensor):
    chunks = []
    cur_start = 0
    for i in tss:
        silence_part = torch.zeros_like(wav[cur_start: i['start']])
        chunks.append(silence_part)
        voiced_part = wav[i['start']: i['end']]
        chunks.append(voiced_part)
        cur_start = i['end']
    silence_part = torch.zeros_like(wav[cur_start:])
    chunks.append(silence_part)
    result = torch.cat(chunks)
    if torch.all(result == 0):
        return wav
    return result


def process_wav(in_path, out_path, sample_rate):
    wav = read_audio(in_path, sampling_rate=sample_rate)
    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sample_rate)
    # merge all speech chunks to one audio
    save_audio(out_path,
            replace_chunks(speech_timestamps, wav), sampling_rate=sample_rate)
    return out_path, wav.size(-1) / sample_rate