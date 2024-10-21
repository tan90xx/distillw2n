import json
import torch
import libs.hifigan as hifigan


def get_vocoder(config, device):

    with open("./libs/hifigan/my_config_v1_16000.json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    ckpt = torch.load("./libs/hifigan/g_00180000.zip")
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder):

    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1) # rkmt 2022.6.1

    # wavs = (wavs.cpu().numpy() * 32768.0).astype("int16")
    # wavs = [wav for wav in wavs]

    return wavs
