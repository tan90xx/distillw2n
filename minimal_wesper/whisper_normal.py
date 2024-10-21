import json
import torch
import yaml

from torch import nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

# FastSpeech2
from libs.FastSpeech2 import FastSpeech2
from libs.FastSpeech2.utils.tools import pad_2D
# HuBERT
from libs.hubert.model import HubertSoft
# Hifigan
from libs import hifigan
from libs.hifigan.model import vocoder_infer

def load_fastspeech2(device='cuda'):
    checkpoint_path = 'https://github.com/rkmt/wesper-demo/releases/download/v0.1/googletts_neutral_best.tar'
    preprocess_config = './minimal_wesper/config/my_preprocess16k_LJ.yaml'
    model_config = './minimal_wesper/config/my_model16000.yaml'
    preprocess_config = yaml.load(open(preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if checkpoint_path.startswith("http"):
        ckpt = torch.hub.load_state_dict_from_url(checkpoint_path, map_location=torch.device('cpu')) if device!='cuda' else torch.hub.load_state_dict_from_url(checkpoint_path)
    else:
        ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu')) if device!='cuda' else torch.load(checkpoint_path)
    model.load_state_dict(ckpt["model"], strict=True)

    model = model.to(device)
    model.eval()
    model.requires_grad_ = False
    return model


def load_hubert(device='cuda'):
    checkpoint_path = "https://github.com/rkmt/wesper-demo/releases/download/v0.1/model-layer12-450000.pt"
    if checkpoint_path.startswith("http"):
        checkpoint = torch.hub.load_state_dict_from_url(checkpoint_path, map_location=torch.device('cpu')) if device!='cuda' else torch.hub.load_state_dict_from_url(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) if device!='cuda' else torch.load(checkpoint_path)
    hubert = HubertSoft().to(device)

    checkpoint = checkpoint['hubert'] if checkpoint['hubert'] is not None else checkpoint
    consume_prefix_in_state_dict_if_present(checkpoint, "module.")

    hubert.load_state_dict(checkpoint, strict=True)
    hubert.eval().to(device)
    return hubert


def load_hifigan(device='cuda'):
    checkpoint_path='https://github.com/rkmt/wesper-demo/releases/download/v0.1/g_00205000'
    with open("./libs/hifigan/my_config_v1_16000.json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    if checkpoint_path.startswith("http"):
        ckpt = torch.hub.load_state_dict_from_url(checkpoint_path, map_location=torch.device('cpu')) if device!='cuda' else torch.hub.load_state_dict_from_url(checkpoint_path)
    else:
        ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu')) if device!='cuda' else torch.load(checkpoint_path)

    vocoder.load_state_dict(ckpt['generator'])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)

    return vocoder


class SynthesizerTrn(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda'
        self.fs2model = load_fastspeech2(device=self.device)
        self.vocoder = load_hifigan(device=self.device)

    def infer(self, c):
        c = c.squeeze(0).detach().cpu().numpy()
        c = pad_2D([c])
        c = torch.from_numpy(c).to(self.device)
        speakers = torch.tensor([0], device=self.device)
        max_src_len = c.shape[1]
        src_lens = torch.tensor([max_src_len], device=self.device)

        with torch.no_grad():
            output = self.fs2model(speakers, c, src_lens, max_src_len)
        mel_len = output[9][0].item()
        mel_prediction = output[1][0, :mel_len].detach().transpose(0, 1)
        
        with torch.no_grad():
            o = vocoder_infer(mel_prediction.unsqueeze(0),self.vocoder,)[0]
        return o


class MyWhisper2Normal(object):
    def __init__(self, args):
        self.device = args.device
        
        self.encoder = load_hubert(device=self.device)
        self.syn = SynthesizerTrn()

    def convert(self, wav_from):
        wav_t = torch.from_numpy(wav_from).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            units = self.encoder.units(wav_t)
        wav_prediction = self.syn.infer(units)
        wav_prediction = (wav_prediction.cpu().numpy() * 32768.0).astype("int16")
        return wav_prediction

