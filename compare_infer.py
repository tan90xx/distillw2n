import torch
import numpy as np
import soundfile as sf
import librosa

def init_model(model_type):

    if model_type == 'quickvc':
        from minimal_quickvc.models import SynthesizerTrn
        from minimal_quickvc.utils import load_checkpoint
        model = SynthesizerTrn().eval().to('cuda')
        model_path = './experiments/quickvc/quickvc.pth'
        _ = load_checkpoint(model_path, model, None)
        embedder_model = torch.hub.load(
            "bshall/hubert:main", "hubert_soft").eval().to('cuda')
        
    elif model_type == 'wesper':
        from minimal_wesper.whisper_normal import SynthesizerTrn, load_hubert
        model = SynthesizerTrn().eval().to('cuda')
        embedder_model = load_hubert(device='cuda')

    return embedder_model, model


class Inferer:
    def __init__(self, model_type):
        self.model_type = model_type
        self.hubert, self.model = init_model(model_type)
        self.conv_sr = 16000

    def vc_fn(self, audio):
        with torch.no_grad():
            wav_src = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to('cuda')
            c = self.hubert.units(wav_src)
            c = c.transpose(2, 1)

            if self.model_type == 'quickvc':
                mel_tgt = torch.zeros(1, 80, 64).to('cuda')
                audio = self.model.infer(c, mel=mel_tgt)

            elif self.model_type == 'wesper':
                audio = self.model.infer(c)
                
            audio = audio.squeeze(0).squeeze(0).cpu().numpy()
            audio = (audio * 32767).astype(np.int16)
        return audio
    
    def file_infer(self, fname, oname):
        audio, _ = librosa.load(fname, sr=self.conv_sr)
        audio_out = self.vc_fn(audio)
        sf.write(oname, audio_out, self.conv_sr)
        return audio_out
    
inferer = Inferer('quickvc')
audio_out = inferer.file_infer('./raw/gt/s000u003w.wav', 's000u003w_quickvc.wav')
inferer = Inferer('wesper')
audio_out = inferer.file_infer('./raw/gt/s000u003w.wav', 's000u003w_wesper.wav')