import torch
import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import numpy as np
from libs import HubertSoft

# Fine-tuned Soft-Hubert Block copied from https://github.com/rkmt/wesper-demo.
def load_hubert(checkpoint_path=None, device='cuda:0'):
    print("### load_hubert", checkpoint_path, device)
    assert checkpoint_path is not None
    print("### loading checkpoint from: ", checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    hubert = HubertSoft().to(device)

    checkpoint = checkpoint['hubert'] if checkpoint['hubert'] is not None else checkpoint
    consume_prefix_in_state_dict_if_present(checkpoint, "module.")

    hubert.load_state_dict(checkpoint, strict=True)
    hubert = hubert.eval().to(device)
    return hubert

def wav2units(wav, encoder, layer=None, device='cuda:0'):
    ''' 
        encoder: HuBERT
    '''
    if type(wav) == np.ndarray:
        wav = torch.tensor([wav], dtype=torch.float32, device=device)
    else:
        wav = wav.to(device)
    assert type(wav) == torch.Tensor
    if len(wav.shape) == 2:
        wav = wav.unsqueeze(0)
    with torch.inference_mode():  # wav -> HuBERT soft units
        if layer is None or layer < 0:
            units = encoder.units(wav)
        else:
            wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
            units, _ = encoder.encode(wav, layer=layer)
    return units