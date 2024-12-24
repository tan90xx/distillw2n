# Torch and related libraries
import torch
import torch.nn as nn
from nnAudio import features
from utils.config import Config

def call_feature_by_name(name, *args, **kwargs):
    func = globals().get(name)
    if func and callable(func):
        return func(*args, **kwargs)
    else:
        print("Function not found or not callable.")

# Learnable MFCCs Extractor
class mfcc(nn.Module):
    def __init__(self, trainable=False, **params):
        super().__init__()
        config = Config({})
        self.spec = features.MFCC(
            sr=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mfcc=config.n_mels,
            trainable_mel=trainable,
            trainable_STFT=trainable,
        )
        # self.conv = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1)
        self.linear = nn.Linear(config.n_mels, config.n_embed_dim)

    def forward(self, input):
        x = self.spec(input)
        # y = torch.repeat_interleave(x, 2, dim=1)
        # y = self.conv(x)
        x = x.permute(0, 2, 1)
        y = self.linear(x)
        y = y.permute(0, 2, 1)
        return y

class melspec(nn.Module):
    def __init__(self, **params):
        super().__init__()
        # self.spec = features.MelSpectrogram(
        #                 sr=16000,
        #                 n_fft=1024,
        #                 win_length=1024,
        #                 hop_length=320,
        #                 n_mels=256,
        #                 fmin=0.0,
        #                 fmax=None,
        #                 trainable_mel=True, 
        #                 trainable_STFT=True
        #             )
        self.spec = features.gammatone.Gammatonegram(
                        sr=16000,
                        n_fft=1024,
                        hop_length=320,
                        n_bins=256,
                        fmin=0.0,
                        fmax=None,
                        trainable_bins=True, 
                        trainable_STFT=True
                    )

    def forward(self, input):
        # logmel = F.interpolate(logmel, scale_factor=2)
        x = self.spec(input)
        return x[..., :-1]
    
class stftspec(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.spec = features.STFT(
            n_fft=1024,
            win_length=1024,
            freq_bins=256,
            hop_length=320,
            output_format="Magnitude",
        ) # trainable=True,

    def forward(self, input):
        return self.spec(input)


# Encoder
class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel, dilation,
        layer_scale_init_value: float = 1e-6,
    ):
        # ConvNeXt Block copied from Vocos.
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 
                                kernel_size=kernel, padding=dilation*(kernel//2), 
                                dilation=dilation, groups=dim
                            )  # depthwise conv
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor, cond = None) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x
    
class DVAEDecoder(nn.Module):
    def __init__(self, idim, odim,
                 n_layer = 12, bn_dim = 64, hidden = 256, 
                 kernel = 7, dilation = 2, up = False
                ):
        super().__init__()
        self.up = up
        self.conv_in = nn.Sequential(
            nn.Conv1d(idim, bn_dim, 3, 1, 1), nn.GELU(),
            nn.Conv1d(bn_dim, hidden, 3, 1, 1)
        )
        self.decoder_block = nn.ModuleList([
            ConvNeXtBlock(hidden, hidden* 4, kernel, dilation,)
            for _ in range(n_layer)])
        self.conv_out = nn.Conv1d(hidden, odim, kernel_size=1, bias=False)
        # self.layernorm1 = nn.LayerNorm(256)
        # self.layernorm2 = nn.LayerNorm(256, bias=False)

    def forward(self, input, conditioning=None):
        # B, T, C
        # x = self.layernorm1(input)
        x = input.transpose(1, 2)
        x = self.conv_in(x)
        for f in self.decoder_block:
            x = f(x, conditioning)
        x = self.conv_out(x)
        x = x.transpose(1, 2)
        # x = self.layernorm2(x)
        return x
        