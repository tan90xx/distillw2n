# Torch and related libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio import features
from utils.utils_env import AttrDict
from huggingface_hub import PyTorchModelHubMixin
from models.soundstream import Decoder

# Learnable MFCCs Extractor
class MFCC(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels, n_embed_dim, trainable=False, **params):
        super().__init__()
        self.spec = features.MFCC(
            sr=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mfcc=n_mels,
            trainable_mel=trainable,
            trainable_STFT=trainable,
        )
        # self.conv = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1)
        self.linear = nn.Linear(n_mels, n_embed_dim)

    def forward(self, input):
        x = self.spec(input)
        # x = x[..., :-1]
        # y = torch.repeat_interleave(x, 2, dim=1)
        # y = self.conv(x)
        x = x.permute(0, 2, 1)
        y = self.linear(x)
        y = y.permute(0, 2, 1)
        return y

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

# Reencoder
class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer
    Reference: https://arxiv.org/abs/1709.07871
    """
    def __init__(self, in_channels, out_channels, cond_channels):
        super(FiLMLayer, self).__init__()
        self.in_channels = in_channels
        self.film = nn.Conv1d(cond_channels, (in_channels + out_channels), 1)

    def forward(self, x, c):
        gamma, beta = torch.chunk(self.film(c.unsqueeze(2)), chunks=2, dim=1)
        return gamma * x + beta

class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, in_channels, cond_channels):
        """
        Style Adaptive Layer Normalization (SALN) module.

        Parameters:
        in_channels: The number of channels in the input feature maps.
        cond_channels: The number of channels in the conditioning input.
        """
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.in_channels = in_channels

        self.saln = nn.Linear(cond_channels, in_channels * 2, 1)
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.saln.bias.data[:self.in_channels], 1)
        nn.init.constant_(self.saln.bias.data[self.in_channels:], 0)

    def forward(self, x, c):
        c = self.saln(c.unsqueeze(1))
        gamma, beta = torch.chunk(c, chunks=2, dim=-1)
        return gamma * self.norm(x) + beta
    
class ConvNeXtBlock_Adapt(nn.Module):
    def __init__(self, gin_channels, layer_scale_init_value: float = 1e-6,):
        super().__init__()
        self.dwconv = nn.Conv1d(256, 256, kernel_size=7, padding=3, groups=256)
        self.norm = StyleAdaptiveLayerNorm(256, gin_channels)
        self.pwconv_2 = nn.Sequential(nn.Linear(256, 256*4),
                                    nn.GELU(),
                                    nn.Linear(256*4, 256))
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(256), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x, c) -> torch.Tensor:
        residual = x # 24,256,102
        x = self.dwconv(x) # 24,512,102
        x = self.norm(x.transpose(1, 2), c)  # 24,512,102
        x = self.pwconv_2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)
        x = residual + x
        return x

class Conditioner(torch.nn.Module):
    def __init__(self, n_layers: int, wavenet_embed_dim: int, 
                 decoder_causal: bool = False, nn_type='conv'):
        super(Conditioner, self).__init__()
        self.nn_type = nn_type
        if nn_type == 'film':
            self.film = FiLMLayer(in_channels=256, out_channels=256, cond_channels=192)
        elif nn_type == 'adapt':
            self.adapt = ConvNeXtBlock_Adapt(gin_channels=192)
        elif nn_type == 'norm':
            self.norm = StyleAdaptiveLayerNorm(256, 192)
        # self.conv_out = torch.nn.Conv1d(256, 512, 1)
    

    def forward(self, c_code, spk_emb): # c_code.shape [B, 256, 100]
        if self.nn_type == 'conv':
            spk_emb = self.spk_proj(spk_emb.unsqueeze(2)) # [B, 256]
            c_code = c_code + spk_emb
            # z = self.conv_out(c_code)
        elif self.nn_type == 'film':
            x = self.film(c_code, spk_emb)
            c_code = self.adapt(c_code, spk_emb)
            # z = self.conv_out(c_code)
        elif self.nn_type == 'adapt':
            c_code = self.adapt(c_code, spk_emb)
            # z = self.conv_out(c_code)
        elif self.nn_type == 'norm':
            x = self.norm(c_code.transpose(1, 2), spk_emb)
            c_code = x.transpose(1, 2)
            # z = self.conv_out(c_code)
        # elif self.nn_type == 'wo':
        #     # z = self.conv_out(c_code)
        return c_code

class DistillW2N(
    torch.nn.Module,
    PyTorchModelHubMixin,
    library_name="distillw2n",
    repo_url="https://github.com/tan90xx/distillw2n",
    docs_url="https://github.com/tan90xx/distillw2n/blob/main/README.md",
    pipeline_tag="audio-to-audio",
    license="mit",
    tags=["neural-vocoder", "audio-generation", "None"],
):
    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False):
        super().__init__()
        self.h = h
        self.spec = MFCC(sample_rate=h.sampling_rate, 
                         n_fft=h.n_fft,
                         win_length=h.win_size,
                         hop_length=h.hop_size*2,
                         n_mels=h.num_mels,
                         n_embed_dim=h.n_embed_dim,
                         trainable=h.trainable
                        )
        self.encoder = DVAEDecoder(idim=h.n_embed_dim, 
                                   odim=h.n_embed_dim, 
                                   n_layer=h.n_encoder_layer
                                  )
        self.adaptor = Conditioner(n_layers=h.n_conditioner_layer, 
                                   wavenet_embed_dim=h.n_embed_dim, 
                                   nn_type=h.conditioner_type
                                  )
        self.decoder = Decoder(n_channels=h.n_channels, padding=h.padding)

    def forward(self, data):
        # x, spk = data["y"], data["spk"]
        x = data["y"]
        x = self.spec(x).transpose(-1, -2)
        x = self.encoder(x)
        units = torch.nn.functional.pad(x, (0, 0, 0, 1, 0, 0))
        x = torch.transpose(units, -1, -2)
        x = self.decoder(x)
        return {"y_g_hat": x, "units": units}