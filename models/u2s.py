# Torch and related libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Reencoder(torch.nn.Module):
    def __init__(self, n_layers: int, wavenet_embed_dim: int, 
                 decoder_causal: bool = False, nn_type='conv'):
        super(Reencoder, self).__init__()
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

# Decoder copied from https://github.com/kaiidams/soundstream-pytorch
class ResNet1d(nn.Module):
    def __init__(
        self,
        n_channels,
        kernel_size: int = 7,
        padding: str = 'valid',
        dilation: int = 1
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self._padding_size = (kernel_size // 2) * dilation
        self.conv0 = nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation)
        self.conv1 = nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size=1)

    def forward(self, input):
        y = input
        x = self.conv0(input)
        x = F.elu(x)
        x = self.conv1(x)
        if self.padding == 'valid':
            y = y[:, :, self._padding_size:-self._padding_size]
        x += y
        x = F.elu(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        padding: str,
        stride: int
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(
                n_channels, n_channels // 2,
                kernel_size=2 * stride,
                padding=(2 * stride) // 2 if padding == 'same' else 0,
                stride=stride),
            nn.ELU(),
            ResNet1d(n_channels // 2, padding=padding, dilation=1),
            ResNet1d(n_channels // 2, padding=padding, dilation=3),
            ResNet1d(n_channels // 2, padding=padding, dilation=9),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)

class Decoder(nn.Module):
    def __init__(self, n_channels: int, padding):
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            nn.Conv1d(16 * n_channels, 16 * n_channels, kernel_size=7, padding=padding),
            nn.ELU(),
            DecoderBlock(16 * n_channels, padding=padding, stride=8),
            DecoderBlock(8 * n_channels, padding=padding, stride=5),
            DecoderBlock(4 * n_channels, padding=padding, stride=4),
            DecoderBlock(2 * n_channels, padding=padding, stride=2),
            nn.Conv1d(n_channels, 1, kernel_size=7, padding=padding),
            nn.Tanh(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)