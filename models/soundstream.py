# Copyright (C) 2023 Katsuya Iida. All rights reserved.

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from itertools import chain
import random
from utils.utils_env import AttrDict

from huggingface_hub import PyTorchModelHubMixin

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
        

class EncoderBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        padding: str,
        stride: int
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            ResNet1d(n_channels // 2, padding=padding, dilation=1),
            ResNet1d(n_channels // 2, padding=padding, dilation=3),
            ResNet1d(n_channels // 2, padding=padding, dilation=9),
            nn.Conv1d(
                n_channels // 2, n_channels,
                kernel_size=2 * stride,
                padding=(2 * stride) // 2 if padding == 'same' else 0,
                stride=stride),
            nn.ELU(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


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


class Encoder(nn.Module):
    def __init__(self, n_channels: int, padding):
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            nn.Conv1d(1, n_channels, kernel_size=7, padding=padding),
            nn.ELU(),
            EncoderBlock(2 * n_channels, padding=padding, stride=2),
            EncoderBlock(4 * n_channels, padding=padding, stride=4),
            EncoderBlock(8 * n_channels, padding=padding, stride=5),
            EncoderBlock(16 * n_channels, padding=padding, stride=8),
            nn.Conv1d(16 * n_channels, 16 * n_channels, kernel_size=3, padding=padding),
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


class ResidualVectorQuantizer(nn.Module):
    weight: torch.Tensor
    running_mean: torch.Tensor
    code_count: torch.Tensor

    def __init__(
        self,
        num_quantizers: int,
        num_embeddings: int,
        embedding_dim: int,
        decay: float = 0.99,
        code_replace_threshold: float = 0.0001,
        eps: float = 1e-10,
    ) -> None:
        super().__init__()
        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.register_buffer("weight", torch.empty(num_quantizers, num_embeddings, embedding_dim))
        self.register_buffer("running_mean", torch.empty(num_quantizers, num_embeddings, embedding_dim))
        self.register_buffer("code_count", torch.empty(num_quantizers, num_embeddings))
        self.decay = decay
        self.eps = eps
        self.code_replace_threshold = code_replace_threshold
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.uniform_(self.weight)
        self.running_mean[:] = self.weight
        init.ones_(self.code_count)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # input: [..., chennel]
        if self.training:
            # Enabling bitrate scalability with quantizer dropout
            n = random.randrange(1, self.num_quantizers)
        else:
            n = self.num_quantizers
        codes = []
        r = input.type_as(self.running_mean).detach()
        with torch.no_grad():
            for i in range(n):
                w = self.weight[i]
                # r: [..., num_embeddings]
                dist = torch.cdist(r, w)
                k = torch.argmin(dist, axis=-1)
                codes.append(k)
                self._update_averages(i, r, k)
                r = r - F.embedding(k, w)
        quantized = input - r
        commitment_loss = torch.mean(torch.square(input - quantized.detach()))
        self.weight.data[:] = self.running_mean / torch.unsqueeze(self.eps + self.code_count, axis=-1)
        return quantized, torch.stack(codes, input.ndim - 1), commitment_loss

    def dequantize(self, input: torch.Tensor, n: Optional[int] = None) -> torch.Tensor:
        # input: [batch_size, length, num_quantizers]
        if n is None:
            n = input.shape[-1]
        assert 0 < n <= self.num_quantizers
        res = 0
        with torch.no_grad():
            for i in range(n):
                k = input[:, :, i]
                w = self.weight[i]
                res += F.embedding(k, w)
        return res

    def _update_averages(self, i: int, r: torch.Tensor, k: torch.Tensor) -> None:
        # https://arxiv.org/pdf/1906.00446.pdf
        # Generating Diverse High-Fidelity Images with VQ-VAE-2
        # 2.1 Vector Quantized Variational AutoEncode

        # k: [...]
        one_hot_k = F.one_hot(torch.flatten(k), self.num_embeddings).type_as(self.code_count)
        code_count_update = torch.mean(one_hot_k, axis=0)
        self.code_count[i].lerp_(code_count_update, 1 - self.decay)

        # r: [..., embedding_dim]
        r = r.reshape(-1, self.embedding_dim)
        running_mean_update = (one_hot_k.T @ r) / r.shape[0]
        self.running_mean[i].lerp_(running_mean_update, 1 - self.decay)

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def replace_vectors(self) -> int:
        # https://arxiv.org/pdf/2107.03312.pdf
        # C. Residual Vector Quantizer:

        # The original paper replaces with an input frame randomly
        # sampled within the current batch.
        # Here we replace with random average of running mean instead.
        num_replaced = torch.sum(self.code_count < self.code_replace_threshold).item()
        if num_replaced > 0:
            for i in range(self.num_quantizers):
                mask = self.code_count[i] < self.code_replace_threshold
                # mask: [num_quantizers, num_embeddings]
                w = torch.rand_like(self.code_count[i])
                w /= torch.sum(w)
                self.running_mean[i, mask] = w.type_as(self.running_mean) @ self.running_mean[i]
                self.code_count[i, mask] = w.type_as(self.code_count) @ self.code_count[i]

        return num_replaced

    @torch.no_grad()
    def calc_entropy(self) -> float:
        p = self.code_count / (self.eps + torch.sum(self.code_count, axis=-1, keepdim=True))
        return -torch.sum(torch.log(p) * p).item() / self.num_quantizers

class SoundStream(
    torch.nn.Module,
    PyTorchModelHubMixin,
    library_name="soundstream",
    repo_url="https://github.com/kaiidams/soundstream-pytorch",
    docs_url="https://github.com/kaiidams/soundstream-pytorch/blob/main/README.md",
    pipeline_tag="audio-to-audio",
    license="mit",
    tags=["neural-codec", "audio-generation", "arxiv:2107.03312"],
):
    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False):
        super().__init__()
        self.h = h
        self.encoder = Encoder(h.n_channels, h.padding)
        self.decoder = Decoder(h.n_channels, h.padding)
        self.quantizer = ResidualVectorQuantizer(
            h.num_quantizers, h.num_embeddings, h.n_channels * 16)

    def forward(self, data):
        x = data["y"]
        print("x", x.shape)
        x = self.encoder(x)
        print("mel", data["x"].shape)
        print("encode", x.shape)
        x = torch.transpose(x, -1, -2)
        x, codes, codebook_loss = self.quantizer(x)
        x = torch.transpose(x, -1, -2)
        print("codes", x.shape)
        x = self.decoder(x)
        print("x output", x.shape)
        return {"y_g_hat": x, "codes": codes, "codebook_loss": codebook_loss}
