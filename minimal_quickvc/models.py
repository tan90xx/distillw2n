import math
import numpy as np
from scipy.signal import get_window
import torch
from torch import nn
from torch.nn import functional as F
from . import commons, modules

from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm


class TorchSTFT(torch.nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.from_numpy(get_window(
            window, win_length, fftbins=True).astype(np.float32))

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window,
            return_complex=True)

        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(magnitude.device))

        # unsqueeze to stay consistent with conv_transpose1d implementation
        return inverse_transform.unsqueeze(-2)

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels,
                              kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size,
                              dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(
            x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Multistream_iSTFT_Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, subbands, gin_channels=0):
        super(Multistream_iSTFT_Generator, self).__init__()
        # self.h = h
        self.subbands = subbands
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u+1-i)//2, output_padding=1-i)))  # 这里k和u不是成倍数的关系，对最终结果很有可能是有影响的，会有checkerboard artifacts的现象

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.post_n_fft = gen_istft_n_fft
        self.ups.apply(commons.init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.reshape_pixelshuffle = []

        self.subband_conv_post = weight_norm(
            Conv1d(ch, self.subbands*(self.post_n_fft + 2), 7, 1, padding=3))

        self.subband_conv_post.apply(commons.init_weights)

        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size

        updown_filter = torch.zeros(
            (self.subbands, self.subbands, self.subbands)).float()
        for k in range(self.subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.multistream_conv_post = weight_norm(
            Conv1d(4, 1, kernel_size=63, bias=False, padding=commons.get_padding(63, 1)))
        self.multistream_conv_post.apply(commons.init_weights)
        self.cond = nn.Conv1d(256, 512, 1)

    def forward(self, x, g=None):
        stft = TorchSTFT(filter_length=self.gen_istft_n_fft,
                                     hop_length=self.gen_istft_hop_size, win_length=self.gen_istft_n_fft).to(x.device)
        # pqmf = PQMF(x.device)

        x = self.conv_pre(x)  # [B, ch, length]
        # print(x.size(),g.size())
        x = x + self.cond(g)  # g [b, 256, 1] => cond(g) [b, 512, 1]

        for i in range(self.num_upsamples):

            # print(x.size(),g.size())
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            # print(x.size(),g.size())
            x = self.ups[i](x)

            # print(x.size(),g.size())
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        # print(x.size(),g.size())
        x = F.leaky_relu(x)
        x = self.reflection_pad(x)
        x = self.subband_conv_post(x)
        x = torch.reshape(
            x, (x.shape[0], self.subbands, x.shape[1]//self.subbands, x.shape[-1]))
        # print(x.size(),g.size())
        spec = torch.exp(x[:, :, :self.post_n_fft // 2 + 1, :])
        phase = math.pi*torch.sin(x[:, :, self.post_n_fft // 2 + 1:, :])
        # print(spec.size(),phase.size())
        y_mb_hat = stft.inverse(torch.reshape(spec, (spec.shape[0]*self.subbands, self.gen_istft_n_fft // 2 + 1, spec.shape[-1])), torch.reshape(
            phase, (phase.shape[0]*self.subbands, self.gen_istft_n_fft // 2 + 1, phase.shape[-1])))
        # print(y_mb_hat.size())
        y_mb_hat = torch.reshape(
            y_mb_hat, (x.shape[0], self.subbands, 1, y_mb_hat.shape[-1]))
        # print(y_mb_hat.size())
        y_mb_hat = y_mb_hat.squeeze(-2)
        # print(y_mb_hat.size())
        # .cuda(x.device) * self.subbands, stride=self.subbands)
        y_mb_hat = F.conv_transpose1d(
            y_mb_hat, self.updown_filter * self.subbands, stride=self.subbands)
        # print(y_mb_hat.size())
        y_g_hat = self.multistream_conv_post(y_mb_hat)
        # print(y_g_hat.size(),y_mb_hat.size())
        return y_g_hat, y_mb_hat

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class SpeakerEncoder(torch.nn.Module):
    def __init__(self, mel_n_channels=80, model_num_layers=3, model_hidden_size=256, model_embedding_size=256):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size,
                            model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    def compute_partial_slices(self, total_frames, partial_frames, partial_hop):
        mel_slices = []
        for i in range(0, total_frames-partial_frames, partial_hop):
            mel_range = torch.arange(i, i+partial_frames)
            mel_slices.append(mel_range)

        return mel_slices

    def embed_utterance(self, mel, partial_frames=128, partial_hop=64):
        mel_len = mel.size(1)
        last_mel = mel[:, -partial_frames:]

        if mel_len > partial_frames:
            mel_slices = self.compute_partial_slices(
                mel_len, partial_frames, partial_hop)
            mels = list(mel[:, s] for s in mel_slices)
            mels.append(last_mel)
            mels = torch.stack(tuple(mels), 0).squeeze(1)

            with torch.no_grad():
                partial_embeds = self(mels)
            embed = torch.mean(partial_embeds, axis=0).unsqueeze(0)
            # embed = embed / torch.linalg.norm(embed, 2)
        else:
            with torch.no_grad():
                embed = self(last_mel)

        return embed


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self,
                 spec_channels=641,
                 segment_size=32,
                 inter_channels=192,
                 hidden_channels=192,
                 filter_channels=768,
                 n_heads=2,
                 n_layers=6,
                 kernel_size=3,
                 p_dropout=0.1,
                 resblock="1",
                 resblock_kernel_sizes=[3, 7, 11],
                 resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                 upsample_rates=[5, 4],
                 upsample_initial_channel=512,
                 upsample_kernel_sizes=[16, 16],
                 gen_istft_n_fft=16,
                 gen_istft_hop_size=4,
                 n_speakers=0,
                 gin_channels=256,
                 use_sdp=False,
                 ms_istft_vits=True,
                 mb_istft_vits=False,
                 subbands=4,
                 istft_vits=False,
                 **kwargs):

        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.ms_istft_vits = ms_istft_vits
        self.mb_istft_vits = mb_istft_vits
        self.istft_vits = istft_vits

        self.use_sdp = use_sdp

        # 768, inter_channels, hidden_channels, 5, 1, 16)
        self.enc_p = PosteriorEncoder(
            256, inter_channels, hidden_channels, 5, 1, 16)

        if ms_istft_vits == True:
            print('Mutli-stream iSTFT VITS')
            self.dec = Multistream_iSTFT_Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                                                   upsample_initial_channel, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, subbands, gin_channels=gin_channels)
        else:
            print('Decoder Error in json file')

        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        self.enc_spk = SpeakerEncoder(
            model_hidden_size=gin_channels, model_embedding_size=gin_channels)

    def forward(self, c, spec, g=None, mel=None, c_lengths=None, spec_lengths=None):
        if c_lengths == None:
            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        if spec_lengths == None:
            spec_lengths = (torch.ones(spec.size(0)) *
                            spec.size(-1)).to(spec.device)

        g = self.enc_spk(mel.transpose(1, 2))
        g = g.unsqueeze(-1)

        _, m_p, logs_p, _ = self.enc_p(c, c_lengths)
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
        z_p = self.flow(z, spec_mask, g=g)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, spec_lengths, self.segment_size)
        o, o_mb = self.dec(z_slice, g=g)

        return o, o_mb, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, c, g=None, mel=None, c_lengths=None):
        if c_lengths == None:
            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        g = self.enc_spk.embed_utterance(mel.transpose(1, 2))
        g = g.unsqueeze(-1)

        z_p, m_p, logs_p, c_mask = self.enc_p(c, c_lengths)
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        o, o_mb = self.dec(z * c_mask, g=g)

        return o
