import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math
import numpy as np
from functools import partial
from einops import rearrange, repeat

from huggingface_hub import PyTorchModelHubMixin
from local_attention import LocalAttention
import torch.nn.functional as F

from utils.utils_env import AttrDict


class WaveGeneratorOscillator(nn.Module):
    """
        synthesize audio with a sawtooth oscillator.
        the sawtooth oscillator is synthesized by a bank of sinusoids
    """
    def __init__(
            self, 
            fs, 
            amplitudes,
            ratio,
            is_remove_above_nyquist=True):
        super().__init__()
        self.fs = fs
        self.is_remove_above_nyquist = is_remove_above_nyquist
        self.amplitudes = amplitudes
        self.ratio = ratio
        self.n_harmonics = len(amplitudes)

    def _remove_above_nyquist(self, amplitudes, pitch, sampling_rate):
        n_harm = amplitudes.shape[-1]
        harmonic_freqs = pitch * torch.arange(1, n_harm + 1, device=pitch.device)
        mask = (harmonic_freqs < sampling_rate / 2).float() + 1e-7
        return amplitudes * mask
        
    def forward(self, f0, initial_phase=None):
        '''
                    f0: B x T x 1 (Hz)
         initial_phase: B x 1 x 1
          ---
              signal: B x T
         final_phase: B x 1 x 1
        '''
        if initial_phase is None:
            initial_phase = torch.zeros(f0.shape[0], 1, 1).to(f0)
        
        mask = (f0 > 0).detach()
        f0 = f0.detach()

        # harmonic synth
        phase  = torch.cumsum(2 * np.pi * f0 / self.fs, axis=1) + initial_phase
        phases = phase * torch.arange(1, self.n_harmonics + 1).to(phase)
        
        # anti-aliasing
        amplitudes = self.amplitudes * self.ratio
        if self.is_remove_above_nyquist:
            amp = self._remove_above_nyquist(amplitudes.to(phase), f0, self.fs)
        else:
            amp = amplitudes.to(phase)
        
        # signal
        signal = (torch.sin(phases) * amp).sum(-1, keepdim=True)
        signal *= mask
        signal = signal.squeeze(-1)
        
        # phase
        final_phase = phase[:, -1:, :] % (2 * np.pi)
 
        return signal, final_phase.detach()

exists = lambda val: val is not None
empty = lambda tensor: tensor.numel() == 0
default = lambda val, d: val if exists(val) else d
cast_tuple = lambda val: (val,) if not isinstance(val, tuple) else val

class PCmer(nn.Module):
    """The encoder that is used in the Transformer model with Conformer convolution modules."""
    
    def __init__(self, 
                num_layers,
                num_heads,
                dim_model,
                dim_keys,
                dim_values,
                residual_dropout,
                attention_dropout):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.dim_keys = dim_keys
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout

        self._layers = nn.ModuleList([self._create_encoder_layer() for _ in range(num_layers)])
    
    def _create_encoder_layer(self):
        """Create a single encoder layer with SelfAttention and Conformer convolution"""
        return nn.ModuleDict({
            'norm': nn.LayerNorm(self.dim_model),
            'attn': SelfAttention(
                dim=self.dim_model,
                heads=self.num_heads,
                causal=False
            ),
            'conformer': ConformerConvModule(self.dim_model),
            'dropout': nn.Dropout(self.residual_dropout)
        })
    
    def forward(self, phone, mask=None):
        """Forward pass through all encoder layers"""
        for layer in self._layers:
            # Self-attention sub-layer with residual connection
            phone = phone + layer['attn'](layer['norm'](phone), mask=mask)
            # Conformer convolution sub-layer with residual connection
            phone = phone + layer['conformer'](phone)
        
        return phone


# helper classes

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = self._calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            #nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Transpose((1, 2)),
            nn.Dropout(dropout)
        )

    def _calc_same_padding(self, kernel_size):
        pad = kernel_size // 2
        return (pad, pad - (kernel_size + 1) % 2)

    def forward(self, x):
        return self.net(x)


class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0, causal=False, 
                 generalized_attention=False, kernel_fn=nn.ReLU(), qr_uniform_q=False, 
                 no_projection=False):
        super().__init__()
        
        self.dim_heads = dim_heads
        self.nb_features = nb_features or int(dim_heads * math.log(dim_heads))
        self.ortho_scaling = ortho_scaling
        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn
        self.no_projection = no_projection
        self.causal = causal
        self.qr_uniform_q = qr_uniform_q

        # Create projection matrix
        self.register_buffer('projection_matrix', self._create_projection_matrix())
        
        # Setup causal attention function
        self.causal_linear_fn = self._setup_causal_attention()

    def _orthogonal_matrix_chunk(self, cols, device=None):
        """Generate a chunk of orthogonal matrix"""
        unstructured = torch.randn((cols, cols), device=device)
        q, r = torch.linalg.qr(unstructured, mode='reduced')
        
        if self.qr_uniform_q:
            q *= torch.diag(r).sign().unsqueeze(0)
        
        return q.T

    def _gaussian_orthogonal_random_matrix(self, nb_rows, nb_columns, device=None):
        """Generate Gaussian orthogonal random matrix"""
        # Generate full blocks
        nb_full_blocks = nb_rows // nb_columns
        blocks = [self._orthogonal_matrix_chunk(nb_columns, device) 
                 for _ in range(nb_full_blocks)]
        
        # Handle remaining rows
        remaining_rows = nb_rows - nb_full_blocks * nb_columns
        if remaining_rows > 0:
            q = self._orthogonal_matrix_chunk(nb_columns, device)
            blocks.append(q[:remaining_rows])
        
        final_matrix = torch.cat(blocks)
        
        # Apply scaling
        if self.ortho_scaling == 0:
            multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
        elif self.ortho_scaling == 1:
            multiplier = math.sqrt(nb_columns) * torch.ones(nb_rows, device=device)
        else:
            raise ValueError(f'Invalid scaling {self.ortho_scaling}')
        
        return torch.diag(multiplier) @ final_matrix

    def _create_projection_matrix(self):
        """Create the projection matrix for attention"""
        return self._gaussian_orthogonal_random_matrix(
            nb_rows=self.nb_features, 
            nb_columns=self.dim_heads
        )

    def _setup_causal_attention(self):
        """Setup causal attention function"""
        if not self.causal:
            return None
            
        try:
            import fast_transformers.causal_product.causal_product_cuda
            return partial(causal_linear_attention)
        except ImportError:
            print('CUDA not available for causal attention, using CPU version')
            return causal_linear_attention_noncuda

    @torch.no_grad()
    def redraw_projection_matrix(self, device=None):
        """Redraw the projection matrix (for Performer)"""
        device = device or self.projection_matrix.device
        new_projection = self._create_projection_matrix().to(device)
        self.projection_matrix.copy_(new_projection)

    def _softmax_kernel(self, data, *, is_query, normalize_data=True, eps=1e-4):
        """Softmax kernel for attention approximation"""
        b, h, *_ = data.shape
        
        # Data normalization
        data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.
        ratio = (self.projection_matrix.shape[0] ** -0.5)

        # Projection calculation
        projection = self.projection_matrix.expand(b, h, -1, -1)
        data_dash = torch.einsum('...id,...jd->...ij', data_normalizer * data, projection)

        # Diagonal term calculation
        diag_data = (data ** 2).sum(dim=-1, keepdim=True) * (data_normalizer ** 2) / 2

        if is_query:
            data_dash = ratio * torch.exp(data_dash - diag_data - data_dash.amax(dim=-1, keepdim=True) + eps)
        else:
            data_dash = ratio * torch.exp(data_dash - diag_data + eps)

        return data_dash

    def _linear_attention(self, q, k, v):
        """Linear attention computation"""
        if v is None:
            return torch.einsum('...ed,...nd->...ne', k, q)

        k_cumsum = k.sum(dim=-2)
        D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
        context = torch.einsum('...nd,...ne->...de', k, v)
        return torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)

    def forward(self, q, k, v):
        """Forward pass of fast attention"""
        if self.no_projection:
            q = q.softmax(dim=-1)
            k = torch.exp(k) if self.causal else k.softmax(dim=-2)
        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn=self.kernel_fn, 
                                  projection_matrix=self.projection_matrix, device=q.device)
            q, k = map(create_kernel, (q, k))
        else:
            q = self._softmax_kernel(q, is_query=True)
            k = self._softmax_kernel(k, is_query=False)

        # Select attention function
        attn_fn = self.causal_linear_fn if self.causal else self._linear_attention
        return attn_fn(q, k, v)
    

class SelfAttention(nn.Module):
    def __init__(self, dim, causal = False, heads = 8, dim_head = 64, local_heads = 0, local_window_size = 256, nb_features = None, feature_redraw_interval = 1000, generalized_attention = False, kernel_fn = nn.ReLU(), qr_uniform_q = False, dropout = 0., no_projection = False):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, nb_features, causal = causal, generalized_attention = generalized_attention, kernel_fn = kernel_fn, qr_uniform_q = qr_uniform_q, no_projection = no_projection)

        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = LocalAttention(window_size = local_window_size, causal = causal, autopad = True, dropout = dropout, look_forward = int(not causal), rel_pos_emb_config = (dim_head, local_heads)) if local_heads > 0 else None

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def redraw_projection_matrix(self):
        self.fast_attention.redraw_projection_matrix()

    def forward(self, x, context = None, mask = None, context_mask = None, name=None, inference=False, **kwargs):
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads
        
        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask
        #print (torch.sum(self.name_embedding))
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []
        #print (name)
        #print (self.name_embedding[name].size())
        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.)
            if cross_attend:
                pass
                #print (torch.sum(self.name_embedding))
                #out = self.fast_attention(q,self.name_embedding[name],None)
                #print (torch.sum(self.name_embedding[...,-1:]))
            else:
                out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask = mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim = 1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return self.dropout(out)


class Mel2Control(nn.Module):
    def __init__(
            self,
            input_channel,
            output_splits):
        super().__init__()
        self.output_splits = output_splits

        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 64, 3, 1, 1),
                nn.GroupNorm(4, 64),
                nn.LeakyReLU(),
                nn.Conv1d(64, 64, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=64,
            dim_keys=64,
            dim_values=64,
            residual_dropout=0.1,
            attention_dropout=0.1)
        self.norm = nn.LayerNorm(64)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(64, self.n_out))
        
    def _split_to_dict(self, tensor, tensor_splits):
        """Split a tensor into a dictionary of multiple tensors."""
        labels = []
        sizes = []

        for k, v in tensor_splits.items():
            labels.append(k)
            sizes.append(v)

        tensors = torch.split(tensor, sizes, dim=-1)
        return dict(zip(labels, tensors))

    def forward(self, x):
        
        '''
        input: 
            B x n_frames x n_mels
        return: 
            dict of B x n_frames x feat
        '''

        x = self.stack(x.transpose(1,2)).transpose(1,2)
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = self._split_to_dict(e, self.output_splits)
    
        return controls

    
def apply_window_to_impulse_response(impulse_response,
                                     window_size: int = 0,
                                     causal: bool = False):
    """Apply a window to an impulse response and put in causal form.
    Args:
        impulse_response: A series of impulse responses frames to window, of shape
        [batch, n_frames, ir_size]. ---------> ir_size means size of filter_bank ??????
        
        window_size: Size of the window to apply in the time domain. If window_size
        is less than 1, it defaults to the impulse_response size.
        causal: Impulse response input is in causal form (peak in the middle).
    Returns:
        impulse_response: Windowed impulse response in causal form, with last
        dimension cropped to window_size if window_size is greater than 0 and less
        than ir_size.
    """
    
    
    # If IR is in causal form, put it in zero-phase form.
    if causal:
        impulse_response = torch.fftshift(impulse_response, axes=-1)
    
    # Get a window for better time/frequency resolution than rectangular.
    # Window defaults to IR size, cannot be bigger.
    #ir_size = int(impulse_response.shape[-1])
    ir_size = int(impulse_response.size(-1))
    if (window_size <= 0) or (window_size > ir_size):
        window_size = ir_size
    window = nn.Parameter(torch.hann_window(window_size), requires_grad = False).cuda()
    # Zero pad the window and put in in zero-phase form.
    
    padding = ir_size - window_size
    if padding > 0:
        half_idx = (window_size + 1) // 2
        window = torch.cat([window[half_idx:],
                            torch.zeros([padding]),
                            window[:half_idx]], axis=0)
    else:
        window = window.roll((window.size(-1)+1)//2, -1)
        
    # Apply the window, to get new IR (both in zero-phase form).

    window = window.unsqueeze(0)
    impulse_response = impulse_response*window
    
    # Put IR in causal form and trim zero padding.
    if padding > 0:
        first_half_start = (ir_size - (half_idx - 1)) + 1
        second_half_end = half_idx + 1
        impulse_response = torch.cat([impulse_response[..., first_half_start:],
                                    impulse_response[..., :second_half_end]],
                                    dim=-1)
    else:
        impulse_response = impulse_response.roll((impulse_response.size(-1)+1)//2, -1)

    return impulse_response

def frequency_impulse_response(magnitudes,
                               window_size: int = 0):
    """Get windowed impulse responses using the frequency sampling method.
    Follows the approach in:
    https://ccrma.stanford.edu/~jos/sasp/Windowing_Desired_Impulse_Response.html
    Args:
        magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
        n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
        last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
        f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
        audio into equally sized frames to match frames in magnitudes.
        window_size: Size of the window to apply in the time domain. If window_size
        is less than 1, it defaults to the impulse_response size.
    Returns:
        impulse_response: Time-domain FIR filter of shape
        [batch, frames, window_size] or [batch, window_size].
    Raises:
        ValueError: If window size is larger than fft size.
    """
    # Get the IR (zero-phase form).
    
    magnitudes = torch.complex(magnitudes, torch.zeros_like(magnitudes))
    impulse_response = torch.fft.irfft(magnitudes)
    
    #print ("impulse response size here:", impulse_response[0,0, :5], impulse_response[0,0, -5:])

    """ This means this?
    First: Initilize at fourier space, where real part equal magnitueds, complex part equal 0
    Second: Convert back to time domain
    """ 
    
    # Window and put in causal form.
    impulse_response = apply_window_to_impulse_response(impulse_response,
                                                        impulse_response.size(-1))
    return impulse_response
    
def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True) -> int:
  """Calculate final size for efficient FFT.
  Args:
    frame_size: Size of the audio frame.
    ir_size: Size of the convolving impulse response.
    power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
      numbers. TPU requires power of 2, while GPU is more flexible.
  Returns:
    fft_size: Size for efficient FFT.
  """
  convolved_frame_size = ir_size + frame_size - 1
  if power_of_2:
    # Next power of 2.
    fft_size = int(2**np.ceil(np.log2(convolved_frame_size)))
  else:
    fft_size = int(fftpack.helper.next_fast_len(convolved_frame_size))
  return fft_size

def fft_convolve(audio,
                 impulse_response,
                 padding = 'same',
                 delay_compensation = -1,
                 mel_scale_noise = False):
    """Filter audio with frames of time-varying impulse responses.
    Time-varying filter. Given audio [batch, n_samples], and a series of impulse
    responses [batch, n_frames, n_impulse_response], splits the audio into frames,
    applies filters, and then overlap-and-adds audio back together.
    Applies non-windowed non-overlapping STFT/ISTFT to efficiently compute
    convolution for large impulse response sizes.
    Args:
        audio: Input audio. Tensor of shape [batch, audio_timesteps].
        impulse_response: Finite impulse response to convolve. Can either be a 2-D
        Tensor of shape [batch, ir_size], or a 3-D Tensor of shape [batch,
        ir_frames, ir_size]. A 2-D tensor will apply a single linear
        time-invariant filter to the audio. A 3-D Tensor will apply a linear
        time-varying filter. Automatically chops the audio into equally shaped
        blocks to match ir_frames.
        padding: Either 'valid' or 'same'. For 'same' the final output to be the
        same size as the input audio (audio_timesteps). For 'valid' the audio is
        extended to include the tail of the impulse response (audio_timesteps +
        ir_timesteps - 1).
        delay_compensation: Samples to crop from start of output audio to compensate
        for group delay of the impulse response. If delay_compensation is less
        than 0 it defaults to automatically calculating a constant group delay of
        the windowed linear phase filter from frequency_impulse_response().
    Returns:
        audio_out: Convolved audio. Tensor of shape
            [batch, audio_timesteps + ir_timesteps - 1] ('valid' padding) or shape
            [batch, audio_timesteps] ('same' padding).
    Raises:
        ValueError: If audio and impulse response have different batch size.
        ValueError: If audio cannot be split into evenly spaced frames. (i.e. the
        number of impulse response frames is on the order of the audio size and
        not a multiple of the audio size.)
    """
    #audio, impulse_response = tf_float32(audio), tf_float32(impulse_response)

    # Add a frame dimension to impulse response if it doesn't have one.
    #ir_shape = impulse_response.shape.as_list()
    ir_shape = impulse_response.size()
    if len(ir_shape) == 2:
        impulse_response = impulse_response.unsqueeze(1)
        #ir_shape = impulse_response.shape.as_list()
        ir_shape = impulse_response.size()

    # Get shapes of audio and impulse response.
    batch_size_ir, n_ir_frames, ir_size = ir_shape
    batch_size, audio_size = audio.size()#shape.as_list()

    # Validate that batch sizes match.
    if batch_size != batch_size_ir:
        raise ValueError('Batch size of audio ({}) and impulse response ({}) must '
                        'be the same.'.format(batch_size, batch_size_ir))

    # Cut audio into frames.
    frame_size = int(np.ceil(audio_size / n_ir_frames))
    hop_size = frame_size
    
    # audio --> batch, time_step
    audio = audio.unsqueeze(1)
    if frame_size!=audio_size:
        filters = torch.eye(frame_size).unsqueeze(1).cuda()
        audio_frames = F.conv1d(audio, filters, stride= hop_size).transpose(1,2)
        n_audio_frams = audio_frames.size(1)
    else:
        audio_frames = audio#.transpose(1,2)
    # Check that number of frames match.
    #print ("audio_frames here:", audio_frames.size())
    n_audio_frames = int(audio_frames.shape[1])

    if n_audio_frames != n_ir_frames:
        raise ValueError(
            'Number of Audio frames ({}) and impulse response frames ({}) do not '
            'match. For small hop size = ceil(audio_size / n_ir_frames), '
            'number of impulse response frames must be a multiple of the audio '
            'size.'.format(n_audio_frames, n_ir_frames))
    
    # Pad and FFT the audio and impulse responses.
    fft_size = get_fft_size(frame_size, ir_size, power_of_2=True)
    
    audio_fft = torch.fft.rfft(audio_frames, fft_size)

    ir_fft = torch.fft.rfft(impulse_response, fft_size)
    # Multiply the FFTs (same as convolution in time).
    audio_ir_fft = torch.multiply(audio_fft, ir_fft)

    # Take the IFFT to resynthesize audio.
    audio_frames_out = torch.fft.irfft(audio_ir_fft)
    #audio_out = torch.signal.overlap_and_add(audio_frames_out, hop_size)
    if frame_size!=audio_size:
        overlap_add_filter = torch.eye(audio_frames_out.size(-1), requires_grad = False).unsqueeze(1).cuda()
        #print (overlap_add_filter.size())
        #print (audio_frames_out.size(), overlap_add_filter.size())
        output_signal = F.conv_transpose1d(audio_frames_out.transpose(1, 2), 
                                                        overlap_add_filter, 
                                                        stride = frame_size, 
                                                        padding = 0).squeeze(1)
    else:
        output_signal = crop_and_compensate_delay(audio_frames_out.squeeze(1), audio_size, ir_size, padding,
                                    delay_compensation)
    return output_signal[...,:frame_size*n_ir_frames]

def frequency_filter(audio,
                     magnitudes,
                     window_size = 0,
                     padding = 'same',
                     mel_scale_noise = False,
                     mel_basis = None):
    """Filter audio with a finite impulse response filter.
    Args:
        audio: Input audio. Tensor of shape [batch, audio_timesteps].
        magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
        n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
        last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
        f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
        audio into equally sized frames to match frames in magnitudes.
        window_size: Size of the window to apply in the time domain. If window_size
        is less than 1, it is set as the default (n_frequencies).
        padding: Either 'valid' or 'same'. For 'same' the final output to be the
        same size as the input audio (audio_timesteps). For 'valid' the audio is
        extended to include the tail of the impulse response (audio_timesteps +
        window_size - 1).
    Returns:
        Filtered audio. Tensor of shape
            [batch, audio_timesteps + window_size - 1] ('valid' padding) or shape
            [batch, audio_timesteps] ('same' padding).
    """

    impulse_response = frequency_impulse_response(magnitudes,
                                                    window_size=window_size)
    
    return fft_convolve(audio, impulse_response, padding=padding, mel_scale_noise=mel_scale_noise)

class SawSinSub(
    torch.nn.Module,
    PyTorchModelHubMixin,
    library_name="sawsinsub",
    repo_url="https://github.com/YatingMusic/ddsp-singing-vocoders",
    docs_url="https://github.com/YatingMusic/ddsp-singing-vocoders/blob/main/README.md",
    pipeline_tag="audio-to-audio",
    license="mit",
    tags=["neural-vocoder", "audio-generation", "arxiv:2208.04756"],
):
    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False):
        super().__init__()
        self.h = h

        # params
        self.register_buffer("sampling_rate", torch.tensor(h.sampling_rate))
        self.register_buffer("block_size", torch.tensor(h.hop_size))
        
        # Mel2Control
        split_map = {
            'f0': 1, 
            'harmonic_magnitude': h.n_mag_harmonic,
            'noise_magnitude': h.n_mag_noise
        }
        self.mel2ctrl = Mel2Control(h.num_mels, split_map)

        # Harmonic Synthsizer
        self.harmonic_amplitudes = nn.Parameter(
            1. / torch.arange(1, h.n_harmonics + 1).float(), requires_grad=False)
        self.ratio = nn.Parameter(torch.tensor([0.4]).float(), requires_grad=False)

        self.harmonic_synthsizer = WaveGeneratorOscillator(
            h.sampling_rate,
            amplitudes=self.harmonic_amplitudes,
            ratio=self.ratio)
        
    def _logb(self, x, base=2.0, safe=False):
        """Logarithm with base as an argument."""
        if safe:
            return safe_divide(safe_log(x), safe_log(base))
        else:
            return torch.log(torch.tensor(x)) / torch.log(torch.tensor(base))   
    
    def _hz_to_midi(self, frequencies):
        """TF-compatible hz_to_midi function."""
        notes = 12.0 * (self._logb(frequencies, 2.0) - self._logb(torch.tensor(440.0).float(), 2.0)) + 69.0
        # Map 0 Hz to MIDI 0 (Replace -inf MIDI with 0.)
        notes = F.relu(notes)
        return notes.float()

    def _midi_to_hz(self, notes):
        """TF-compatible midi_to_hz function."""
        return 440.0 * (2.0**((notes - 69.0) / 12.0))

    def _unit_to_hz2(self, unit,
                hz_min,
                hz_max,
                clip: bool = False) :
        """Map unit interval [0, 1] to [hz_min, hz_max], scaling logarithmically."""
        midi_max = self._hz_to_midi(hz_max)
        midi_min = self._hz_to_midi(hz_min)
        return self._midi_to_hz((midi_max - midi_min) * unit + midi_min)
    
    def _upsample(self, signal, factor):
        signal = signal.permute(0, 2, 1)
        signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
        return signal.permute(0, 2, 1)
        
    def _scale_function(self, x):
        return 2 * torch.sigmoid(x)**(np.log(10)) + 1e-7

    def infer(self, data, initial_phase=None):
        mel = data["x"]
        if len(mel.shape) < 3:
            mel = mel.unsqueeze(0)
        mel = mel.permute(0, 2, 1)
        ctrls = self.mel2ctrl(mel)

        f0_unit = ctrls["f0"]
        f0_unit = torch.sigmoid(f0_unit)
        f0 = self._unit_to_hz2(f0_unit, hz_min=80.0, hz_max=1000.0)
        f0[f0<80] = 0.

        pitch = f0

        src_param = self._scale_function(ctrls['harmonic_magnitude'])
        noise_param = self._scale_function(ctrls['noise_magnitude'])

        # exciter signal
        B, n_frames, _ = pitch.shape

        # upsample
        pitch = self._upsample(pitch, self.block_size)

        # harmonic
        harmonic, final_phase = self.harmonic_synthsizer(pitch, initial_phase)
        harmonic = frequency_filter(
                        harmonic,
                        src_param)

        # noise part
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = frequency_filter(
                        noise,
                        noise_param)
        signal = harmonic + noise
        signal = signal.unsqueeze(1)

        return {"y_g_hat": signal}

    def forward(self, data, initial_phase=None):
        '''
            mel: B x n_frames x n_mels
        '''
        mel = data["x"]
        mel = mel.permute(0, 2, 1)
        ctrls = self.mel2ctrl(mel)

        # unpack
        f0_unit = ctrls['f0']# units
        f0_unit = torch.sigmoid(f0_unit)
        f0 = self._unit_to_hz2(f0_unit, hz_min=80.0, hz_max=1000.0)
        f0[f0<80] = 0.

        pitch = f0
        
        src_param = self._scale_function(ctrls['harmonic_magnitude'])
        noise_param = self._scale_function(ctrls['noise_magnitude'])

        # exciter signal
        B, n_frames, _ = pitch.shape

        # upsample
        pitch = self._upsample(pitch, self.block_size)

        # harmonic
        harmonic, final_phase = self.harmonic_synthsizer(pitch, initial_phase)
        harmonic = frequency_filter(
                        harmonic,
                        src_param)

        # noise part
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = frequency_filter(
                        noise,
                        noise_param)
        signal = harmonic + noise
        signal = signal.unsqueeze(1)

        return {"y_g_hat": signal, "f0": f0.squeeze(), "final_phase": final_phase, "parts": (harmonic, noise),  "params": (src_param, noise_param)}
