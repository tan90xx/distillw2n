# Copied from https://github.com/chaufanglin/Normal2Whisper/blob/main/utils.py
import numpy as np
from scipy.signal import lfilter
import soundfile as sf
import librosa
from librosa import lpc
import pyworld as pw

def wav2world(x, fs, fft_size=None):
    """Convenience function to do all WORLD analysis steps in a single call.
    In this case only `frame_period` can be configured and other parameters
    are fixed to their defaults. Likewise, F0 estimation is fixed to
    DIO plus StoneMask refinement.
    Parameters
    ----------
    x : ndarray
        Input waveform signal.
    fs : int
        Sample rate of input signal in Hz.
    fft_size : int
        Length of Fast Fourier Transform (in number of samples)
        The resulting dimension of `ap` adn `sp` will be `fft_size` // 2 + 1
    Returns
    -------
    f0 : ndarray
        F0 contour.
    sp : ndarray
        Spectral envelope.
    ap : ndarray
        Aperiodicity.
    t  : ndarray
        Temporal position of each frame.
    """
    f0, t = pw.harvest(x, fs)
    sp = pw.cheaptrick(x, f0, t, fs, fft_size=fft_size)
    ap = pw.d4c(x, f0, t, fs, fft_size=fft_size)
    return f0, sp, ap, t


def moving_average(data, length):
    output = np.empty(data.shape)
    maf = np.bartlett(length)/length  # Bartlett window is a triangular window
    for i in range(data.shape[0]):
        output[i,:] = np.convolve(data[i,:], maf,'same')
    return output


def gfm_iaif_glottal_remove(s_gvl, nv=48, ng=3, d=0.99, win=None):
    """
    Glootal removal function based on GFM-IAIF.

    Note:
    Function originally coded by Olivier Perrotin (https://github.com/operrotin/GFM-IAIF). 
    This code is translated to Python and adapted by Zhaofeng Lin (linzh@tcd.ie)
    Parameters:
    ----------
        s_gvl: Speech signal frame
        nv: Order of LP analysis for vocal tract (def. 48)
        ng: Order of LP analysis for glottal source (def. 3)
        d: Leaky integration coefficient (def. 0.99)
        win: Window used before LPC (def. Hanning)

    Returns:
    -------
        s_v: Speech signal with glottis contribution cancelled 
    """

    # ----- Set default parameters -------------------------------------------
    if win is None:
        # Window for LPC estimation
        win = np.hanning(len(s_gvl))

    # ----- Addition of pre-frame --------------------------------------------
    # For the successive removals of the estimated LPC envelopes, a
    # mean-normalized pre-frame ramp is added at the beginning of the frame
    # in order to diminish ripple. The ramp is removed after each filtering.
    Lpf = nv + 1  # Pre-frame length
    x_gvl = np.concatenate([np.linspace(-s_gvl[0], s_gvl[0], Lpf), s_gvl])  # Prepend
    idx_pf = np.arange(Lpf, len(x_gvl))  # Indexes that exclude the pre-frame

    # ----- Cancel lip radiation contribution --------------------------------
    # Define lip radiation filter
    al = [1, -d]

    # Integration of signal using filter 1/[1 -d z^(-1)]
    # - Input signal (for LPC estimation)
    s_gv = lfilter([1], al, s_gvl)
    # - Pre-framed input signal (for LPC envelope removal)
    x_gv = lfilter([1], al, x_gvl)

    # ----- Gross glottis estimation -----------------------------------------
    # Iterative estimation of glottis with ng first order filters
    ag1 = lpc(s_gv*win, order=1)         # First 1st order LPC estimation

    for i in range(ng-2):
        # Cancel current estimate of glottis contribution from speech signal
        x_v1x = lfilter(ag1,1,x_gv)        # Inverse filtering
        s_v1x = x_v1x[idx_pf]        # Remove pre-ramp

        # Next 1st order LPC estimation
        ag1x = lpc(s_v1x*win, order=1)        # 1st order LPC

        # Update gross estimate of glottis contribution
        ag1 = np.convolve(ag1,ag1x)        # Combine 1st order estimation with previous


    # ----- Gross vocal tract estimation -------------------------------------
    # Cancel gross estimate of glottis contribution from speech signal
    x_v1 = lfilter(ag1,1,x_gv)       # Inverse filtering
    s_v1 = x_v1[idx_pf]         # Remove pre-ramp

    # Gross estimate of the vocal tract filter
    av1 = lpc(s_v1*win, order=nv)        # nv order LPC estimation

    # ----- Fine glottis estimation ------------------------------------------
    # Cancel gross estimate of vocal tract contribution from speech signal
    x_g1 = lfilter(av1,1,x_gv)       # Inverse filtering
    s_g1 = x_g1[idx_pf]         # Remove pre-ramp

    # Fine estimate of the glottis filter
    ag = lpc(s_g1*win, order=ng)        # ng order LPC estimation

    # ----- Fine vocal tract estimation --------------------------------------
    # Cancel fine estimate of glottis contribution from speech signal
    x_v = lfilter(ag,1,x_gv)       # Inverse filtering
    s_v = x_v[idx_pf]         # Remove pre-ramp

    return s_v


def pesudo_whisper_gen(s_n, fs, Lv=16):
    """
    Pesudo whispered speech generating function, using GFM-IAIF and moving averge filtering.

    Note:
    This code is written by Zhaofeng Lin (linzh@tcd.ie)

    Parameters:
    ----------
        s_n: Normal speech wavform 
        fs: Sample rate
        Lv: order of LP analysis for vocal tract (default: 16)

    Returns:
    -------
        y_pw: Pesudo whispered speech wavform
    """

    EPSILON = 1e-8

    # Overlapp-add (OLA) method
    nfft = pw.get_cheaptrick_fft_size(fs)
    win_length = int(30*fs/1000) # 30ms * fs / 1000
    nhop = round(win_length / 2)
    window = np.hamming(win_length)
    nframes = int(np.ceil(s_n.size / nhop))

    s_gfm = np.zeros(s_n.shape)     # allocate output speech without glottal source

    for n in range(nframes):
        startPoint = n * nhop     # starting point of windowing
        if startPoint + win_length > s_n.size:
            s_gfm[startPoint - nhop + win_length: ] = EPSILON
            continue
        else:
            sn_frame = s_n[startPoint : startPoint+win_length] * window

        s_gfm_frame = gfm_iaif_glottal_remove(sn_frame, Lv)

        s_gfm[startPoint: startPoint + win_length] = s_gfm[startPoint: startPoint + win_length] + s_gfm_frame

    # Extract GFM
    f0_gfm, sp_gfm, ap_gfm, _ = wav2world(s_gfm, fs)

    # Moving Averge Filtering
    maf_freq = 400  # 400 Hz
    maf_w_len = round(maf_freq/fs * nfft)    # 400 Hz
    sp_maf = moving_average(sp_gfm, maf_w_len)

    # Zero F0 and unit Ap
    f0_zero = np.zeros(f0_gfm.shape) + EPSILON
    ap_unit = np.ones(ap_gfm.shape) - EPSILON

    y_pw = pw.synthesize(f0_zero, sp_maf, ap_unit, fs, pw.default_frame_period)

    return y_pw


def process_wav(in_path, out_path, sample_rate):
    normal, fs_ = sf.read(in_path)
    if sample_rate != fs_:
        normal = librosa.resample(normal, fs_, sample_rate)
    pesudo_whisper = pesudo_whisper_gen(normal, sample_rate)
    sf.write(out_path, pesudo_whisper, sample_rate)
    return out_path, len(pesudo_whisper) / sample_rate