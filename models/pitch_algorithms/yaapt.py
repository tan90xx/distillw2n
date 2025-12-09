from typing import Tuple

import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import numpy as np

from .base import ThresholdPitchAlgorithm


class YAAPTPitchAlgorithm(ThresholdPitchAlgorithm):
    """YAAPT pitch detection algorithm implementation."""

    def __init__(
        self,
        frame_length: float = 35.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Convert frame_length from milliseconds to samples
        self.frame_length_samples = int((frame_length / 1000.0) * self.sample_rate)

        # Calculate frame spacing in milliseconds for YAAPT
        self.frame_space_ms = (self.hop_size / self.sample_rate) * 1000.0

        # Configure YAAPT parameters
        self.yaapt_params = {
            "frame_length": frame_length,  # frame length in ms
            "frame_space": self.frame_space_ms,  # frame spacing in ms
            "f0_min": self.fmin,  # minimum pitch
            "f0_max": self.fmax,  # maximum pitch
            "nccf_thresh1": 0.25,  # lower NCCF threshold
            "nccf_thresh2": 0.9,  # upper NCCF threshold
            "nccf_maxcands": 4,  # maximum number of candidates
            "shc_maxpeaks": 4,  # maximum number of SHC peaks
            "shc_pwidth": 50,  # SHC window width
            "shc_thresh1": 5,  # SHC threshold 1
            "shc_thresh2": 1.25,  # SHC threshold 2
            "f0_double": 150,  # pitch doubling threshold
            "f0_half": 150,  # pitch halving threshold
            "merit_boost": 0.20,  # merit boost
            "merit_pivot": 0.99,  # merit pivot
            "merit_extra": 0.4,  # merit extra
            "median_value": 7,  # median filter order
            "dp_w1": 0.15,  # DP weight for voiced-voiced transitions
            "dp_w2": 0.5,  # DP weight for voiced-unvoiced transitions
            "dp_w3": 0.1,  # DP weight for unvoiced-unvoiced transitions
            "dp_w4": 0.9,  # DP weight for local costs
            "spec_pitch_min_std": 0.05,  # minimum spectral pitch std dev
        }

    def _extract_pitch_with_threshold(
        self, audio: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Create signal object for pYAAPT
        signal = basic.SignalObj(audio, self.sample_rate)

        # Update NLFER threshold based on input threshold
        self.yaapt_params["nlfer_thresh1"] = threshold

        # Extract pitch using YAAPT
        pitch = pYAAPT.yaapt(signal, **self.yaapt_params)

        # Get pitch values and voicing decisions
        pitch_values = pitch.samp_values

        # use YAAPT’s own frame positions for accurate time stamps
        # frames_pos is in samples, so divide by sample_rate to get seconds
        times = np.array(pitch.frames_pos) / self.sample_rate

        return (
            times,
            pitch_values,
            (pitch_values >= self.fmin).astype(np.float32),
        )

    def _get_default_threshold(self) -> float:
        return 0.675
