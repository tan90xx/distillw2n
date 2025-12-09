from typing import Tuple

import numpy as np
from pysptk import sptk

from .base import ThresholdPitchAlgorithm


class RAPTPitchAlgorithm(ThresholdPitchAlgorithm):
    def _extract_pitch_with_threshold(
        self, audio: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        audio_scaled = np.clip(audio * 32767, -32768, 32767)
        # RAPT expects a special range.
        # Map threshold from [0,1] to [-0.6,0.7]
        norm_threshold = -0.6 + threshold * (0.7 - (-0.6))

        f0 = sptk.rapt(
            audio_scaled,
            self.sample_rate,
            self.hop_size,
            min=self.fmin,
            max=self.fmax,
            voice_bias=norm_threshold,
            otype="f0",
        )

        # Build time‐axis (center of RAPT’s ~3‑period window)
        n_frames = len(f0)
        # RAPT’s window ≈ 3 periods of the lowest F0:
        window_center = int((self.sample_rate / self.fmin) * 1.5)
        times = (np.arange(n_frames) * self.hop_size + window_center) / self.sample_rate

        return times, f0, (f0 >= self.fmin).astype(np.float32)

    def _get_default_threshold(self) -> float:
        return 0.525
