from typing import Tuple

import numpy as np
from pysptk import sptk

from .base import ThresholdPitchAlgorithm


class SWIPEPitchAlgorithm(ThresholdPitchAlgorithm):
    def _extract_pitch_with_threshold(
        self, audio: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        # SWIPE expects a special range.
        # Map threshold from [0,1] to [0.2,0.5]
        norm_threshold = np.clip(0.2 + threshold * (0.5 - 0.2), 0.2, 0.5)

        f0 = sptk.swipe(
            audio,
            self.sample_rate,
            self.hop_size,
            min=self.fmin,
            max=self.fmax,
            threshold=norm_threshold,
            otype="f0",
        )

        # Time axis: approximate center of variable window using half-hop
        n_frames = len(f0)
        half_hop = self.hop_size / 2
        times = (np.arange(n_frames) * self.hop_size + half_hop) / self.sample_rate

        return times, f0, (f0 >= self.fmin).astype(np.float32)

    def _get_default_threshold(self) -> float:
        return 0.4
