from typing import Tuple

import numpy as np
from swift_f0 import SwiftF0

from .base import ContinuousPitchAlgorithm


class SwiftF0PitchAlgorithm(ContinuousPitchAlgorithm):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.detector = SwiftF0()

    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        result = self.detector.detect_from_array(audio, self.sample_rate)

        return result.timestamps, result.pitch_hz, result.confidence

    def _get_default_threshold(self) -> float:
        return 0.887
