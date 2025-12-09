from typing import Tuple

import numpy as np
import penn
import torch

from .base import ContinuousPitchAlgorithm


class PENNPitchAlgorithm(ContinuousPitchAlgorithm):
    def __init__(
        self,
        device: str = None,
        batch_size: int = 2048,
        center: str = "half-hop",
        **kwargs,
    ):
        """Initialize PENN pitch detector.
        Args:
            device: Computation device ("cpu", "cuda", or specific GPU index)
            batch_size: Number of frames to process per batch
            center: Frame centering strategy ('half-window', 'half-hop', or 'zero')
        """
        super().__init__(**kwargs)

        # Convert hop_size from samples to seconds as required by PENN
        self.hopsize_seconds = float(self.hop_size) / self.sample_rate

        # Handle device selection
        if device is None:
            self.gpu = 0 if torch.cuda.is_available() else None
        elif device == "cpu":
            self.gpu = None
        elif device == "cuda":
            self.gpu = 0 if torch.cuda.is_available() else None
        elif isinstance(device, int):
            if device >= torch.cuda.device_count():
                raise ValueError(f"GPU index {device} is out of range")
            self.gpu = device
        else:
            raise ValueError(f"Unsupported device specification: {device}")

        self.batch_size = batch_size
        self.center = center

    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Ensure audio is float32 and in correct shape
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Convert to torch tensor and add batch dimension if needed
        if len(audio.shape) == 1:
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        else:
            audio_tensor = torch.from_numpy(audio)

        pitch, periodicity = penn.from_audio(
            audio=audio_tensor,
            sample_rate=self.sample_rate,
            hopsize=self.hopsize_seconds,
            fmin=self.fmin,
            fmax=self.fmax,
            batch_size=self.batch_size,
            center=self.center,
            gpu=self.gpu,
        )

        # Calculate time points (center-aligned frames)
        n_frames = pitch.shape[1]

        # PENN uses fixed window size of 2048 samples at 8kHz
        # Calculate time offset based on centering strategy
        if self.center == "half-window":
            # Center is at half window (1024 samples at 8kHz)
            time_offset = 1024 / penn.SAMPLE_RATE
        elif self.center == "half-hop":
            # Center is at half hop (40 samples at 8kHz)
            time_offset = 40 / penn.SAMPLE_RATE
        else:  # "zero"
            # Center is at window start
            time_offset = 0

        # Calculate time points
        times = (np.arange(n_frames) * self.hopsize_seconds) + time_offset

        # Convert to numpy and remove batch dimension
        return (
            times,
            pitch.squeeze().cpu().numpy(),
            periodicity.squeeze().cpu().numpy(),
        )

    def _get_default_threshold(self) -> float:
        return 0.388
