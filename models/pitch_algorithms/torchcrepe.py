from typing import Callable, Tuple

import numpy as np
import torch
import torchcrepe

from .base import ContinuousPitchAlgorithm


class TorchCREPEPitchAlgorithm(ContinuousPitchAlgorithm):
    def __init__(
        self,
        decoder: Callable = torchcrepe.decode.viterbi,
        model: str = "full",
        device: str = None,
        **kwargs,
    ):
        """Initialize TorchCREPE pitch detector.

        Args:
            decoder: Strategy for converting network output to pitch ('weighted_argmax', 'argmax' or 'viterbi')
            model: Model capacity ('tiny', or 'full')
            device: Computation device ("cpu" or "cuda")
        """
        super().__init__(**kwargs)
        self.decoder = decoder
        self.model = model

        if device is None:
            # Auto-select: prioritize CUDA if available, otherwise fall back to CPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda" and not torch.cuda.is_available():
            # User requested CUDA, but it's not available; fall back to CPU
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = device

    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Convert to tensor and add batch dimension
        audio_tensor = torch.from_numpy(audio).to(self.device).unsqueeze(0)

        # Run prediction
        pitch, periodicity = torchcrepe.predict(
            audio_tensor,
            self.sample_rate,
            self.hop_size,
            self.fmin,
            self.fmax,
            model=self.model,
            return_periodicity=True,
            decoder=self.decoder,
            device=self.device,
            batch_size=2048,
            pad=True,  # Ensure consistent frame count
        )

        # Convert to numpy arrays and remove batch dimension
        pitch = pitch.squeeze().cpu().numpy()
        periodicity = periodicity.squeeze().cpu().numpy()

        # Calculate time points (center-aligned frames)
        n_frames = len(pitch)
        frame_centers = np.arange(n_frames) * self.hop_size + (1024 // 2)
        times = frame_centers / self.sample_rate

        return times, pitch, periodicity

    def _get_default_threshold(self) -> float:
        return 0.613
