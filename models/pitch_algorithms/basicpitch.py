from typing import Tuple

import numpy as np
from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
from basic_pitch.inference import Model, predict

from .base import ContinuousPitchAlgorithm


class BasicPitchPitchAlgorithm(ContinuousPitchAlgorithm):
    def __init__(
        self,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        minimum_note_length: float = 0.058,
        melodia_trick: bool = True,
        **kwargs,
    ):
        """Initialize Basic Pitch algorithm.

        Basic Pitch is a polyphonic automatic music transcription model that can detect
        multiple simultaneous pitches. This wrapper extracts the most prominent pitch
        per frame to fit the monophonic PitchAlgorithm interface.

        Args:
            onset_threshold: Threshold for onset detection (0.0-1.0, default: 0.5)
            frame_threshold: Threshold for frame-level note detection (0.0-1.0, default: 0.3)
            minimum_note_length: Minimum note length in seconds (default: 0.058)
            melodia_trick: Use melodia trick for better monophonic performance (default: True)
        """
        super().__init__(**kwargs)

        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.minimum_note_length = minimum_note_length
        self.melodia_trick = melodia_trick

        # Load the Basic Pitch model once during initialization
        onnx_model_path = build_icassp_2022_model_path(FilenameSuffix.onnx)
        self.model = Model(onnx_model_path)

    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        import os
        import tempfile
        from contextlib import redirect_stdout
        from io import StringIO

        import soundfile as sf

        # Basic Pitch expects a file path, so we need to save the audio temporarily
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Write audio array to temporary file
            sf.write(temp_path, audio, self.sample_rate)

            # Run Basic Pitch prediction with file path (suppress print output)
            with redirect_stdout(StringIO()):
                model_output, _, _ = predict(
                    temp_path,
                    self.model,
                    onset_threshold=self.onset_threshold,
                    frame_threshold=self.frame_threshold,
                    minimum_note_length=self.minimum_note_length,
                    minimum_frequency=self.fmin,
                    maximum_frequency=self.fmax,
                    melodia_trick=self.melodia_trick,
                )
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        # Extract note activation matrix: shape (time, pitch)
        note_activations = model_output["note"]
        n_frames = note_activations.shape[0]

        # Basic Pitch uses 88 piano keys (A0 to C8), starting from MIDI note 21
        num_pitches = note_activations.shape[1]
        midi_start = 21  # A0
        midi_numbers = np.arange(midi_start, midi_start + num_pitches)
        frequencies = 440.0 * (2.0 ** ((midi_numbers - 69) / 12.0))

        # Create frequency mask for valid range (vectorized)
        valid_freq_mask = (frequencies >= self.fmin) & (frequencies <= self.fmax)

        # Apply frequency constraints to all frames at once
        masked_activations = note_activations * valid_freq_mask[np.newaxis, :]

        # Find most prominent pitch per frame (vectorized)
        max_indices = np.argmax(masked_activations, axis=1)
        max_confidences = masked_activations[np.arange(len(max_indices)), max_indices]

        # Convert MIDI indices to frequencies
        pitch_estimates = frequencies[max_indices]

        # Calculate timestamps (Basic Pitch outputs at 100Hz)
        frame_rate = 100.0  # Fixed frame rate per Basic Pitch constants
        times = np.arange(n_frames) / frame_rate

        return times, pitch_estimates, max_confidences

    def _get_default_threshold(self) -> float:
        return 0.45
