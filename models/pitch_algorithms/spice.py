import os
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from .base import ContinuousPitchAlgorithm

# This global flag ensures that the complex, one-time TensorFlow setup
# is performed only once during the entire program's execution.
TF_CONFIGURED = False


class SPICEPitchAlgorithm(ContinuousPitchAlgorithm):
    """
    SPICE (Self-supervised Pitch Estimation) implementation using TensorFlow Hub.

    SPICE is a self-supervised pitch estimation model that provides robust
    pitch detection for monophonic audio signals.
    """

    _name = "SPICE"

    def __init__(
        self,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        global TF_CONFIGURED

        # Configure TensorFlow once
        if not TF_CONFIGURED:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
            tf.get_logger().setLevel("ERROR")
            try:
                gpus = tf.config.list_physical_devices("GPU")
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
            TF_CONFIGURED = True

        self.model_srate = 16000  # SPICE expects 16kHz

        # Set device
        gpus_available = len(tf.config.list_physical_devices("GPU")) > 0
        if device == "cuda" and gpus_available:
            self.tf_device = "/GPU:0"
        else:
            self.tf_device = "/CPU:0"

        # Load the SPICE model from TensorFlow Hub
        with tf.device(self.tf_device):
            self.model = hub.load("https://tfhub.dev/google/spice/2")

        # SPICE-specific constants for pitch conversion
        # These constants are taken from https://tfhub.dev/google/spice/2
        self.PT_OFFSET = 25.58
        self.PT_SLOPE = 63.07
        self.FMIN = 10.0
        self.BINS_PER_OCTAVE = 12.0

    def _spice_output_to_hz(self, pitch_output: np.ndarray) -> np.ndarray:
        """
        Convert SPICE pitch output (0-1 range) to frequency in Hz.

        Args:
            pitch_output: SPICE pitch predictions in range [0, 1]

        Returns:
            Frequency values in Hz
        """
        # Convert using SPICE's specific constants
        cqt_bin = pitch_output * self.PT_SLOPE + self.PT_OFFSET
        frequency = self.FMIN * (2.0 ** (cqt_bin / self.BINS_PER_OCTAVE))

        # Handle any invalid values
        frequency = np.nan_to_num(frequency, nan=0.0, posinf=0.0, neginf=0.0)

        return frequency

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for SPICE model.

        Args:
            audio: Input audio array

        Returns:
            Preprocessed audio ready for SPICE
        """
        # Convert to mono if stereo
        if len(audio.shape) == 2:
            audio = audio.mean(axis=1)

        # Convert to float32
        audio = audio.astype(np.float32)

        # Resample to 16kHz if necessary
        if self.sample_rate != self.model_srate:
            try:
                from resampy import resample

                audio = resample(audio, self.sample_rate, self.model_srate)
            except ImportError:
                # Fallback to scipy if resampy is not available
                from scipy.signal import resample

                target_length = int(len(audio) * self.model_srate / self.sample_rate)
                audio = resample(audio, target_length).astype(np.float32)

        # Ensure audio is normalized to [-1, 1] range
        audio_max = np.max(np.abs(audio))
        if audio_max > 1.0:
            audio = audio / audio_max

        return audio

    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pitch and periodicity from audio using SPICE model.

        Args:
            audio: Input audio array

        Returns:
            Tuple of (times, frequencies, confidences)
        """
        # Preprocess audio
        processed_audio = self._preprocess_audio(audio)

        # Run SPICE model prediction
        with tf.device(self.tf_device):
            # SPICE expects a constant tensor
            audio_tensor = tf.constant(processed_audio, dtype=tf.float32)

            # Get model output
            model_output = self.model.signatures["serving_default"](audio_tensor)

            # Extract pitch and uncertainty
            pitch_outputs = model_output["pitch"].numpy()
            uncertainty_outputs = model_output["uncertainty"].numpy()

        # Convert uncertainty to confidence
        confidence_outputs = 1.0 - uncertainty_outputs

        # Convert SPICE pitch outputs to Hz
        frequency_outputs = self._spice_output_to_hz(pitch_outputs)

        # Create time array
        # SPICE processes the entire audio at once, so we need to create
        # a time array that matches the output length
        n_frames = len(pitch_outputs)
        if n_frames > 0:
            # Estimate the hop size based on the input length and output length
            total_duration = len(processed_audio) / self.model_srate
            frame_hop_time = (
                total_duration / max(1, n_frames - 1)
                if n_frames > 1
                else total_duration
            )
            time_outputs = np.arange(n_frames) * frame_hop_time
        else:
            time_outputs = np.array([])

        return time_outputs, frequency_outputs, confidence_outputs

    def cleanup(self):
        """Clean up the loaded model"""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
            tf.keras.backend.clear_session()
            import gc

            gc.collect()

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup()

    def _get_default_threshold(self) -> float:
        return 0.825
