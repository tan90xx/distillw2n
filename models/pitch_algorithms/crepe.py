import os
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from numpy.lib.stride_tricks import as_strided

from .base import ContinuousPitchAlgorithm

# This global flag ensures that the complex, one-time TensorFlow setup
# is performed only once during the entire program's execution.
TF_CONFIGURED = False


class CREPEPitchAlgorithm(ContinuousPitchAlgorithm):
    def __init__(
        self,
        viterbi: bool = True,
        model: str = "full",
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

        self.viterbi = viterbi
        self.model_capacity = model
        self.step_size = (self.hop_size / self.sample_rate) * 1000
        self.model_srate = 16000  # CREPE expects 16kHz

        # Set device
        gpus_available = len(tf.config.list_physical_devices("GPU")) > 0
        if device == "cuda" and gpus_available:
            self.tf_device = "/GPU:0"
        else:
            self.tf_device = "/CPU:0"

        # Load the model once during initialization
        with tf.device(self.tf_device):
            self.model = self._build_and_load_model()

        # Precompute cents mapping for efficiency - EXACTLY like original
        self.cents_mapping = np.linspace(0, 7180, 360) + 1997.3794084376191

    def _build_and_load_model(self):
        """Build and load the CREPE model"""
        import bz2

        from tensorflow.keras.layers import (
            BatchNormalization,
            Conv2D,
            Dense,
            Dropout,
            Flatten,
            Input,
            MaxPool2D,
            Permute,
            Reshape,
        )
        from tensorflow.keras.models import Model

        capacity_multiplier = {
            "tiny": 4,
            "small": 8,
            "medium": 16,
            "large": 24,
            "full": 32,
        }[self.model_capacity]

        layers = [1, 2, 3, 4, 5, 6]
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        widths = [512, 64, 64, 64, 64, 64]
        strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

        x = Input(shape=(1024,), name="input", dtype="float32")
        y = Reshape(target_shape=(1024, 1, 1), name="input-reshape")(x)

        for l, f, w, s in zip(layers, filters, widths, strides):
            y = Conv2D(
                f,
                (w, 1),
                strides=s,
                padding="same",
                activation="relu",
                name="conv%d" % l,
            )(y)
            y = BatchNormalization(name="conv%d-BN" % l)(y)
            y = MaxPool2D(
                pool_size=(2, 1),
                strides=None,
                padding="valid",
                name="conv%d-maxpool" % l,
            )(y)
            y = Dropout(0.25, name="conv%d-dropout" % l)(y)

        y = Permute((2, 1, 3), name="transpose")(y)
        y = Flatten(name="flatten")(y)
        y = Dense(360, activation="sigmoid", name="classifier")(y)

        model = Model(inputs=x, outputs=y)

        # Find and load weights - handle both compressed and uncompressed
        filename = f"model-{self.model_capacity}.h5"
        weights_path = None

        # Try to find weights in CREPE package location
        try:
            import crepe

            crepe_dir = os.path.dirname(crepe.__file__)
            weights_path = os.path.join(crepe_dir, filename)

            # If .h5 file doesn't exist, try to decompress .h5.bz2
            if not os.path.exists(weights_path):
                compressed_path = weights_path + ".bz2"
                if os.path.exists(compressed_path):
                    print(f"Decompressing {compressed_path}...")
                    with bz2.BZ2File(compressed_path, "rb") as source:
                        with open(weights_path, "wb") as target:
                            target.write(source.read())
                    print("Decompression complete")

        except ImportError:
            # Fallback: try current directory
            weights_path = os.path.join(os.path.dirname(__file__), filename)

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Could not find CREPE model weights: {filename}. "
                f"Make sure CREPE is properly installed with model weights."
            )

        model.load_weights(weights_path)
        # No need to compile for inference-only usage
        return model

    def _to_local_average_cents(
        self, salience: np.ndarray, center: Optional[int] = None
    ) -> np.ndarray:
        """Convert salience to cents using local weighted average - EXACT copy from original"""
        if salience.ndim == 1:
            if center is None:
                center = int(np.argmax(salience))
            start = max(0, center - 4)
            end = min(len(salience), center + 5)
            salience = salience[start:end]
            product_sum = np.sum(salience * self.cents_mapping[start:end])
            weight_sum = np.sum(salience)
            return product_sum / weight_sum
        if salience.ndim == 2:
            return np.array(
                [
                    self._to_local_average_cents(salience[i, :])
                    for i in range(salience.shape[0])
                ]
            )

        raise Exception("label should be either 1d or 2d ndarray")

    def _to_viterbi_cents(self, salience: np.ndarray) -> np.ndarray:
        """Apply Viterbi smoothing to pitch estimates - EXACT copy from original"""
        from hmmlearn import hmm

        # uniform prior on the starting pitch
        starting = np.ones(360) / 360

        # transition probabilities inducing continuous pitch
        xx, yy = np.meshgrid(range(360), range(360))
        transition = np.maximum(12 - abs(xx - yy), 0)
        transition = transition / np.sum(transition, axis=1)[:, None]

        # emission probability = fixed probability for self, evenly distribute the
        # others
        self_emission = 0.1
        emission = np.eye(360) * self_emission + np.ones(shape=(360, 360)) * (
            (1 - self_emission) / 360
        )

        # fix the model parameters because we are not optimizing the model
        model = hmm.CategoricalHMM(360, starting, transition)
        model.startprob_, model.transmat_, model.emissionprob_ = (
            starting,
            transition,
            emission,
        )

        # find the Viterbi path
        observations = np.argmax(salience, axis=1)
        path = model.predict(observations.reshape(-1, 1), [len(observations)])

        return np.array(
            [
                self._to_local_average_cents(salience[i, :], path[i])
                for i in range(len(observations))
            ]
        )

    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract pitch and periodicity from audio using the loaded CREPE model"""

        # Convert to mono if stereo
        if len(audio.shape) == 2:
            audio = audio.mean(1)  # make mono
        audio = audio.astype(np.float32)

        # Resample if necessary
        if self.sample_rate != self.model_srate:
            from resampy import resample

            audio = resample(audio, self.sample_rate, self.model_srate)

        # pad so that frames are centered around their timestamps (i.e. first frame
        # is zero centered). - EXACTLY like original with center=True
        audio = np.pad(audio, 512, mode="constant", constant_values=0)

        # make 1024-sample frames of the audio with hop length of 10 milliseconds
        hop_length = int(self.model_srate * self.step_size / 1000)
        n_frames = 1 + int((len(audio) - 1024) / hop_length)
        frames = as_strided(
            audio,
            shape=(1024, n_frames),
            strides=(audio.itemsize, hop_length * audio.itemsize),
        )
        frames = frames.transpose().copy()

        # normalize each frame -- this is expected by the model
        frames -= np.mean(frames, axis=1)[:, np.newaxis]
        frames /= np.clip(np.std(frames, axis=1)[:, np.newaxis], 1e-8, None)

        # run prediction and convert the frequency bin weights to Hz
        with tf.device(self.tf_device):
            activation = self.model.predict(frames, verbose=0)

        # Extract confidence
        confidence = activation.max(axis=1)

        # Convert to pitch
        if self.viterbi:
            cents = self._to_viterbi_cents(activation)
        else:
            cents = self._to_local_average_cents(activation)

        # Convert cents to frequency
        frequency = 10 * 2 ** (cents / 1200)
        frequency[np.isnan(frequency)] = 0

        # Create time array EXACTLY like original
        time = np.arange(confidence.shape[0]) * self.step_size / 1000.0

        return time, frequency, confidence

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
        return 0.775
