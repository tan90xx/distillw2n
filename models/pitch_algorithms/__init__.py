import numpy as np
import librosa
from importlib import import_module
from typing import Dict, List, Optional, Type

from .base import PitchAlgorithm

# Algorithm metadata - maps names to (module_name, class_name, required_packages)
_ALGORITHM_METADATA = {
    "CREPE": ("crepe", "CREPEPitchAlgorithm", ["crepe", "tensorflow"]),
    "PENN": ("penn", "PENNPitchAlgorithm", ["penn"]),
    "Praat": ("praat", "PraatPitchAlgorithm", ["praat-parselmouth"]),
    "RAPT": ("rapt", "RAPTPitchAlgorithm", ["pysptk"]),
    "SWIPE": ("swipe", "SWIPEPitchAlgorithm", ["pysptk"]),
    "TorchCREPE": (
        "torchcrepe",
        "TorchCREPEPitchAlgorithm",
        ["torchcrepe", "torch"],
    ),
    "YAAPT": ("yaapt", "YAAPTPitchAlgorithm", ["AMFM-decompy"]),
    "pYIN": ("pyin", "pYINPitchAlgorithm", ["librosa"]),
    "BasicPitch": ("basicpitch", "BasicPitchPitchAlgorithm", ["basic-pitch"]),
    "SwiftF0": ("swiftf0", "SwiftF0PitchAlgorithm", ["swift-f0"]),
    "SPICE": (
        "spice",
        "SPICEPitchAlgorithm",
        ["tensorflow", "tensorflow-hub"],
    ),
    "RMVPE": (
        "rmvpe",
        "RMVPEPitchAlgorithm",
        ["torch"],
    ),
}

# The _REGISTRY now acts as a cache for lazily loaded algorithms.
_REGISTRY: Dict[str, Type[PitchAlgorithm]] = {}
_IMPORT_ERRORS: Dict[str, str] = {}

def extract_pitch(
    audio: np.array,
    selected_algorithms: List[str],
    sr: int = 22050,
    hop_size: int = 256,
    fmin: float = 65,
    fmax: float = 300,
    pitch_threshold: Optional[
        float
    ] = None,  # Override threshold for all algorithms
):
    """Extract picth using different pitch detection algorithms.

    Args:
        audio: Audio array
        selected_algorithms: List of algorithm names
        sr: Sample rate
        hop_size: Hop size for analysis
        fmin: Minimum frequency
        fmax: Maximum frequency
        pitch_threshold: Override threshold for all algorithms (None = use each algorithm's default)
    """
    try:
        audio_max = np.max(np.abs(audio))
        if audio_max > 1.0:
            audio = audio / audio_max
        audio_duration = librosa.get_duration(y=audio, sr=sr)
    except Exception as e:
        raise RuntimeError(f"Error loading audio file: {e}")

    # Filter algorithms based on selection
    filtered_algorithms = [get_algorithm(algo) for algo in selected_algorithms]
    if not filtered_algorithms:
        raise ValueError("No valid algorithms selected.")

    results = []
    for algo_class in filtered_algorithms:
        algo_name = algo_class.get_name()
        try:
            algo_instance = algo_class(
                sample_rate=sr, hop_size=hop_size, fmin=fmin, fmax=fmax
            )
            # Get the threshold to use
            if pitch_threshold is not None:
                threshold = pitch_threshold  # Use override
            else:
                threshold = (
                    algo_instance._get_default_threshold()
                )  # Use algorithm's default

            if algo_instance.supports_continuous_periodicity:
                pitch, periodicity = (
                    algo_instance.extract_continuous_periodicity(audio)
                )
            else:
                pitch, periodicity, _ = algo_instance.extract_pitch(
                    audio, thresholds=threshold
                )

            results.append(
                (algo_name, pitch, periodicity, threshold)
            )
        except Exception as e:
            print(f"Error processing {algo_name}: {e}")
            continue

    if len(results) == 1:
        algo_name, pitch, periodicity, threshold = results[0]
        return pitch # , periodicity
    
    elif len(results) > 1:
        all_pitches = []
        # all_periodicities = []
        
        for algo_name, pitch, periodicity, threshold in results:
            all_pitches.append(pitch)
            # all_periodicities.append(periodicity)
        
        min_length = min(len(pitch) for pitch in all_pitches)
        all_pitches = [pitch[:min_length] for pitch in all_pitches]
        # all_periodicities = [periodicity[:min_length] for periodicity in all_periodicities]
        
        pitch_array = np.array(all_pitches)
        # periodicity_array = np.array(all_periodicities)
        
        median_pitch = np.median(pitch_array, axis=0)
        # median_periodicity = np.median(periodicity_array, axis=0)
        
        return median_pitch # , median_periodicity
    
    else:
        raise ValueError("No algorithms successfully processed the audio.")


def get_algorithm(
    name: str, fail_silently: bool = False
) -> Optional[Type[PitchAlgorithm]]:
    """
    Get algorithm class by name using lazy loading.
    An algorithm's module is only imported the first time it is requested.
    """
    # 1. Check if the algorithm is already cached
    if name in _REGISTRY:
        return _REGISTRY[name]

    # 2. Check if the algorithm name is valid
    if name not in _ALGORITHM_METADATA:
        if fail_silently:
            return None
        raise ValueError(f"Unknown algorithm: {name}")

    # 3. If not cached, try to import it now (the "lazy" part)
    module_name, class_name, deps = _ALGORITHM_METADATA[name]
    try:
        module = import_module(f".{module_name}", package=__package__)
        cls = getattr(module, class_name)
        _REGISTRY[name] = cls  # Cache the successfully imported class
        return cls
    except ImportError as e:
        _IMPORT_ERRORS[name] = str(e)
        if fail_silently:
            return None
        raise ImportError(
            f"Algorithm '{name}' requires packages: {deps}\n"
            f"Error: {_IMPORT_ERRORS[name]}\n"
            f"See README for installation instructions."
        )
    except Exception as e:
        _IMPORT_ERRORS[name] = str(e)
        if fail_silently:
            return None
        raise e


def list_algorithms() -> List[str]:
    """
    Return a list of all possible algorithm names without importing them.
    This is fast and suitable for populating command-line choices.
    """
    return list(_ALGORITHM_METADATA.keys())


def get_available_algorithms() -> List[str]:
    """
    Actively tries to import all algorithms and returns a list of those that are available.
    This function is "eager" and should be used for diagnostic purposes.
    """
    available = []
    for name in _ALGORITHM_METADATA:
        if get_algorithm(name, fail_silently=True):
            available.append(name)
    return available


def get_algorithm_dependencies(name: str) -> list:
    """Get required packages for a specific algorithm."""
    if name not in _ALGORITHM_METADATA:
        raise ValueError(f"Unknown algorithm: {name}")
    return _ALGORITHM_METADATA[name][2]


def register_algorithm(name: str, algorithm_class: Type[PitchAlgorithm]):
    """Register a custom algorithm."""
    if not issubclass(algorithm_class, PitchAlgorithm):
        raise TypeError("Algorithm must subclass PitchAlgorithm")
    _REGISTRY[name] = algorithm_class
