import os
from typing import Dict, List, Any
import numpy as np
from transformers import pipeline
from .classifier_interface import EmotionClassifier

# Ensure transformers uses PyTorch and lowers verbosity
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


class Wav2VecEmotionClassifier(EmotionClassifier):
    """
    Wav2Vec-based emotion classifier using HuggingFace transformers.
    Uses r-f/wav2vec-english-speech-emotion-recognition model.
    """

    def __init__(self, model_name: str = "r-f/wav2vec-english-speech-emotion-recognition", device: int = -1):
        """
        Initialize the Wav2Vec classifier.
        
        Args:
            model_name: HuggingFace model identifier
            device: -1 for CPU, >= 0 for GPU index
        """
        self.model_name = model_name
        self.device = device
        self._pipeline = None

    def _ensure_loaded(self):
        """Lazy load the model on first use."""
        if self._pipeline is None:
            print(f"Loading model: {self.model_name}...")
            self._pipeline = pipeline(
                "audio-classification",
                model=self.model_name,
                device=self.device
            )
            print("✅ Model loaded!")

    def classify(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """
        Classify emotion from audio using Wav2Vec model.
        
        Args:
            audio: Normalized audio samples
            sample_rate: Sample rate in Hz
            
        Returns:
            List of predictions with 'label' and 'score', sorted by confidence
        """
        self._ensure_loaded()
        if self._pipeline is None:
            raise RuntimeError("Model failed to load.")
        result = self._pipeline(audio, sampling_rate=sample_rate)
        # Ensure sorted by score descending
        # return sorted(result, key=lambda x: x['score'], reverse=True)
        return result

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._pipeline is not None
