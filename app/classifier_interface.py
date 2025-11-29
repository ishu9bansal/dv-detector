from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np


class EmotionClassifier(ABC):
    """
    Abstract interface for emotion classification from audio.
    Implementations handle model loading, inference, and result formatting.
    """

    @abstractmethod
    def classify(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """
        Classify emotion from audio samples.
        
        Args:
            audio: Normalized audio samples as numpy array
            sample_rate: Audio sample rate in Hz
            
        Returns:
            List of emotion predictions, each with 'label' and 'score' keys,
            sorted by confidence descending
        """
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        pass
