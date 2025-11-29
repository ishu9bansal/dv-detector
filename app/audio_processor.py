import numpy as np
from typing import Dict, Any, List, Optional
from .model import get_classifier

class AudioProcessor:
    def __init__(self, sample_rate: int = 16000, window_seconds: int = 3):
        self.sample_rate = sample_rate
        self.window_samples = sample_rate * window_seconds
        self.buffer: List[float] = []
        self.classifier = None

    def _ensure_model(self):
        if self.classifier is None:
            self.classifier = get_classifier()

    def add_chunk(self, chunk: np.ndarray) -> None:
        self.buffer.extend(chunk.tolist())

    def process(self, input_timestamp: Optional[int] = None) -> Dict[str, Any]:
        if len(self.buffer) < self.window_samples:
            return {"ready": False}

        audio = np.array(self.buffer[-self.window_samples:])
        audio_level = float(np.abs(audio).mean())

        if audio_level <= 0.01:
            return {
                "ready": True,
                "silence": True,
                "level": int(min(audio_level * 1000, 100))
            }

        # Normalize
        peak = float(np.abs(audio).max())
        audio = audio / max(peak, 1e-5)

        # Inference
        self._ensure_model()
        result = self.classifier(audio, sampling_rate=self.sample_rate)

        return {
            "ready": True,
            "silence": False,
            "emotion": result[0]['label'],
            "confidence": float(result[0]['score']),
            "allEmotions": result,
            "level": int(min(audio_level * 1000, 100)),
            "inputTimestamp": input_timestamp,
        }
