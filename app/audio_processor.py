import numpy as np
from typing import Dict, Any, List, Optional, Deque
from collections import deque
from .classifier_interface import EmotionClassifier

class AudioProcessor:
    def __init__(self, classifier: EmotionClassifier, sample_rate: int = 16000, window_seconds: int = 3):
        self.classifier = classifier
        self.sample_rate = sample_rate
        self.window_samples = sample_rate * window_seconds
        # Fixed-size ring buffer to avoid unbounded growth
        self.buffer: Deque[float] = deque(maxlen=self.window_samples)
        self.latest_input_timestamp: Optional[int] = None

    def add_chunk(self, chunk: np.ndarray, input_timestamp: Optional[int] = None) -> None:
        # Keep only the last window worth of samples
        self.buffer.extend(chunk.tolist())
        if input_timestamp is not None:
            self.latest_input_timestamp = input_timestamp

    def process(self) -> Dict[str, Any]:
        # if len(self.buffer) < self.window_samples:
        #     return {"ready": False, "notReadyReason": "insufficient_buffer"}

        audio = np.array(list(self.buffer), dtype=np.float32)
        audio_level = float(np.abs(audio).mean())

        if audio_level <= 0.01:
            return {
                "ready": False,
                "notReadyReason": "silence",
                "silence": True,
                "level": int(min(audio_level * 1000, 100))
            }

        # Normalize
        peak = float(np.abs(audio).max())
        audio = audio / max(peak, 1e-5)

        # Inference
        try:
            result = self.classifier.classify(audio, sample_rate=self.sample_rate)
        except Exception as e:
            return {"ready": False, "notReadyReason": f"inference_error: {e}"}

        return {
            "ready": True,
            "silence": False,
            "emotion": result[0]['label'],
            "confidence": float(result[0]['score']),
            "allEmotions": result,
            "level": int(min(audio_level * 1000, 100)),
            "inputTimestamp": self.latest_input_timestamp,
        }

    def is_model_loaded(self) -> bool:
        return self.classifier.is_loaded()

    def self_test(self) -> Dict[str, Any]:
        """Run a lightweight self-test to verify classifier loads and runs."""
        try:
            test_audio = (np.random.randn(self.sample_rate).astype(np.float32) * 0.05)
            result = self.classifier.classify(test_audio, sample_rate=self.sample_rate)
            ok = isinstance(result, list) and len(result) > 0 and 'label' in result[0] and 'score' in result[0]
            return {
                "ok": bool(ok),
                "details": "classifier returned labels" if ok else "unexpected classifier output",
                "top": result[0] if ok else None,
            }
        except Exception as e:
            return {"ok": False, "details": f"error: {e}"}
