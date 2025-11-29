from app.wav2vec_classifier import Wav2VecEmotionClassifier
from app.audio_processor import AudioProcessor
import numpy as np

if __name__ == "__main__":
    # Initialize classifier
    classifier = Wav2VecEmotionClassifier()
    processor = AudioProcessor(classifier=classifier, sample_rate=16000, window_seconds=1)
    
    print("Running classifier self-test...")
    test = processor.self_test()
    print("Self-test:", test)

    # Also test processing pipeline readiness
    # Initially buffer insufficient
    res = processor.process()
    print("Process (empty buffer):", res)

    # Add synthetic audio to meet window
    audio = (np.random.randn(processor.sample_rate).astype(np.float32) * 0.05)
    processor.add_chunk(audio, input_timestamp=0)
    res2 = processor.process()
    print("Process (with audio):", res2)
