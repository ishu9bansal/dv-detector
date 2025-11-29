from app.audio_processor import AudioProcessor
import numpy as np

if __name__ == "__main__":
    processor = AudioProcessor(sample_rate=16000, window_seconds=1)
    print("Running classifier self-test...")
    test = processor.self_test()
    print("Self-test:", test)

    # Also test processing pipeline readiness
    # Initially buffer insufficient
    res = processor.process()
    print("Process (empty buffer):", res)

    # Add synthetic audio to meet window
    audio = (np.random.randn(processor.sample_rate).astype(np.float32) * 0.05)
    processor.add_chunk(audio)
    res2 = processor.process(input_timestamp=0)
    print("Process (with audio):", res2)
