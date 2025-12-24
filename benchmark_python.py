
import torch
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import time
import os

def benchmark_python():
    print("Loading whisper-tiny model...")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    
    # Load sample input
    if not os.path.exists("sample_input.bin"):
        print("sample_input.bin not found. Run export_weights.py first.")
        return
        
    with open("sample_input.bin", "rb") as f:
        input_features = np.frombuffer(f.read(), dtype=np.float32).reshape(1, 80, 3000)
    
    input_features = torch.from_numpy(input_features)
    
    print("Running benchmark...")
    # Warmup
    _ = model.generate(input_features)
    
    # Measure
    start = time.time()
    generated_ids = model.generate(input_features)
    end = time.time()
    
    print(f"Python Transcription time: {end - start:.4f} seconds")
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(f"Transcription: {decoded[0]}")

if __name__ == "__main__":
    benchmark_python()
