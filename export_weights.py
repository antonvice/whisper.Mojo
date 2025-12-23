
import torch
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import os
import requests
import io
import soundfile as sf
from scipy.signal import resample

def export_whisper_tiny():
    print("Loading whisper-tiny model...")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    
    state_dict = model.state_dict()
    output_path = "whisper_tiny_weights.bin"
    
    with open(output_path, "wb") as f:
        # Encoder stem
        f.write(state_dict["model.encoder.conv1.weight"].numpy().astype(np.float32).tobytes())
        f.write(state_dict["model.encoder.conv1.bias"].numpy().astype(np.float32).tobytes())
        f.write(state_dict["model.encoder.conv2.weight"].numpy().astype(np.float32).tobytes())
        f.write(state_dict["model.encoder.conv2.bias"].numpy().astype(np.float32).tobytes())
        f.write(state_dict["model.encoder.embed_positions.weight"].numpy().astype(np.float32).tobytes())
        
        # Encoder blocks
        for i in range(4):
            f.write(state_dict[f"model.encoder.layers.{i}.self_attn.q_proj.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.encoder.layers.{i}.self_attn.q_proj.bias"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.encoder.layers.{i}.self_attn.k_proj.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.encoder.layers.{i}.self_attn.v_proj.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.encoder.layers.{i}.self_attn.v_proj.bias"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.encoder.layers.{i}.self_attn.out_proj.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.encoder.layers.{i}.self_attn.out_proj.bias"].numpy().astype(np.float32).tobytes())
            
            f.write(state_dict[f"model.encoder.layers.{i}.self_attn_layer_norm.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.encoder.layers.{i}.self_attn_layer_norm.bias"].numpy().astype(np.float32).tobytes())
            
            f.write(state_dict[f"model.encoder.layers.{i}.fc1.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.encoder.layers.{i}.fc1.bias"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.encoder.layers.{i}.fc2.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.encoder.layers.{i}.fc2.bias"].numpy().astype(np.float32).tobytes())
            
            f.write(state_dict[f"model.encoder.layers.{i}.final_layer_norm.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.encoder.layers.{i}.final_layer_norm.bias"].numpy().astype(np.float32).tobytes())
            
        # Encoder final norm
        f.write(state_dict["model.encoder.layer_norm.weight"].numpy().astype(np.float32).tobytes())
        f.write(state_dict["model.encoder.layer_norm.bias"].numpy().astype(np.float32).tobytes())
        
        # Decoder stem
        f.write(state_dict["model.decoder.embed_tokens.weight"].numpy().astype(np.float32).tobytes())
        f.write(state_dict["model.decoder.embed_positions.weight"].numpy().astype(np.float32).tobytes())
        
        # Decoder blocks
        for i in range(4):
            f.write(state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.decoder.layers.{i}.self_attn.out_proj.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.decoder.layers.{i}.self_attn.out_proj.bias"].numpy().astype(np.float32).tobytes())
            
            f.write(state_dict[f"model.decoder.layers.{i}.self_attn_layer_norm.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.decoder.layers.{i}.self_attn_layer_norm.bias"].numpy().astype(np.float32).tobytes())
            
            f.write(state_dict[f"model.decoder.layers.{i}.encoder_attn.q_proj.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.decoder.layers.{i}.encoder_attn.q_proj.bias"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.decoder.layers.{i}.encoder_attn.k_proj.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.decoder.layers.{i}.encoder_attn.v_proj.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.decoder.layers.{i}.encoder_attn.v_proj.bias"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.decoder.layers.{i}.encoder_attn.out_proj.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.decoder.layers.{i}.encoder_attn.out_proj.bias"].numpy().astype(np.float32).tobytes())
            
            f.write(state_dict[f"model.decoder.layers.{i}.encoder_attn_layer_norm.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.decoder.layers.{i}.encoder_attn_layer_norm.bias"].numpy().astype(np.float32).tobytes())
            
            f.write(state_dict[f"model.decoder.layers.{i}.fc1.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.decoder.layers.{i}.fc1.bias"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.decoder.layers.{i}.fc2.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.decoder.layers.{i}.fc2.bias"].numpy().astype(np.float32).tobytes())
            
            f.write(state_dict[f"model.decoder.layers.{i}.final_layer_norm.weight"].numpy().astype(np.float32).tobytes())
            f.write(state_dict[f"model.decoder.layers.{i}.final_layer_norm.bias"].numpy().astype(np.float32).tobytes())
            
        # Decoder final norm
        f.write(state_dict["model.decoder.layer_norm.weight"].numpy().astype(np.float32).tobytes())
        f.write(state_dict["model.decoder.layer_norm.bias"].numpy().astype(np.float32).tobytes())
        
    print(f"Weights exported to {output_path}")

    # Download sample audio from internet
    audio_url = "https://www.kozco.com/tech/LRMonoPhase4.wav" # A short test file
    print(f"Downloading sample audio from {audio_url}...")
    try:
        response = requests.get(audio_url)
        audio_data, samplerate = sf.read(io.BytesIO(response.content))
        
        # Whisper expects 16kHz mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Resample to 16000Hz using scipy
        if samplerate != 16000:
            print(f"Resampling from {samplerate}Hz to 16000Hz...")
            num_samples = int(len(audio_data) * 16000 / samplerate)
            audio_data = resample(audio_data, num_samples).astype(np.float32)
            samplerate = 16000
        
        # Limit to 30 seconds
        max_samples = samplerate * 30
        audio_data = audio_data[:max_samples]
        
        input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features
        print(f"Input features shape: {input_features.shape}")
        
        with open("sample_input.bin", "wb") as f:
            f.write(input_features.numpy().astype(np.float32).tobytes())
        print("Sample input exported to sample_input.bin")

        # Get expected tokens (greedy)
        print("Finding expected tokens (PyTorch)...")
        generated_ids = model.generate(input_features)
        print(f"Generated token IDs: {generated_ids[0][:10].tolist()}")
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=False)
        print(f"Expected transcription: {decoded[0]}")
        
        with open("expected_tokens.txt", "w") as f:
            f.write(str(list(generated_ids[0].numpy())))
            
        # Export vocabulary
        print("Exporting vocabulary...")
        vocab = processor.tokenizer.get_vocab()
        # Sort by ID
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        with open("vocab.txt", "w", encoding="utf-8") as f:
            for token, _ in sorted_vocab:
                # Replace newlines and handle special characters to keep it simple for Mojo
                safe_token = token.replace("\n", "\\n")
                f.write(f"{safe_token}\n")
        print("Vocabulary exported to vocab.txt")
            
    except Exception as e:
        print(f"Error downloading or processing audio: {e}")
        # Fallback to random noise
        audio_array = np.random.randn(16000 * 5).astype(np.float32)
        input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features
        with open("sample_input.bin", "wb") as f:
            f.write(input_features.numpy().astype(np.float32).tobytes())
        print("Fallback sample input exported to sample_input.bin")

if __name__ == "__main__":
    export_whisper_tiny()
