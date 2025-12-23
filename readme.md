# 🎙️ Whisper.Mojo

A high-performance implementation of OpenAI's **Whisper** model (Tiny version) written entirely in **Mojo** 🔥.

## 🚀 Overview

This project brings the power of OpenAI's Whisper to the Mojo programming language. By implementing the architecture from the ground up, we leverage Mojo's unique ability to combine Python-like syntax with C-level performance through hardware acceleration, SIMD, and low-level memory control.

> [!NOTE] 
> This implementation currently supports **Whisper-Tiny** with greedy decoding for English transcription.

## ✨ Features

- **🎯 Pure Mojo Implementation**: Every layer (Encoder, Decoder, Multi-Head Attention) is written in Mojo.
- **⚡ SIMD Acceleration**: Core tensor operations (Matmul, LayerNorm, GeLU) are vectorized using Mojo's SIMD primitives.
- **🎧 Real-world Audio**: Integrated pipeline to process real audio files (MP3/WAV) into Mel spectrograms.
- **🔍 Bit-Perfect Tokenization**: Fully compatible with OpenAI's tokenizer, producing identical results to the PyTorch reference implementation.
- **💪 Memory Efficient**: Manual memory management using `LegacyUnsafePointer` for maximum control.

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `main.mojo` | 🎮 The entry point. Orchestrates weight loading, audio processing, and transcription. |
| `whisper.mojo` | 🧠 The "Brain". Contains the `Whisper` model, `Encoder`, and `Decoder` logic. |
| `layers.mojo` | 🧱 Core building blocks: `MultiHeadAttention` and `ResidualAttentionBlock`. |
| `whisper_tensor.mojo` | 🧬 Mathematical foundation. Implements `Tensor` and operations like `matmul`, `conv1d`, and `softmax`. |
| `tokenizer.mojo` | 🔤 Decodes the model's numeric output (token IDs) back into human-readable text. |
| `loader.mojo` | 📥 Efficiently loads model weights from a binary format into Mojo Tensors. |
| `export_weights.py` | 🐍 Python bridge. Handles model downloading, weight exporting, and audio preprocessing. |
| `vocab.txt` | 📚 The vocabulary file used for decoding tokens. |

## 🛠️ Getting Started

### 📋 Prerequisites

- **Mojo SDK** (v24.5+)
- **Python Environment** (for weight export) with:
  - `torch`, `transformers`, `soundfile`, `scipy`, `requests`

### 🏗️ Installation & Execution

1. **Clone & Setup**
   ```bash
   git clone https://github.com/antonvice/whisper.Mojo.git
   cd whisper.Mojo
   ```

2. **Export Weights & Prepare Audio**
   This script downloads the Whisper-Tiny weights and converts a sample audio file into a format Mojo can read.
   ```bash
   uv run export_weights.py
   ```

3. **Run Transcription**
   Launch the Mojo model to transcribe the prepared audio:
   ```bash
   mojo run main.mojo
   ```

## 📊 Performance Note

You might notice that `mojo run main.mojo` takes a few moments to execute. This is primarily because:
1. **JIT Compilation**: `mojo run` compiles the code on-the-fly. For production speed, use `mojo build main.mojo` to create a standalone binary.
2. **Current Optimization**: This is a reference implementation. While it uses SIMD for matmuls, many other loops (like the Attention scores) are currently single-threaded. Future versions will implement `parallelize` and tiling for even greater speeds.

## 📝 Example Output

```text
Initializing Whisper Tiny in Mojo...
Loading weights from whisper_tiny_weights.bin...
Transcription:
--------------------
 This is my voice on the left. This is my voice on the left hand side...
--------------------
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
