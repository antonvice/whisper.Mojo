# 🎙️ Whisper.Mojo

A high-performance implementation of OpenAI's **Whisper** model (Tiny version) written entirely in **Mojo** 🔥.

## 🚀 Overview

This project brings the power of OpenAI's Whisper to the Mojo programming language. By implementing the architecture from the ground up, we leverage Mojo's unique ability to combine Python-like syntax with C-level performance through hardware acceleration, SIMD, and low-level memory control.

> [!NOTE] 
> This implementation currently supports **Whisper-Tiny** with greedy decoding for English transcription.

## ✨ Features

- **🎯 Pure Mojo Implementation**: Every layer (Encoder, Decoder, Multi-Head Attention) is written in Mojo.
- **🚀 Ultra-Fast Inference**: Uses **KV-Caching** for incremental decoding, reducing complexity from $O(L^2)$ to $O(L)$.
- **🧵 Multi-core Parallelization**: Parallelized attention heads and tensor operations using Mojo's `parallelize` algorithm.
- **⚡ SIMD Acceleration**: Core math operations (Matmul, LayerNorm, GeLU) are vectorized using Mojo's SIMD primitives.
- **🎧 Real-world Audio**: Integrated pipeline to process real audio files (MP3/WAV) into Mel spectrograms.
- **🔍 Bit-Perfect Tokenization**: Fully compatible with OpenAI's tokenizer, producing identical results to the PyTorch reference implementation.

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `main.mojo` | 🎮 The entry point. Orchestrates weight loading, audio processing, and transcription. |
| `whisper.mojo` | 🧠 The "Brain". Contains the `Whisper` model and incremental decoding logic. |
| `layers.mojo` | 🧱 Core building blocks with **KVCache** support and parallelized attention. |
| `whisper_tensor.mojo` | 🧬 Mathematical foundation. Implements parallelized & SIMD-optimized tensor ops. |
| `tokenizer.mojo` | 🔤 Decodes numeric tokens into human-readable text. |
| `loader.mojo` | 📥 Efficient binary weight loader. |
| `export_weights.py` | 🐍 Python bridge for weight export and audio preprocessing. |

## 🛠️ Getting Started

### 📋 Prerequisites

- **Mojo SDK** (v24.5+)
- **Python Environment** with `torch`, `transformers`, `soundfile`, `scipy`, `requests`

### 🏗️ Installation & Execution

1. **Clone & Setup**
   ```bash
   git clone https://github.com/antonvice/whisper.Mojo.git
   cd whisper.Mojo
   ```

2. **Prepare Weights & Audio**
   ```bash
   uv run export_weights.py
   ```

3. **Build & Run (Recommended for Speed)**
   For the best performance, compile to a native binary:
   ```bash
   mojo build main.mojo
   ./main
   ```

## 📊 Optimization Details

This implementation is designed to showcase Mojo's performance advantages:

1. **KV-Cache**: Instead of re-computing the entire sequence for every new token, we cache the keys and values of previous tokens.
2. **Parallel Heads**: All attention heads in a layer are processed simultaneously on multiple CPU cores.
3. **SIMD Vectorization**: Inner loops are manually tuned to use 256-bit or 512-bit registers (depending on hardware).

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
