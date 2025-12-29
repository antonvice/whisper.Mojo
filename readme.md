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

## 📈 Changelog

### [2025-12-29] - 🚀 Performance Breakthrough: Sub-Python Latency
- **🏁 Beat Python Baseline**: Achieved a total transcription time of **0.74s**, outperforming the Python/PyTorch reference implementation (0.78s).
- **🏎️ Register-Heavy Decoder**: Implemented a peak-performance decoder attention path using register-cached heads. Optimized decoding by switching to serial head loops, eliminating thread pool overhead for small tasks.
- **🔄 Contiguous Encoder Pipeline**: Redesigned the encoder to use transposed-output convolutions. Implemented **Blocked Parallelization** (grain size 16) in `conv1d` to eliminate cache-line contention (false sharing).
- **🧵 Parallel Logit Projection**: Optimized the final $1 \times 51k$ output layer to parallelize across all cores, maximizing throughput for small batch sizes.
- **🧱 MAX Engine Integration**: Leveraged Modular's MAX Engine specialized matmul kernels for all encoder Transformer blocks.
- **🛡️ Warning Cleanup**: Resolved compiler warnings in `Tensor.__moveinit__` while ensuring proper move semantics.

### [2025-12-26] - Performance Optimization Sprint (Part 2)
- **🚀 Advanced `conv1d` Vectorization**: Implemented a "Transpose-DotProduct" strategy for 1D convolutions, enabling full SIMD utilization. Optimized core Whisper filters (K=3) with manual unrolling and hoisting of accumulation logic.
- **⚡ Matrix-Matrix Matmul Tiling**: Enhanced the matrix multiplication kernel with 8x tiling and unrolling for the $N$ dimension. This significantly reduced memory pressure and improved throughput for large encoder blocks ($M=1500$).
- **🧬 Optimized Prefill (Attention)**: Optimized the prefill/encoder path by switching from manual scalar loops to high-performance `matmul`-based head processing. Added parallelized extraction and scatter of attention heads.
- **💾 Layout-Aware Weight Loading**: Integrated pre-transposition of convolutional weights during model loading to ensure optimal memory layout for inference.
- **🛡️ Robust SIMD Kernels**: Implemented generalized tail-handling in `matmul` and `conv1d`, ensuring stability across arbitrary sequence lengths and filter sizes.
- **📈 Benchmark Results**: Successfully reduced total transcription time to **~1.59s** (from ~3.3s), achieving a **2x overall speedup**. Encoder runtime reduced by over 35%.

### [2025-12-25] - Performance Optimization Sprint (Part 1)
*   **🚀 Optimized Matmul**: Implemented dynamic parallelization that adapts to matrix shapes. Added 1D tiling for better cache reuse and switched to hardware-native SIMD widths using `simdwidthof`.
*   **⚡ Vectorized Attention**: Fully vectorized the inner loops of `MultiHeadAttention`, accelerating both the dot-product score calculation and the weighted value sum.
*   **🧬 Optimized Tensor Primitives**: Vectorized `LayerNorm`, `Softmax`, and `GeLU` operations. Added safe tail-handling for non-multiple sequence lengths.
*   **💾 Fast Memory Operations**: Replaced slow scalar loops in KV-cache management with high-performance `memcpy` transfers.
*   **🧵 Threading Improvements**: Optimized thread distribution in decoder layers to ensure all CPU cores are utilized during incremental decoding (single-token generation).

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
