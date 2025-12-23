from whisper import Whisper, WhisperConfig
from whisper_tensor import Tensor
from loader import WeightLoader
from tokenizer import Tokenizer
import sys
from collections import List
from memory import memcpy


fn main() raises:
    print("Initializing Whisper Tiny in Mojo...")
    var whisper = Whisper()

    print("Loading weights from whisper_tiny_weights.bin...")
    var loader = WeightLoader("whisper_tiny_weights.bin")
    whisper.load(loader)

    print("Loading vocabulary from vocab.txt...")
    var tokenizer = Tokenizer("vocab.txt")

    print("Loading sample input from sample_input.bin...")
    var mel = Tensor(80, 3000)
    with open("sample_input.bin", "r") as f:
        var bytes = f.read_bytes()
        var src = bytes.unsafe_ptr().bitcast[Float32]()
        memcpy(dest=mel.data, src=src, count=80 * 3000)

    print("Running forward pass (transcribe)...")
    var tokens = whisper.transcribe(mel)

    print("Successfully ran forward pass.")
    print("\nToken IDs:")
    for i in range(len(tokens)):
        print(tokens[i], end=" ")
    print()

    var text = tokenizer.decode(tokens)
    print("\n" + "=" * 40)
    print("FINAL TRANSCRIPTION:")
    print("=" * 40)
    print(text)
    print("=" * 40)
    print("\nDone.")
