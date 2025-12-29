# Whisper-Tiny static configuration
# All dimensions are compile-time constants for optimization

alias D_MODEL: Int = 384
alias N_HEADS: Int = 6
alias N_LAYERS: Int = 4
alias VOCAB_SIZE: Int = 51865
alias HEAD_DIM: Int = 64  # D_MODEL // N_HEADS
alias MAX_SEQ_LEN: Int = 1500  # Encoder sequence length
alias MAX_TOKENS: Int = 448  # Max decoder tokens
alias N_MELS: Int = 80  # Mel spectrogram bins

# Derived dimensions for matmul
alias D_MODEL_X3: Int = 1152  # D_MODEL * 3 for QKV projection
alias D_MODEL_X4: Int = 1536  # D_MODEL * 4 for FFN
alias CONV1_IN: Int = 240  # N_MELS * 3 (kernel size)
alias CONV2_IN: Int = 1152  # D_MODEL * 3 (kernel size)
