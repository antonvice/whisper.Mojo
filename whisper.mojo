from whisper_tensor import Tensor, conv1d, layer_norm, matmul, gelu, argmax, transpose_conv_weights
from memory import memcpy, LegacyUnsafePointer
from algorithm import parallelize
from sys import simd_width_of as simdwidthof
from time import perf_counter
from layers import ResidualAttentionBlock, KVCache, LayerCache
from loader import WeightLoader
from math import sin, cos
from collections import List


struct WhisperConfig:
    var d_model: Int
    var n_heads: Int
    var n_layers: Int
    var vocab_size: Int

    fn __init__(
        out self, d_model: Int, n_heads: Int, n_layers: Int, vocab_size: Int
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size

    @staticmethod
    fn tiny() -> WhisperConfig:
        return WhisperConfig(384, 6, 4, 51865)


struct WhisperEncoder:
    var conv1_w: Tensor
    var conv1_b: Tensor
    var conv2_w: Tensor
    var conv2_b: Tensor
    var pos_emb: Tensor
    var blocks: List[ResidualAttentionBlock]
    var ln_post_w: Tensor
    var ln_post_b: Tensor

    fn __init__(out self, config: WhisperConfig):
        self.conv1_w = Tensor(0, 0)
        self.conv1_b = Tensor(0, 0)
        self.conv2_w = Tensor(0, 0)
        self.conv2_b = Tensor(0, 0)
        self.pos_emb = Tensor(0, 0)
        self.blocks = List[ResidualAttentionBlock]()
        for _ in range(config.n_layers):
            self.blocks.append(
                ResidualAttentionBlock(
                    config.d_model, config.n_heads, is_decoder=False
                )
            )
        self.ln_post_w = Tensor(0, 0)
        self.ln_post_b = Tensor(0, 0)

    fn load(mut self, mut loader: WeightLoader):
        self.conv1_w = transpose_conv_weights(loader.next_tensor(384, 80 * 3), 384, 80, 3)
        self.conv1_b = loader.next_tensor(1, 384)
        
        self.conv2_w = transpose_conv_weights(loader.next_tensor(384, 384 * 3), 384, 384, 3)
        self.conv2_b = loader.next_tensor(1, 384)
        self.pos_emb = loader.next_tensor(1500, 384)
        for i in range(len(self.blocks)):
            self.blocks[i].load(loader, is_decoder_block=False)
        self.ln_post_w = loader.next_tensor(1, 384)
        self.ln_post_b = loader.next_tensor(1, 384)

    fn forward(self, mel: Tensor) -> Tensor:
        var x1 = Tensor(384, 3000)
        conv1d(x1, mel, self.conv1_w, self.conv1_b, stride=1, padding=1)
        gelu(x1)

        var x2 = Tensor(384, 1500)
        conv1d(x2, x1, self.conv2_w, self.conv2_b, stride=2, padding=1)
        gelu(x2)

        var x = Tensor(1500, 384)
        alias width = simdwidthof[DType.float32]()
        @parameter
        fn prep_worker(i: Int):
            for j in range(0, 384, width):
                var val = SIMD[DType.float32, width](0.0)
                var p_vec = self.pos_emb.data.load[width=width](i * 384 + j)
                for w_idx in range(width):
                    val[w_idx] = x2.data[(j + w_idx) * 1500 + i] + p_vec[w_idx]
                x.data.store[width=width](i * 384 + j, val)
        parallelize[prep_worker](1500)

        for i in range(len(self.blocks)):
            var dummy_cache = LayerCache()
            x = self.blocks[i].forward(
                x, Tensor(0, 0), dummy_cache, use_cache=False
            )

        var out = Tensor(x.rows, x.cols)
        layer_norm(out, x, self.ln_post_w, self.ln_post_b)
        return out


struct WhisperDecoder:
    var token_emb: Tensor
    var pos_emb: Tensor
    var blocks: List[ResidualAttentionBlock]
    var ln_post_w: Tensor
    var ln_post_b: Tensor

    fn __init__(out self, config: WhisperConfig):
        self.token_emb = Tensor(0, 0)
        self.pos_emb = Tensor(0, 0)
        self.blocks = List[ResidualAttentionBlock]()
        for _ in range(config.n_layers):
            self.blocks.append(
                ResidualAttentionBlock(
                    config.d_model, config.n_heads, is_decoder=True
                )
            )
        self.ln_post_w = Tensor(0, 0)
        self.ln_post_b = Tensor(0, 0)

    fn load(mut self, mut loader: WeightLoader):
        self.token_emb = loader.next_tensor(51865, 384)
        self.pos_emb = loader.next_tensor(448, 384)
        for i in range(len(self.blocks)):
            self.blocks[i].load(loader, is_decoder_block=True)
        self.ln_post_w = loader.next_tensor(1, 384)
        self.ln_post_b = loader.next_tensor(1, 384)

    fn forward(
        self,
        tokens: List[Int],
        enc_out: Tensor,
        mut cache: KVCache,
        use_cache: Bool = False,
        start_pos: Int = 0,
    ) -> Tensor:
        var L_tgt = len(tokens)
        var x = Tensor(L_tgt, 384)
        for i in range(L_tgt):
            var token_id = tokens[i]
            for j in range(384):
                x.set(
                    i,
                    j,
                    self.token_emb.get(token_id, j)
                    + self.pos_emb.get(start_pos + i, j),
                )

        for i in range(len(self.blocks)):
            x = self.blocks[i].forward(
                x, enc_out, cache=cache.layers[i], use_cache=use_cache
            )

        var out = Tensor(x.rows, x.cols)
        layer_norm(out, x, self.ln_post_w, self.ln_post_b)

        var last_hidden = Tensor(1, 384)
        memcpy(dest=last_hidden.data, src=out.data.offset((L_tgt - 1) * 384), count=384)

        var logits = Tensor(1, 51865)
        matmul(logits, last_hidden, self.token_emb, Tensor(0, 0))
        return logits^


struct Whisper:
    var encoder: WhisperEncoder
    var decoder: WhisperDecoder
    var config: WhisperConfig

    fn __init__(out self):
        self.config = WhisperConfig.tiny()
        self.encoder = WhisperEncoder(self.config)
        self.decoder = WhisperDecoder(self.config)

    fn load(mut self, mut loader: WeightLoader):
        self.encoder.load(loader)
        self.decoder.load(loader)

    fn transcribe(self, mel: Tensor) -> List[Int]:
        var enc_out = self.encoder.forward(mel)

        var tokens = List[Int]()
        tokens.append(50258)  # <|startoftranscript|>
        tokens.append(50259)  # <|en|>
        tokens.append(50359)  # <|transcribe|>
        tokens.append(50363)  # <|notimestamps|>

        # Pre-allocate KV cache for the decoder
        var cache = KVCache(self.config.n_layers, self.config.d_model, 448)

        # First pass: process the prefix tokens to fill the cache
        var logits = self.decoder.forward(
            tokens, enc_out, cache=cache, use_cache=True, start_pos=0
        )
        var next_token = argmax(logits)
        
        var all_tokens = List[Int]()
        for i in range(len(tokens)):
            all_tokens.append(tokens[i])
        all_tokens.append(next_token)

        for _ in range(195):  # Already produced 1 token from prefix
            if next_token == 50257:  # <|endoftext|>
                break

            # Incremental pass: only process the LAST token
            var last_token_list = List[Int]()
            last_token_list.append(next_token)

            logits = self.decoder.forward(
                last_token_list,
                enc_out,
                cache=cache,
                use_cache=True,
                start_pos=cache.layers[0].current_len - 1,
            )
            next_token = argmax(logits)
            all_tokens.append(next_token)

        return all_tokens^
