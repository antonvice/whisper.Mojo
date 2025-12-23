from whisper_tensor import Tensor, conv1d, layer_norm, matmul, gelu, argmax
from layers import ResidualAttentionBlock
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
        self.conv1_w = loader.next_tensor(384, 80 * 3)
        self.conv1_b = loader.next_tensor(1, 384)
        self.conv2_w = loader.next_tensor(384, 384 * 3)
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
        for i in range(1500):
            for j in range(384):
                x.set(i, j, x2.get(j, i) + self.pos_emb.get(i, j))

        for i in range(len(self.blocks)):
            x = self.blocks[i].forward(x)

        var out = Tensor(x.rows, x.cols)
        layer_norm(out, x, self.ln_post_w, self.ln_post_b)
        return out^


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

    fn forward(self, tokens: List[Int], enc_out: Tensor) -> Tensor:
        var L_tgt = len(tokens)
        var x = Tensor(L_tgt, 384)
        for i in range(L_tgt):
            var token_id = tokens[i]
            for j in range(384):
                x.set(
                    i,
                    j,
                    self.token_emb.get(token_id, j) + self.pos_emb.get(i, j),
                )

        for i in range(len(self.blocks)):
            x = self.blocks[i].forward(x, enc_out)

        var out = Tensor(x.rows, x.cols)
        layer_norm(out, x, self.ln_post_w, self.ln_post_b)

        var last_hidden = Tensor(1, 384)
        for j in range(384):
            last_hidden.store(j, out.get(L_tgt - 1, j))

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

        for _ in range(200):
            var logits = self.decoder.forward(tokens, enc_out)
            var next_token = argmax(logits)
            tokens.append(next_token)
            if next_token == 50257:  # <|endoftext|>
                break

        return tokens^
