from whisper_tensor import Tensor, matmul, layer_norm, softmax, gelu
from loader import WeightLoader
from math import sqrt


struct MultiHeadAttention(Copyable, ImplicitlyCopyable, Movable):
    var q_proj_w: Tensor
    var q_proj_b: Tensor
    var k_proj_w: Tensor
    var v_proj_w: Tensor
    var v_proj_b: Tensor
    var out_proj_w: Tensor
    var out_proj_b: Tensor
    var n_heads: Int
    var head_dim: Int
    var d_model: Int

    fn __init__(out self, d_model: Int, n_heads: Int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj_w = Tensor(d_model, d_model)
        self.q_proj_b = Tensor(1, d_model)
        self.k_proj_w = Tensor(d_model, d_model)
        self.v_proj_w = Tensor(d_model, d_model)
        self.v_proj_b = Tensor(1, d_model)
        self.out_proj_w = Tensor(d_model, d_model)
        self.out_proj_b = Tensor(1, d_model)

    fn load(mut self, mut loader: WeightLoader, is_self_attn: Bool):
        self.q_proj_w = loader.next_tensor(self.d_model, self.d_model)
        self.q_proj_b = loader.next_tensor(1, self.d_model)
        self.k_proj_w = loader.next_tensor(self.d_model, self.d_model)
        self.v_proj_w = loader.next_tensor(self.d_model, self.d_model)
        self.v_proj_b = loader.next_tensor(1, self.d_model)
        self.out_proj_w = loader.next_tensor(self.d_model, self.d_model)
        self.out_proj_b = loader.next_tensor(1, self.d_model)

    fn forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Bool = False
    ) -> Tensor:
        var q_len = query.rows
        var k_len = key.rows

        var q = Tensor(q_len, self.d_model)
        matmul(q, query, self.q_proj_w, self.q_proj_b)

        var k = Tensor(k_len, self.d_model)
        var k_bias_empty = Tensor(0, 0)
        matmul(k, key, self.k_proj_w, k_bias_empty)

        var v = Tensor(k_len, self.d_model)
        matmul(v, value, self.v_proj_w, self.v_proj_b)

        var out = Tensor(q_len, self.d_model)

        for h in range(self.n_heads):
            var scores = Tensor(q_len, k_len)
            var scale = 1.0 / sqrt(Float32(self.head_dim))

            for i in range(q_len):
                for j in range(k_len):
                    var dot: Float32 = 0.0
                    for d in range(self.head_dim):
                        var q_val = q.get(i, h * self.head_dim + d)
                        var k_val = k.get(j, h * self.head_dim + d)
                        dot += q_val * k_val

                    var score = dot * scale
                    if mask and j > i:
                        score = -1e10

                    scores.set(i, j, score)

            softmax(scores)

            for i in range(q_len):
                for d in range(self.head_dim):
                    var val: Float32 = 0.0
                    for j in range(k_len):
                        val += scores.get(i, j) * v.get(
                            j, h * self.head_dim + d
                        )
                    out.set(i, h * self.head_dim + d, val)

        var final_out = Tensor(q_len, self.d_model)
        matmul(final_out, out, self.out_proj_w, self.out_proj_b)
        return final_out^

    fn __copyinit__(out self, existing: Self):
        self.q_proj_w = existing.q_proj_w
        self.q_proj_b = existing.q_proj_b
        self.k_proj_w = existing.k_proj_w
        self.v_proj_w = existing.v_proj_w
        self.v_proj_b = existing.v_proj_b
        self.out_proj_w = existing.out_proj_w
        self.out_proj_b = existing.out_proj_b
        self.n_heads = existing.n_heads
        self.head_dim = existing.head_dim
        self.d_model = existing.d_model

    fn __moveinit__(out self, deinit existing: Self):
        self.q_proj_w = existing.q_proj_w^
        self.q_proj_b = existing.q_proj_b^
        self.k_proj_w = existing.k_proj_w^
        self.v_proj_w = existing.v_proj_w^
        self.v_proj_b = existing.v_proj_b^
        self.out_proj_w = existing.out_proj_w^
        self.out_proj_b = existing.out_proj_b^
        self.n_heads = existing.n_heads
        self.head_dim = existing.head_dim
        self.d_model = existing.d_model


struct ResidualAttentionBlock(Copyable, ImplicitlyCopyable, Movable):
    var attn: MultiHeadAttention
    var attn_ln_w: Tensor
    var attn_ln_b: Tensor
    var cross_attn: MultiHeadAttention
    var cross_attn_ln_w: Tensor
    var cross_attn_ln_b: Tensor
    var mlp_fc1_w: Tensor
    var mlp_fc1_b: Tensor
    var mlp_fc2_w: Tensor
    var mlp_fc2_b: Tensor
    var mlp_ln_w: Tensor
    var mlp_ln_b: Tensor
    var is_decoder: Bool
    var d_model: Int

    fn __init__(out self, d_model: Int, n_heads: Int, is_decoder: Bool):
        self.d_model = d_model
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.attn_ln_w = Tensor(1, d_model)
        self.attn_ln_b = Tensor(1, d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn_ln_w = Tensor(1, d_model)
        self.cross_attn_ln_b = Tensor(1, d_model)
        self.mlp_fc1_w = Tensor(d_model * 4, d_model)
        self.mlp_fc1_b = Tensor(1, d_model * 4)
        self.mlp_fc2_w = Tensor(d_model, d_model * 4)
        self.mlp_fc2_b = Tensor(1, d_model)
        self.mlp_ln_w = Tensor(1, d_model)
        self.mlp_ln_b = Tensor(1, d_model)
        self.is_decoder = is_decoder

    fn load(mut self, mut loader: WeightLoader, is_decoder_block: Bool):
        self.attn.load(loader, is_self_attn=True)
        self.attn_ln_w = loader.next_tensor(1, self.d_model)
        self.attn_ln_b = loader.next_tensor(1, self.d_model)

        if is_decoder_block:
            self.cross_attn.load(loader, is_self_attn=False)
            self.cross_attn_ln_w = loader.next_tensor(1, self.d_model)
            self.cross_attn_ln_b = loader.next_tensor(1, self.d_model)

        self.mlp_fc1_w = loader.next_tensor(self.d_model * 4, self.d_model)
        self.mlp_fc1_b = loader.next_tensor(1, self.d_model * 4)
        self.mlp_fc2_w = loader.next_tensor(self.d_model, self.d_model * 4)
        self.mlp_fc2_b = loader.next_tensor(1, self.d_model)
        self.mlp_ln_w = loader.next_tensor(1, self.d_model)
        self.mlp_ln_b = loader.next_tensor(1, self.d_model)

    fn forward(self, x: Tensor, enc_out: Tensor = Tensor(0, 0)) -> Tensor:
        var x_norm = Tensor(x.rows, x.cols)
        layer_norm(x_norm, x, self.attn_ln_w, self.attn_ln_b)

        var self_attn_out = self.attn.forward(
            x_norm, x_norm, x_norm, mask=self.is_decoder
        )

        var current_x = Tensor(x.rows, x.cols)
        for i in range(x.size):
            current_x.store(i, x.load(i) + self_attn_out.load(i))

        if self.is_decoder and enc_out.size > 0:
            var x_norm_cross = Tensor(current_x.rows, current_x.cols)
            layer_norm(
                x_norm_cross,
                current_x,
                self.cross_attn_ln_w,
                self.cross_attn_ln_b,
            )

            var cross_attn_out = self.cross_attn.forward(
                x_norm_cross, enc_out, enc_out
            )

            var x_res2 = Tensor(current_x.rows, current_x.cols)
            for i in range(current_x.size):
                x_res2.store(i, current_x.load(i) + cross_attn_out.load(i))
            current_x = x_res2

        var x_norm_mlp = Tensor(current_x.rows, current_x.cols)
        layer_norm(x_norm_mlp, current_x, self.mlp_ln_w, self.mlp_ln_b)

        var hidden = Tensor(current_x.rows, self.d_model * 4)
        matmul(hidden, x_norm_mlp, self.mlp_fc1_w, self.mlp_fc1_b)
        gelu(hidden)

        var mlp_out = Tensor(current_x.rows, current_x.cols)
        matmul(mlp_out, hidden, self.mlp_fc2_w, self.mlp_fc2_b)

        var final_out = Tensor(current_x.rows, current_x.cols)
        for i in range(current_x.size):
            final_out.store(i, current_x.load(i) + mlp_out.load(i))

        return final_out^

    fn __copyinit__(out self, existing: Self):
        self.attn = existing.attn
        self.attn_ln_w = existing.attn_ln_w
        self.attn_ln_b = existing.attn_ln_b
        self.cross_attn = existing.cross_attn
        self.cross_attn_ln_w = existing.cross_attn_ln_w
        self.cross_attn_ln_b = existing.cross_attn_ln_b
        self.mlp_fc1_w = existing.mlp_fc1_w
        self.mlp_fc1_b = existing.mlp_fc1_b
        self.mlp_fc2_w = existing.mlp_fc2_w
        self.mlp_fc2_b = existing.mlp_fc2_b
        self.mlp_ln_w = existing.mlp_ln_w
        self.mlp_ln_b = existing.mlp_ln_b
        self.is_decoder = existing.is_decoder
        self.d_model = existing.d_model

    fn __moveinit__(out self, deinit existing: Self):
        self.attn = existing.attn^
        self.attn_ln_w = existing.attn_ln_w^
        self.attn_ln_b = existing.attn_ln_b^
        self.cross_attn = existing.cross_attn^
        self.cross_attn_ln_w = existing.cross_attn_ln_w^
        self.cross_attn_ln_b = existing.cross_attn_ln_b^
        self.mlp_fc1_w = existing.mlp_fc1_w^
        self.mlp_fc1_b = existing.mlp_fc1_b^
        self.mlp_fc2_w = existing.mlp_fc2_w^
        self.mlp_fc2_b = existing.mlp_fc2_b^
        self.mlp_ln_w = existing.mlp_ln_w^
        self.mlp_ln_b = existing.mlp_ln_b^
        self.is_decoder = existing.is_decoder
        self.d_model = existing.d_model
