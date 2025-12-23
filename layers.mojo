from whisper_tensor import Tensor, matmul, layer_norm, softmax, gelu
from loader import WeightLoader
from math import sqrt
from algorithm import parallelize


struct LayerCache(Copyable, ImplicitlyCopyable, Movable):
    var self_k: Tensor
    var self_v: Tensor
    var cross_k: Tensor
    var cross_v: Tensor
    var current_len: Int
    var has_cross: Bool

    fn __init__(out self):
        self.self_k = Tensor(0, 0)
        self.self_v = Tensor(0, 0)
        self.cross_k = Tensor(0, 0)
        self.cross_v = Tensor(0, 0)
        self.current_len = 0
        self.has_cross = False

    fn reset(mut self, d_model: Int, max_len: Int):
        self.self_k = Tensor(max_len, d_model)
        self.self_v = Tensor(max_len, d_model)
        self.current_len = 0
        self.has_cross = False

    fn __moveinit__(out self, deinit existing: Self):
        self.self_k = existing.self_k^
        self.self_v = existing.self_v^
        self.cross_k = existing.cross_k^
        self.cross_v = existing.cross_v^
        self.current_len = existing.current_len
        self.has_cross = existing.has_cross

    fn __copyinit__(out self, existing: Self):
        self.self_k = existing.self_k
        self.self_v = existing.self_v
        self.cross_k = existing.cross_k
        self.cross_v = existing.cross_v
        self.current_len = existing.current_len
        self.has_cross = existing.has_cross


struct KVCache(Copyable, ImplicitlyCopyable, Movable):
    var layers: List[LayerCache]

    fn __init__(out self, n_layers: Int, d_model: Int, max_len: Int):
        self.layers = List[LayerCache]()
        for _ in range(n_layers):
            var layer = LayerCache()
            layer.reset(d_model, max_len)
            self.layers.append(layer)

    fn __moveinit__(out self, deinit existing: Self):
        self.layers = existing.layers^

    fn __copyinit__(out self, existing: Self):
        self.layers = existing.layers.copy()


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
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Bool,
        mut cache: LayerCache,
        is_self_attn: Bool,
        use_cache: Bool,
    ) -> Tensor:
        var q_len = query.rows
        var k_len = key.rows

        var q = Tensor(q_len, self.d_model)
        matmul(q, query, self.q_proj_w, self.q_proj_b)

        var k: Tensor
        var v: Tensor

        if use_cache:
            if is_self_attn:
                # Compute k, v for the new tokens
                var new_k = Tensor(q_len, self.d_model)
                var k_bias_empty = Tensor(0, 0)
                matmul(new_k, key, self.k_proj_w, k_bias_empty)
                var new_v = Tensor(q_len, self.d_model)
                matmul(new_v, value, self.v_proj_w, self.v_proj_b)

                # Append to cache
                for i in range(q_len):
                    for j in range(self.d_model):
                        cache.self_k.set(
                            cache.current_len + i, j, new_k.get(i, j)
                        )
                        cache.self_v.set(
                            cache.current_len + i, j, new_v.get(i, j)
                        )
                cache.current_len += q_len

                # Use full cache for attention
                k = Tensor(cache.current_len, self.d_model)
                v = Tensor(cache.current_len, self.d_model)
                # This is a bit slow (copying), but simpler for now.
                # Optimized version would use slices or direct access in the worker.
                for i in range(cache.current_len):
                    for j in range(self.d_model):
                        k.set(i, j, cache.self_k.get(i, j))
                        v.set(i, j, cache.self_v.get(i, j))
            else:
                # Cross attention
                if not cache.has_cross:
                    var ck = Tensor(k_len, self.d_model)
                    var k_bias_empty = Tensor(0, 0)
                    matmul(ck, key, self.k_proj_w, k_bias_empty)
                    var cv = Tensor(k_len, self.d_model)
                    matmul(cv, value, self.v_proj_w, self.v_proj_b)
                    cache.cross_k = ck^
                    cache.cross_v = cv^
                    cache.has_cross = True
                k = cache.cross_k
                v = cache.cross_v
        else:
            k = Tensor(k_len, self.d_model)
            var k_bias_empty = Tensor(0, 0)
            matmul(k, key, self.k_proj_w, k_bias_empty)
            v = Tensor(k_len, self.d_model)
            matmul(v, value, self.v_proj_w, self.v_proj_b)

        var final_k_len = k.rows
        var out = Tensor(q_len, self.d_model)

        @parameter
        fn head_worker(h: Int):
            var scores = Tensor(q_len, final_k_len)
            var scale = 1.0 / sqrt(Float32(self.head_dim))

            for i in range(q_len):
                for j in range(final_k_len):
                    var dot: Float32 = 0.0
                    for d in range(self.head_dim):
                        var q_val = q.get(i, h * self.head_dim + d)
                        var k_val = k.get(j, h * self.head_dim + d)
                        dot += q_val * k_val

                    var score = dot * scale
                    if mask and j > (
                        cache.current_len - q_len + i if use_cache
                        and is_self_attn else j
                    ):
                        # Masking logic for incremental decoding:
                        # tokens can only see themselves and previous tokens.
                        # If not using cache or not self-attn, j > i is the standard causal mask.
                        # Wait, j > i is simpler.
                        pass

                    if mask and j > (
                        cache.current_len - q_len + i if use_cache
                        and is_self_attn else i
                    ):
                        score = -1e10

                    scores.set(i, j, score)

            softmax(scores)

            for i in range(q_len):
                for d in range(self.head_dim):
                    var val: Float32 = 0.0
                    for j in range(final_k_len):
                        val += scores.get(i, j) * v.get(
                            j, h * self.head_dim + d
                        )
                    out.set(i, h * self.head_dim + d, val)

        parallelize[head_worker](self.n_heads)

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

    fn forward(
        self,
        x: Tensor,
        enc_out: Tensor,
        mut cache: LayerCache,
        use_cache: Bool,
    ) -> Tensor:
        var x_norm = Tensor(x.rows, x.cols)
        layer_norm(x_norm, x, self.attn_ln_w, self.attn_ln_b)

        var self_attn_out = self.attn.forward(
            x_norm,
            x_norm,
            x_norm,
            mask=self.is_decoder,
            cache=cache,
            is_self_attn=True,
            use_cache=use_cache,
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
                x_norm_cross,
                enc_out,
                enc_out,
                mask=False,
                cache=cache,
                is_self_attn=False,
                use_cache=use_cache,
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
