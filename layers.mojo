from whisper_tensor import (
    Tensor, matmul, layer_norm, softmax, gelu,
    matmul_384x384, matmul_384x1536, matmul_1536x384,
    matmul_Q_K, matmul_S_V
)
from config import MAX_SEQ_LEN, HEAD_DIM
from loader import WeightLoader
from math import sqrt, exp
from algorithm import parallelize
from sys import simd_width_of as simdwidthof
from memory import memcpy, LegacyUnsafePointer


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
        self.cross_k = Tensor(1500, d_model)
        self.cross_v = Tensor(1500, d_model)
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
        if q_len == MAX_SEQ_LEN:
            try:
                matmul_384x384[MAX_SEQ_LEN](q, query, self.q_proj_w, self.q_proj_b)
            except:
                matmul(q, query, self.q_proj_w, self.q_proj_b)
        else:
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
                var dest_offset = cache.current_len * self.d_model
                memcpy(dest=cache.self_k.data.offset(dest_offset), src=new_k.data, count=q_len * self.d_model)
                memcpy(dest=cache.self_v.data.offset(dest_offset), src=new_v.data, count=q_len * self.d_model)
                cache.current_len += q_len

                # Use full cache for attention without copying
                k = Tensor.view(cache.self_k.data, cache.current_len, self.d_model)
                v = Tensor.view(cache.self_v.data, cache.current_len, self.d_model)
            else:
                # Cross attention
                if not cache.has_cross:
                    var k_bias_empty = Tensor(0, 0)
                    matmul(cache.cross_k, key, self.k_proj_w, k_bias_empty)
                    matmul(cache.cross_v, value, self.v_proj_w, self.v_proj_b)
                    cache.has_cross = True
                
                k = Tensor.view(cache.cross_k.data, k_len, self.d_model)
                v = Tensor.view(cache.cross_v.data, k_len, self.d_model)
        else:
            k = Tensor(k_len, self.d_model)
            var k_bias_empty = Tensor(0, 0)
            if k_len == MAX_SEQ_LEN:
                try:
                    matmul_384x384[MAX_SEQ_LEN](k, key, self.k_proj_w, k_bias_empty)
                except:
                    matmul(k, key, self.k_proj_w, k_bias_empty)
            else:
                matmul(k, key, self.k_proj_w, k_bias_empty)

            v = Tensor(k_len, self.d_model)
            if k_len == MAX_SEQ_LEN:
                try:
                    matmul_384x384[MAX_SEQ_LEN](v, value, self.v_proj_w, self.v_proj_b)
                except:
                    matmul(v, value, self.v_proj_w, self.v_proj_b)
            else:
                matmul(v, value, self.v_proj_w, self.v_proj_b)

        var final_k_len = k.rows
        var out = Tensor(q_len, self.d_model)
        alias width = 8

        @parameter
        fn head_worker(h: Int):
            var scale = 1.0 / sqrt(Float32(self.head_dim))
            
            if q_len == 1:
                var scores_ptr = LegacyUnsafePointer[Float32].alloc(1500)
                var max_score = Float32(-1e10)
                
                # Hoist query head into registers (head_dim=64, so 8 SIMD registers)
                var q0 = q.data.load[width=width](h * self.head_dim + 0)
                var q1 = q.data.load[width=width](h * self.head_dim + 8)
                var q2 = q.data.load[width=width](h * self.head_dim + 16)
                var q3 = q.data.load[width=width](h * self.head_dim + 24)
                var q4 = q.data.load[width=width](h * self.head_dim + 32)
                var q5 = q.data.load[width=width](h * self.head_dim + 40)
                var q6 = q.data.load[width=width](h * self.head_dim + 48)
                var q7 = q.data.load[width=width](h * self.head_dim + 56)
                
                # 1. Compute scores
                for j in range(final_k_len):
                    var k_ptr = k.data.offset(j * self.d_model + h * self.head_dim)
                    var dot_simd = q0 * k_ptr.load[width=width](0)
                    dot_simd += q1 * k_ptr.load[width=width](8)
                    dot_simd += q2 * k_ptr.load[width=width](16)
                    dot_simd += q3 * k_ptr.load[width=width](24)
                    dot_simd += q4 * k_ptr.load[width=width](32)
                    dot_simd += q5 * k_ptr.load[width=width](40)
                    dot_simd += q6 * k_ptr.load[width=width](48)
                    dot_simd += q7 * k_ptr.load[width=width](56)
                    
                    var score = dot_simd.reduce_add() * scale
                    if mask and j > (cache.current_len - 1 if use_cache and is_self_attn else 0):
                        score = -1e10
                    
                    if score > max_score: max_score = score
                    scores_ptr[j] = score
                
                # 2. Softmax inline
                var sum_exp_simd = SIMD[DType.float32, width](0.0)
                var rounded_len = (final_k_len // width) * width
                for j in range(0, rounded_len, width):
                    var s = scores_ptr.load[width=width](j)
                    var e = exp(s - max_score)
                    scores_ptr.store[width=width](j, e)
                    sum_exp_simd += e
                
                var sum_exp = sum_exp_simd.reduce_add()
                for j in range(rounded_len, final_k_len):
                    var e = exp(scores_ptr[j] - max_score)
                    scores_ptr[j] = e
                    sum_exp += e
                
                var inv_sum_exp = 1.0 / sum_exp
                var inv_sum_simd = SIMD[DType.float32, width](inv_sum_exp)
                for j in range(0, rounded_len, width):
                    scores_ptr.store[width=width](j, scores_ptr.load[width=width](j) * inv_sum_simd)
                for j in range(rounded_len, final_k_len):
                    scores_ptr[j] *= inv_sum_exp
                
                # 3. Weighted sum (hoist output into registers)
                var o0 = SIMD[DType.float32, width](0.0)
                var o1 = SIMD[DType.float32, width](0.0)
                var o2 = SIMD[DType.float32, width](0.0)
                var o3 = SIMD[DType.float32, width](0.0)
                var o4 = SIMD[DType.float32, width](0.0)
                var o5 = SIMD[DType.float32, width](0.0)
                var o6 = SIMD[DType.float32, width](0.0)
                var o7 = SIMD[DType.float32, width](0.0)
                
                for j in range(final_k_len):
                    var s = SIMD[DType.float32, width](scores_ptr[j])
                    var v_ptr = v.data.offset(j * self.d_model + h * self.head_dim)
                    o0 += s * v_ptr.load[width=width](0)
                    o1 += s * v_ptr.load[width=width](8)
                    o2 += s * v_ptr.load[width=width](16)
                    o3 += s * v_ptr.load[width=width](24)
                    o4 += s * v_ptr.load[width=width](32)
                    o5 += s * v_ptr.load[width=width](40)
                    o6 += s * v_ptr.load[width=width](48)
                    o7 += s * v_ptr.load[width=width](56)
                
                out.data.store[width=width](h * self.head_dim + 0, o0)
                out.data.store[width=width](h * self.head_dim + 8, o1)
                out.data.store[width=width](h * self.head_dim + 16, o2)
                out.data.store[width=width](h * self.head_dim + 24, o3)
                out.data.store[width=width](h * self.head_dim + 32, o4)
                out.data.store[width=width](h * self.head_dim + 40, o5)
                out.data.store[width=width](h * self.head_dim + 48, o6)
                out.data.store[width=width](h * self.head_dim + 56, o7)
                
                scores_ptr.free()
            else:
                # Optimized block-based path for encoder/prefill using matmul
                var q_h = Tensor(q_len, self.head_dim)
                var k_h = Tensor(final_k_len, self.head_dim)
                var v_h = Tensor(final_k_len, self.head_dim)
                
                # Parallelize head-data extraction
                @parameter
                fn extract_worker(i: Int):
                    if i < q_len:
                        var q_off = i * self.d_model + h * self.head_dim
                        memcpy(dest=q_h.data.offset(i * self.head_dim), src=q.data.offset(q_off), count=self.head_dim)
                    
                    if i < final_k_len:
                        var k_off = i * self.d_model + h * self.head_dim
                        memcpy(dest=k_h.data.offset(i * self.head_dim), src=k.data.offset(k_off), count=self.head_dim)
                        memcpy(dest=v_h.data.offset(i * self.head_dim), src=v.data.offset(k_off), count=self.head_dim)
                
                parallelize[extract_worker](max(q_len, final_k_len))
                
                var scores = Tensor(q_len, final_k_len)
                var empty_bias = Tensor(0, 0)
                if q_len == MAX_SEQ_LEN:
                    try:
                        matmul_Q_K[MAX_SEQ_LEN, HEAD_DIM](scores, q_h, k_h, empty_bias)
                    except:
                        matmul(scores, q_h, k_h, empty_bias)
                else:
                    matmul(scores, q_h, k_h, empty_bias)
                
                # Scale, Mask and Softmax
                @parameter
                fn scale_mask_worker(i: Int):
                    var scale_simd = SIMD[DType.float32, width](scale)
                    for j in range(0, final_k_len - final_k_len % width, width):
                        var s_vec = scores.data.load[width=width](i * final_k_len + j) * scale_simd
                        if mask:
                            for w in range(width):
                                if j + w > (cache.current_len - q_len + i if use_cache and is_self_attn else i):
                                    s_vec[w] = -1e10
                        scores.data.store[width=width](i * final_k_len + j, s_vec)
                    
                    for j in range(final_k_len - final_k_len % width, final_k_len):
                        var score = scores.get(i, j) * scale
                        if mask and j > (cache.current_len - q_len + i if use_cache and is_self_attn else i):
                            score = -1e10
                        scores.set(i, j, score)
                parallelize[scale_mask_worker](q_len)
                
                softmax(scores)
                
                var v_h_T = Tensor(self.head_dim, final_k_len)
                for i in range(final_k_len):
                    for j in range(self.head_dim):
                        v_h_T.data[j * final_k_len + i] = v_h.data[i * self.head_dim + j]
                
                var out_h = Tensor(q_len, self.head_dim)
                if q_len == MAX_SEQ_LEN:
                    try:
                        matmul_S_V[MAX_SEQ_LEN, HEAD_DIM](out_h, scores, v_h_T, empty_bias)
                    except:
                        matmul(out_h, scores, v_h_T, empty_bias)
                else:
                    matmul(out_h, scores, v_h_T, empty_bias)
                
                @parameter
                fn scatter_worker(i: Int):
                    var out_off = i * self.d_model + h * self.head_dim
                    memcpy(dest=out.data.offset(out_off), src=out_h.data.offset(i * self.head_dim), count=self.head_dim)
                parallelize[scatter_worker](q_len)

        if q_len == 1:
            for h in range(self.n_heads):
                head_worker(h)
        else:
            parallelize[head_worker](self.n_heads)


        var final_out = Tensor(q_len, self.d_model)
        if q_len == MAX_SEQ_LEN:
            try:
                matmul_384x384[MAX_SEQ_LEN](final_out, out, self.out_proj_w, self.out_proj_b)
            except:
                matmul(final_out, out, self.out_proj_w, self.out_proj_b)
        else:
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
        alias width = 8
        @parameter
        fn res1_worker(i: Int):
            var off = i * width
            current_x.data.store[width=width](off, x.data.load[width=width](off) + self_attn_out.data.load[width=width](off))
        parallelize[res1_worker](x.size // width)

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
            @parameter
            fn res2_worker(i: Int):
                var off = i * width
                x_res2.data.store[width=width](off, current_x.data.load[width=width](off) + cross_attn_out.data.load[width=width](off))
            parallelize[res2_worker](current_x.size // width)
            current_x = x_res2

        var x_norm_mlp = Tensor(current_x.rows, current_x.cols)
        layer_norm(x_norm_mlp, current_x, self.mlp_ln_w, self.mlp_ln_b)

        var hidden = Tensor(current_x.rows, self.d_model * 4)
        if current_x.rows == MAX_SEQ_LEN:
            try:
                matmul_384x1536[MAX_SEQ_LEN](hidden, x_norm_mlp, self.mlp_fc1_w, self.mlp_fc1_b)
            except:
                matmul(hidden, x_norm_mlp, self.mlp_fc1_w, self.mlp_fc1_b)
        else:
            matmul(hidden, x_norm_mlp, self.mlp_fc1_w, self.mlp_fc1_b)
        gelu(hidden)

        var mlp_out = Tensor(current_x.rows, current_x.cols)
        if current_x.rows == MAX_SEQ_LEN:
            try:
                matmul_1536x384[MAX_SEQ_LEN](mlp_out, hidden, self.mlp_fc2_w, self.mlp_fc2_b)
            except:
                matmul(mlp_out, hidden, self.mlp_fc2_w, self.mlp_fc2_b)
        else:
            matmul(mlp_out, hidden, self.mlp_fc2_w, self.mlp_fc2_b)

        var final_out = Tensor(current_x.rows, current_x.cols)
        @parameter
        fn res3_worker(i: Int):
            var off = i * width
            final_out.data.store[width=width](off, current_x.data.load[width=width](off) + mlp_out.data.load[width=width](off))
        parallelize[res3_worker](current_x.size // width)

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
