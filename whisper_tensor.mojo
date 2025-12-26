from memory import LegacyUnsafePointer, memcpy, memset_zero
from math import sqrt, tanh, exp
from algorithm import parallelize
from sys import simd_width_of as simdwidthof


struct Tensor(Copyable, ImplicitlyCopyable, Movable):
    var data: LegacyUnsafePointer[Float32]
    var rows: Int
    var cols: Int
    var size: Int
    var is_view: Bool

    fn __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.size = rows * cols
        self.is_view = False
        self.data = LegacyUnsafePointer[Float32].alloc(self.size)
        memset_zero(self.data, self.size)

    @staticmethod
    fn view(data: LegacyUnsafePointer[Float32], rows: Int, cols: Int) -> Self:
        var t = Self(0, 0)
        t.data = data
        t.rows = rows
        t.cols = cols
        t.size = rows * cols
        t.is_view = True
        return t

    fn __copyinit__(out self, existing: Self):
        self.rows = existing.rows
        self.cols = existing.cols
        self.size = existing.size
        self.is_view = existing.is_view
        if existing.is_view:
            self.data = existing.data
        else:
            self.data = LegacyUnsafePointer[Float32].alloc(self.size)
            memcpy(dest=self.data, src=existing.data, count=self.size)

    fn __moveinit__(out self, deinit existing: Self):
        self.rows = existing.rows
        self.cols = existing.cols
        self.size = existing.size
        self.is_view = existing.is_view
        self.data = existing.data
        existing.data = LegacyUnsafePointer[Float32]()

    fn deinit(owned self):
        if self.data and not self.is_view:
            self.data.free()

    fn load(self, idx: Int) -> Float32:
        return self.data[idx]

    fn store(mut self, idx: Int, val: Float32):
        self.data[idx] = val

    fn get(self, r: Int, c: Int) -> Float32:
        return self.data[r * self.cols + c]

    fn set(mut self, r: Int, c: Int, val: Float32):
        self.data[r * self.cols + c] = val


fn matmul(mut C: Tensor, A: Tensor, B: Tensor, bias: Tensor):
    alias width = simdwidthof[DType.float32]()
    var M = A.rows
    var N = B.rows
    var K = A.cols

    var K_rounded = (K // width) * width
    if M <= 4:
        @parameter
        fn worker_vector(n: Int):
            for m in range(M):
                var sum = SIMD[DType.float32, width](0.0)
                var row_a = m * K
                var row_b = n * K
                for k in range(0, K_rounded, width):
                    sum += A.data.load[width=width](row_a + k) * B.data.load[width=width](row_b + k)
                
                var final_sum = sum.reduce_add()
                for k in range(K_rounded, K):
                    final_sum += A.data[row_a + k] * B.data[row_b + k]
                
                if bias.size > 0:
                    final_sum += bias.data[n]
                C.store(m * N + n, final_sum)
        parallelize[worker_vector](N)
    else:
        # For encoder (M=1500), parallelize over M
        @parameter
        fn worker_matrix(m: Int):
            alias tile_n = 8
            var n_idx = 0
            while n_idx + tile_n <= N:
                var sum0 = SIMD[DType.float32, width](0.0)
                var sum1 = SIMD[DType.float32, width](0.0)
                var sum2 = SIMD[DType.float32, width](0.0)
                var sum3 = SIMD[DType.float32, width](0.0)
                var sum4 = SIMD[DType.float32, width](0.0)
                var sum5 = SIMD[DType.float32, width](0.0)
                var sum6 = SIMD[DType.float32, width](0.0)
                var sum7 = SIMD[DType.float32, width](0.0)
                
                var a_row_off = m * K
                for k in range(0, K_rounded, width):
                    var a_vec = A.data.load[width=width](a_row_off + k)
                    sum0 += a_vec * B.data.load[width=width]((n_idx + 0) * K + k)
                    sum1 += a_vec * B.data.load[width=width]((n_idx + 1) * K + k)
                    sum2 += a_vec * B.data.load[width=width]((n_idx + 2) * K + k)
                    sum3 += a_vec * B.data.load[width=width]((n_idx + 3) * K + k)
                    sum4 += a_vec * B.data.load[width=width]((n_idx + 4) * K + k)
                    sum5 += a_vec * B.data.load[width=width]((n_idx + 5) * K + k)
                    sum6 += a_vec * B.data.load[width=width]((n_idx + 6) * K + k)
                    sum7 += a_vec * B.data.load[width=width]((n_idx + 7) * K + k)
                
                var f0 = sum0.reduce_add()
                var f1 = sum1.reduce_add()
                var f2 = sum2.reduce_add()
                var f3 = sum3.reduce_add()
                var f4 = sum4.reduce_add()
                var f5 = sum5.reduce_add()
                var f6 = sum6.reduce_add()
                var f7 = sum7.reduce_add()
                
                for k in range(K_rounded, K):
                    var a_val = A.data[a_row_off + k]
                    f0 += a_val * B.data[(n_idx + 0) * K + k]
                    f1 += a_val * B.data[(n_idx + 1) * K + k]
                    f2 += a_val * B.data[(n_idx + 2) * K + k]
                    f3 += a_val * B.data[(n_idx + 3) * K + k]
                    f4 += a_val * B.data[(n_idx + 4) * K + k]
                    f5 += a_val * B.data[(n_idx + 5) * K + k]
                    f6 += a_val * B.data[(n_idx + 6) * K + k]
                    f7 += a_val * B.data[(n_idx + 7) * K + k]

                var c_base = m * N + n_idx
                C.store(c_base + 0, f0 + (bias.data[n_idx + 0] if bias.size > 0 else 0))
                C.store(c_base + 1, f1 + (bias.data[n_idx + 1] if bias.size > 0 else 0))
                C.store(c_base + 2, f2 + (bias.data[n_idx + 2] if bias.size > 0 else 0))
                C.store(c_base + 3, f3 + (bias.data[n_idx + 3] if bias.size > 0 else 0))
                C.store(c_base + 4, f4 + (bias.data[n_idx + 4] if bias.size > 0 else 0))
                C.store(c_base + 5, f5 + (bias.data[n_idx + 5] if bias.size > 0 else 0))
                C.store(c_base + 6, f6 + (bias.data[n_idx + 6] if bias.size > 0 else 0))
                C.store(c_base + 7, f7 + (bias.data[n_idx + 7] if bias.size > 0 else 0))
                n_idx += tile_n
            
            # Remainder
            for n in range(n_idx, N):
                var sum = SIMD[DType.float32, width](0.0)
                var row_a = m * K
                var row_b = n * K
                for k in range(0, K_rounded, width):
                    sum += A.data.load[width=width](row_a + k) * B.data.load[width=width](row_b + k)
                var final_sum = sum.reduce_add()
                for k in range(K_rounded, K):
                    final_sum += A.data[row_a + k] * B.data[row_b + k]
                C.store(m * N + n, final_sum + (bias.data[n] if bias.size > 0 else 0))
        
        parallelize[worker_matrix](M)


fn layer_norm(
    mut out: Tensor,
    inp: Tensor,
    gamma: Tensor,
    beta: Tensor,
    eps: Float32 = 1e-5,
):
    alias width = simdwidthof[DType.float32]()
    var rows = inp.rows
    var cols = inp.cols

    @parameter
    fn worker(i: Int):
        var sum_simd = SIMD[DType.float32, width](0.0)
        var sq_sum_simd = SIMD[DType.float32, width](0.0)
        for j in range(0, cols, width):
            var val = inp.data.load[width=width](i * cols + j)
            sum_simd += val
            sq_sum_simd += val * val
        
        var sum = sum_simd.reduce_add()
        var sq_sum = sq_sum_simd.reduce_add()

        var mean = sum / cols
        var var_val = (sq_sum / cols) - (mean * mean)
        var inv_std = 1.0 / sqrt(var_val + eps)

        for j in range(0, cols, width):
            var x = inp.data.load[width=width](i * cols + j)
            var g = gamma.data.load[width=width](j)
            var b = beta.data.load[width=width](j)
            var res = (x - SIMD[DType.float32, width](mean)) * SIMD[
                DType.float32, width
            ](inv_std) * g + b
            out.data.store(i * cols + j, res)

    parallelize[worker](rows)


fn gelu(mut t: Tensor):
    alias width = simdwidthof[DType.float32]()
    var SQRT_2_PI = Float32(0.79788456)
    var COEFF = Float32(0.044715)

    @parameter
    fn worker(i_block: Int):
        var i = i_block * width
        var x = t.data.load[width=width](i)
        var x3 = x * x * x
        var inner = SIMD[DType.float32, width](SQRT_2_PI) * (
            x + SIMD[DType.float32, width](COEFF) * x3
        )
        var res = (
            SIMD[DType.float32, width](0.5)
            * x
            * (SIMD[DType.float32, width](1.0) + tanh(inner))
        )
        t.data.store(i, res)

    parallelize[worker](t.size // width)


fn softmax(mut t: Tensor):
    alias width = simdwidthof[DType.float32]()
    var cols = t.cols
    @parameter
    fn worker(i: Int):
        var max_val = t.get(i, 0)
        var j_idx = 0
        if cols >= width:
            var max_val_simd = SIMD[DType.float32, width](max_val)
            for j in range(0, cols - cols % width, width):
                max_val_simd = max(max_val_simd, t.data.load[width=width](i * cols + j))
                j_idx += width
            max_val = max_val_simd.reduce_max()
        
        for j in range(j_idx, cols):
            if t.get(i, j) > max_val:
                max_val = t.get(i, j)

        var sum_exp: Float32 = 0.0
        j_idx = 0
        if cols >= width:
            var sum_exp_simd = SIMD[DType.float32, width](0.0)
            for j in range(0, cols - cols % width, width):
                var val = exp(t.data.load[width=width](i * cols + j) - max_val)
                t.data.store(i * cols + j, val)
                sum_exp_simd += val
                j_idx += width
            sum_exp = sum_exp_simd.reduce_add()
        
        for j in range(j_idx, cols):
            var val = exp(t.get(i, j) - max_val)
            t.set(i, j, val)
            sum_exp += val

        j_idx = 0
        if cols >= width:
            for j in range(0, cols - cols % width, width):
                var val = t.data.load[width=width](i * cols + j) / sum_exp
                t.data.store(i * cols + j, val)
                j_idx += width
        
        for j in range(j_idx, cols):
            t.set(i, j, t.get(i, j) / sum_exp)

    parallelize[worker](t.rows)


fn transpose_conv_weights(w: Tensor, C_out: Int, C_in: Int, K: Int) -> Tensor:
    var new_w = Tensor(C_out * K, C_in)
    for co in range(C_out):
        for ci in range(C_in):
            for k in range(K):
                new_w.data[(co * K + k) * C_in + ci] = w.data[co * (C_in * K) + ci * K + k]
    return new_w

fn conv1d(
    mut out: Tensor,
    inp: Tensor,
    weight: Tensor,
    bias: Tensor,
    stride: Int,
    padding: Int,
):
    alias width = simdwidthof[DType.float32]()
    var C_out = out.rows
    var L_out = out.cols
    var C_in = inp.rows
    var L_in = inp.cols
    var K = weight.size // (C_out * C_in)

    # 1. Transpose input from (C_in, L_in) to (L_in, C_in)
    var inp_T = Tensor(L_in, C_in)
    @parameter
    fn trans_inp(li: Int):
        for ci in range(C_in):
            inp_T.data[li * C_in + ci] = inp.data[ci * L_in + li]
    parallelize[trans_inp](L_in)

    # Note: Weights should be pre-transposed to (C_out, K, C_in) for maximum speed.
    # For now, we'll assume they are already in the right layout or we'll handle it once.

    @parameter
    fn worker(co: Int):
        var b = bias.data[co]
        for lo in range(L_out):
            var dot_simd = SIMD[DType.float32, width](0.0)
            var start_l = lo * stride - padding
            
            # Manually unroll K=3 for Whisper
            if K == 3:
                # k = 0
                var li0 = start_l
                if li0 >= 0 and li0 < L_in:
                    var li0_off = li0 * C_in
                    var w0_off = (co * 3 + 0) * C_in
                    for ci in range(0, C_in, width):
                        dot_simd += inp_T.data.load[width=width](li0_off + ci) * weight.data.load[width=width](w0_off + ci)
                
                # k = 1
                var li1 = start_l + 1
                if li1 >= 0 and li1 < L_in:
                    var li1_off = li1 * C_in
                    var w1_off = (co * 3 + 1) * C_in
                    for ci in range(0, C_in, width):
                        dot_simd += inp_T.data.load[width=width](li1_off + ci) * weight.data.load[width=width](w1_off + ci)
                
                # k = 2
                var li2 = start_l + 2
                if li2 >= 0 and li2 < L_in:
                    var li2_off = li2 * C_in
                    var w2_off = (co * 3 + 2) * C_in
                    for ci in range(0, C_in, width):
                        dot_simd += inp_T.data.load[width=width](li2_off + ci) * weight.data.load[width=width](w2_off + ci)
            else:
                for k in range(K):
                    var li = start_l + k
                    if li >= 0 and li < L_in:
                        var li_off = li * C_in
                        var w_off = (co * K + k) * C_in
                        for ci in range(0, C_in, width):
                            dot_simd += inp_T.data.load[width=width](li_off + ci) * weight.data.load[width=width](w_off + ci)
            
            out.data[co * L_out + lo] = dot_simd.reduce_add() + b

    parallelize[worker](C_out, 32)


fn argmax(t: Tensor) -> Int:
    var max_val = t.load(0)
    var max_idx = 0
    for i in range(1, t.size):
        var val = t.load(i)
        if val > max_val:
            max_val = val
            max_idx = i
    return max_idx
