from memory import LegacyUnsafePointer, memcpy, memset_zero
from math import sqrt, tanh, exp
from algorithm import parallelize
from sys import simdwidthof


struct Tensor(Copyable, ImplicitlyCopyable, Movable):
    var data: LegacyUnsafePointer[Float32]
    var rows: Int
    var cols: Int
    var size: Int

    fn __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.size = rows * cols
        self.data = LegacyUnsafePointer[Float32].alloc(self.size)
        memset_zero(self.data, self.size)

    fn __copyinit__(out self, existing: Self):
        self.rows = existing.rows
        self.cols = existing.cols
        self.size = existing.size
        self.data = LegacyUnsafePointer[Float32].alloc(self.size)
        memcpy(dest=self.data, src=existing.data, count=self.size)

    fn __moveinit__(out self, deinit existing: Self):
        self.rows = existing.rows
        self.cols = existing.cols
        self.size = existing.size
        self.data = existing.data
        _ = existing.data.address  # Just access it to mark as used
        existing.data = LegacyUnsafePointer[Float32]()
        _ = existing.data

    fn deinit(owned self):
        if self.data:
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

    # For decoder (M=1), parallelize over N (output features)
    if M <= 4:
        @parameter
        fn worker_vector(n: Int):
            for m in range(M):
                var sum = SIMD[DType.float32, width](0.0)
                for k in range(0, K, width):
                    var a_vec = A.data.load[width=width](m * K + k)
                    var b_vec = B.data.load[width=width](n * K + k)
                    sum += a_vec * b_vec
                var final_sum = sum.reduce_add()
                if bias.size > 0:
                    final_sum += bias.data[n]
                C.store(m * N + n, final_sum)
        parallelize[worker_vector](N)
    else:
        # For encoder (M=1500), parallelize over M
        @parameter
        fn worker_matrix(m: Int):
            # Tiling n to improve cache reuse of A's row
            # Process multiple n rows per a_vec load
            alias tile_n = 4
            for n_base in range(0, N, tile_n):
                var sum0 = SIMD[DType.float32, width](0.0)
                var sum1 = SIMD[DType.float32, width](0.0)
                var sum2 = SIMD[DType.float32, width](0.0)
                var sum3 = SIMD[DType.float32, width](0.0)
                
                for k in range(0, K, width):
                    var a_vec = A.data.load[width=width](m * K + k)
                    sum0 += a_vec * B.data.load[width=width]((n_base + 0) * K + k)
                    if n_base + 1 < N: sum1 += a_vec * B.data.load[width=width]((n_base + 1) * K + k)
                    if n_base + 2 < N: sum2 += a_vec * B.data.load[width=width]((n_base + 2) * K + k)
                    if n_base + 3 < N: sum3 += a_vec * B.data.load[width=width]((n_base + 3) * K + k)
                
                C.store(m * N + n_base + 0, sum0.reduce_add() + (bias.data[n_base + 0] if bias.size > 0 else 0))
                if n_base + 1 < N: C.store(m * N + n_base + 1, sum1.reduce_add() + (bias.data[n_base + 1] if bias.size > 0 else 0))
                if n_base + 2 < N: C.store(m * N + n_base + 2, sum2.reduce_add() + (bias.data[n_base + 2] if bias.size > 0 else 0))
                if n_base + 3 < N: C.store(m * N + n_base + 3, sum3.reduce_add() + (bias.data[n_base + 3] if bias.size > 0 else 0))
        
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
        # Find max - Safe tail handling
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

        # Exp and Sum - Safe tail handling
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

        # Division - Safe tail handling
        j_idx = 0
        if cols >= width:
            for j in range(0, cols - cols % width, width):
                var val = t.data.load[width=width](i * cols + j) / sum_exp
                t.data.store(i * cols + j, val)
                j_idx += width
        
        for j in range(j_idx, cols):
            t.set(i, j, t.get(i, j) / sum_exp)

    parallelize[worker](t.rows)


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

    @parameter
    fn worker(co: Int):
        for lo in range(L_out):
            var sum: Float32 = 0.0
            var start_l = lo * stride - padding
            
            # Vectorize over ci? No, ci is the outer loop here.
            # But we can vectorize the ci loop if we have enough elements.
            var sum_simd = SIMD[DType.float32, width](0.0)
            for ci in range(C_in):
                # For small Kernels like 3, we can just unroll k
                for k in range(K):
                    var li = start_l + k
                    if li >= 0 and li < L_in:
                        var w_idx = co * (C_in * K) + ci * K + k
                        sum += inp.data[ci * L_in + li] * weight.data[w_idx]

            if bias.size > 0:
                sum += bias.load(co)
            out.set(co, lo, sum)

    parallelize[worker](C_out)



fn argmax(t: Tensor) -> Int:
    var max_val = t.load(0)
    var max_idx = 0
    for i in range(1, t.size):
        var val = t.load(i)
        if val > max_val:
            max_val = val
            max_idx = i
    return max_idx
