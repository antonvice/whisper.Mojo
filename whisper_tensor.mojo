from memory import LegacyUnsafePointer, memcpy, memset_zero
from math import sqrt, tanh, exp


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
    alias width = 8
    for m in range(A.rows):
        for n in range(B.rows):
            var sum = SIMD[DType.float32, width](0.0)
            for k in range(0, A.cols, width):
                var a_vec = A.data.load[width=width](m * A.cols + k)
                var b_vec = B.data.load[width=width](n * B.cols + k)
                sum += a_vec * b_vec
            var final_sum = sum.reduce_add()
            if bias.size > 0:
                final_sum += bias.data[n]
            C.store(m * C.cols + n, final_sum)


fn layer_norm(
    mut out: Tensor,
    inp: Tensor,
    gamma: Tensor,
    beta: Tensor,
    eps: Float32 = 1e-5,
):
    alias width = 8
    var cols = inp.cols
    for i in range(inp.rows):
        var sum: Float32 = 0.0
        var sq_sum: Float32 = 0.0
        for j in range(cols):
            var val = inp.load(i * cols + j)
            sum += val
            sq_sum += val * val

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


fn gelu(mut t: Tensor):
    alias width = 8
    var SQRT_2_PI = Float32(0.79788456)
    var COEFF = Float32(0.044715)
    for i in range(0, t.size, width):
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


fn softmax(mut t: Tensor):
    for i in range(t.rows):
        var max_val = t.get(i, 0)
        for j in range(1, t.cols):
            var val = t.get(i, j)
            if val > max_val:
                max_val = val

        var sum_exp: Float32 = 0.0
        for j in range(t.cols):
            var val = exp(t.get(i, j) - max_val)
            t.set(i, j, val)
            sum_exp += val

        for j in range(t.cols):
            t.set(i, j, t.get(i, j) / sum_exp)


fn conv1d(
    mut out: Tensor,
    inp: Tensor,
    weight: Tensor,
    bias: Tensor,
    stride: Int,
    padding: Int,
):
    var C_out = out.rows
    var L_out = out.cols
    var C_in = inp.rows
    var L_in = inp.cols
    var K = weight.size // (C_out * C_in)

    for co in range(C_out):
        for lo in range(L_out):
            var sum: Float32 = 0.0
            var start_l = lo * stride - padding
            for ci in range(C_in):
                for k in range(K):
                    var li = start_l + k
                    if li >= 0 and li < L_in:
                        var w_idx = co * (C_in * K) + ci * K + k
                        sum += inp.get(ci, li) * weight.load(w_idx)

            if bias.size > 0:
                sum += bias.load(co)
            out.set(co, lo, sum)


fn argmax(t: Tensor) -> Int:
    var max_val = t.load(0)
    var max_idx = 0
    for i in range(1, t.size):
        var val = t.load(i)
        if val > max_val:
            max_val = val
            max_idx = i
    return max_idx
