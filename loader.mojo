from whisper_tensor import Tensor
from memory import LegacyUnsafePointer, memcpy


struct WeightLoader:
    var raw_data: LegacyUnsafePointer[Float32]
    var size: Int
    var offset: Int

    fn __init__(out self, filename: String) raises:
        with open(filename, "r") as f:
            var bytes = f.read_bytes()
            self.size = len(bytes) // 4
            self.raw_data = LegacyUnsafePointer[Float32].alloc(self.size)

            # Using unsafe_ptr() for List
            var src = bytes.unsafe_ptr().bitcast[Float32]()
            memcpy(dest=self.raw_data, src=src, count=self.size)
            self.offset = 0

    fn next_tensor(mut self, rows: Int, cols: Int) -> Tensor:
        var count = rows * cols
        var t = Tensor(rows, cols)
        var src = self.raw_data + self.offset
        memcpy(dest=t.data, src=src, count=count)
        self.offset += count
        return t

    fn deinit(owned self):
        if self.raw_data:
            self.raw_data.free()
