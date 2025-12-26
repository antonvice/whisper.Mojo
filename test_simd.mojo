from sys import simd_width_of
from sys import simdwidthof

fn main():
    print(simdwidthof[DType.float32]())
    print(simd_width_of[DType.float32]())
