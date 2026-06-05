module {
  func.func private @fill_zeros_bf16_32_32_vector(memref<32x32xbf16>)
  func.func private @matmul_scalar_bf16_bf16(memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>)
  func.func private @add_bf16_vector(memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>)
  func.func private @matmul_bf16_bf16(memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>)
  func.func @gemm_0_0_0(%arg0: memref<32x64xbf16>, %arg1: memref<64x32xbf16>, %arg2: memref<32x32xbf16>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "___", tag = "gemm_(None, (), None, ())"} {
    call @fill_zeros_bf16_32_32_vector(%arg2) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>) -> ()
    return
  }
  func.func @top(%arg0: memref<32x64xbf16>, %arg1: memref<64x32xbf16>, %arg2: memref<32x32xbf16>) attributes {dataflow, itypes = "___"} {
    return
  }
}
