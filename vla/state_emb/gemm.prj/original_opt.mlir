module {
  func.func private @fill_zeros_bf16_16_64_vector(memref<16x64xbf16>)
  func.func private @matmul_scalar_bf16_bf16(memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>)
  func.func private @add_bf16_vector(memref<16x64xbf16>, memref<16x64xbf16>, memref<16x64xbf16>)
  func.func private @matmul_bf16_bf16(memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>)
  func.func @gemm_0_0_12(%arg0: memref<16x32xbf16>, %arg1: memref<32x64xbf16>, %arg2: memref<16x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_16_64_vector(%arg2) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_0_13(%arg0: memref<16x32xbf16>, %arg1: memref<32x64xbf16>, %arg2: memref<16x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_16_64_vector(%arg2) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_0_14(%arg0: memref<16x32xbf16>, %arg1: memref<32x64xbf16>, %arg2: memref<16x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_16_64_vector(%arg2) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_0_0x3(%arg0: memref<16x32xbf16>, %arg1: memref<32x64xbf16>, %arg2: memref<16x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_16_64_vector(%arg2) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_0_1x3(%arg0: memref<16x32xbf16>, %arg1: memref<32x64xbf16>, %arg2: memref<16x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_16_64_vector(%arg2) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_0_2x3(%arg0: memref<16x32xbf16>, %arg1: memref<32x64xbf16>, %arg2: memref<16x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_16_64_vector(%arg2) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_0_3x3(%arg0: memref<16x32xbf16>, %arg1: memref<32x64xbf16>, %arg2: memref<16x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_16_64_vector(%arg2) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
    return
  }
  func.func @top(%arg0: memref<16x32xbf16>, %arg1: memref<32x960xbf16>, %arg2: memref<16x960xbf16>) attributes {dataflow, itypes = "___"} {
    return
  }
}
