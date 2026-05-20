module {
  func.func private @fill_zeros_bf16_64_64_vector(memref<64x64xbf16>)
  func.func private @matmul_scalar_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func private @add_bf16_vector(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func private @matmul_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func @gemm_0_0_0x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg2) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_0_1x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg2) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_0_2x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg2) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_0_3x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg2) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_1_0x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg2) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_1_1x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg2) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_1_2x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg2) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_1_3x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg2) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_2_0x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg2) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_2_1x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg2) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_2_2x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg2) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_2_3x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg2) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_3_0x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg2) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_3_1x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg2) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_3_2x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg2) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @gemm_0_3_3x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg2) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @top(%arg0: memref<1024x64xbf16>, %arg1: memref<64x1024xbf16>, %arg2: memref<1024x1024xbf16>) attributes {dataflow, itypes = "___"} {
    return
  }
}
