module {
  func.func private @fill_zeros_bf16_16_64_vector(memref<16x64xbf16>)
  func.func private @matmul_scalar_bf16_bf16(memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>)
  func.func private @add_bf16_vector(memref<16x64xbf16>, memref<16x64xbf16>, memref<16x64xbf16>)
  func.func @gemm_0_0_12(%arg0: memref<16x32xbf16>, %arg1: memref<32x64xbf16>, %arg2: memref<16x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<16x64xbf16>
    %alloc_0 = memref.alloc() : memref<16x64xbf16>
    call @fill_zeros_bf16_16_64_vector(%alloc_0) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<16x64xbf16> to memref<16x64xbf16>
    %alloc_1 = memref.alloc() : memref<16x64xbf16>
    call @fill_zeros_bf16_16_64_vector(%alloc_1) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<16x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<16x64xbf16>, memref<16x64xbf16>, memref<16x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<16x64xbf16> to memref<16x64xbf16>
    return
  }
  func.func @gemm_0_0_13(%arg0: memref<16x32xbf16>, %arg1: memref<32x64xbf16>, %arg2: memref<16x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<16x64xbf16>
    %alloc_0 = memref.alloc() : memref<16x64xbf16>
    call @fill_zeros_bf16_16_64_vector(%alloc_0) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<16x64xbf16> to memref<16x64xbf16>
    %alloc_1 = memref.alloc() : memref<16x64xbf16>
    call @fill_zeros_bf16_16_64_vector(%alloc_1) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<16x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<16x64xbf16>, memref<16x64xbf16>, memref<16x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<16x64xbf16> to memref<16x64xbf16>
    return
  }
  func.func @gemm_0_0_14(%arg0: memref<16x32xbf16>, %arg1: memref<32x64xbf16>, %arg2: memref<16x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<16x64xbf16>
    %alloc_0 = memref.alloc() : memref<16x64xbf16>
    call @fill_zeros_bf16_16_64_vector(%alloc_0) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<16x64xbf16> to memref<16x64xbf16>
    %alloc_1 = memref.alloc() : memref<16x64xbf16>
    call @fill_zeros_bf16_16_64_vector(%alloc_1) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<16x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<16x64xbf16>, memref<16x64xbf16>, memref<16x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<16x64xbf16> to memref<16x64xbf16>
    return
  }
  func.func @gemm_0_0_0x3(%arg0: memref<16x32xbf16>, %arg1: memref<32x64xbf16>, %arg2: memref<16x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<16x64xbf16>
    %alloc_0 = memref.alloc() : memref<16x64xbf16>
    call @fill_zeros_bf16_16_64_vector(%alloc_0) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<16x64xbf16> to memref<16x64xbf16>
    %alloc_1 = memref.alloc() : memref<16x64xbf16>
    call @fill_zeros_bf16_16_64_vector(%alloc_1) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<16x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<16x64xbf16>, memref<16x64xbf16>, memref<16x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<16x64xbf16> to memref<16x64xbf16>
    return
  }
  func.func @gemm_0_0_1x3(%arg0: memref<16x32xbf16>, %arg1: memref<32x64xbf16>, %arg2: memref<16x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<16x64xbf16>
    %alloc_0 = memref.alloc() : memref<16x64xbf16>
    call @fill_zeros_bf16_16_64_vector(%alloc_0) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<16x64xbf16> to memref<16x64xbf16>
    %alloc_1 = memref.alloc() : memref<16x64xbf16>
    call @fill_zeros_bf16_16_64_vector(%alloc_1) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<16x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<16x64xbf16>, memref<16x64xbf16>, memref<16x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<16x64xbf16> to memref<16x64xbf16>
    return
  }
  func.func @gemm_0_0_2x3(%arg0: memref<16x32xbf16>, %arg1: memref<32x64xbf16>, %arg2: memref<16x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<16x64xbf16>
    %alloc_0 = memref.alloc() : memref<16x64xbf16>
    call @fill_zeros_bf16_16_64_vector(%alloc_0) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<16x64xbf16> to memref<16x64xbf16>
    %alloc_1 = memref.alloc() : memref<16x64xbf16>
    call @fill_zeros_bf16_16_64_vector(%alloc_1) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<16x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<16x64xbf16>, memref<16x64xbf16>, memref<16x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<16x64xbf16> to memref<16x64xbf16>
    return
  }
  func.func @gemm_0_0_3x3(%arg0: memref<16x32xbf16>, %arg1: memref<32x64xbf16>, %arg2: memref<16x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<16x64xbf16>
    %alloc_0 = memref.alloc() : memref<16x64xbf16>
    call @fill_zeros_bf16_16_64_vector(%alloc_0) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<16x64xbf16> to memref<16x64xbf16>
    %alloc_1 = memref.alloc() : memref<16x64xbf16>
    call @fill_zeros_bf16_16_64_vector(%alloc_1) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<16x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<16x64xbf16>, memref<16x64xbf16>, memref<16x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<16x64xbf16> to memref<16x64xbf16>
    return
  }
  func.func @top(%arg0: memref<16x32xbf16>, %arg1: memref<32x960xbf16>, %arg2: memref<16x960xbf16>) attributes {dataflow, itypes = "___"} {
    return
  }
}
