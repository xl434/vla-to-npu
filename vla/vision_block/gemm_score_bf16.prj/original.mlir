module {
  func.func private @fill_zeros_bf16_64_64_vector(memref<64x64xbf16>)
  func.func private @matmul_scalar_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func private @add_bf16_vector(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func @gemm_0_0_0x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    %alloc_0 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_0) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_1 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @gemm_0_0_1x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    %alloc_0 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_0) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_1 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @gemm_0_0_2x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    %alloc_0 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_0) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_1 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @gemm_0_0_3x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    %alloc_0 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_0) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_1 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @gemm_0_1_0x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    %alloc_0 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_0) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_1 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @gemm_0_1_1x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    %alloc_0 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_0) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_1 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @gemm_0_1_2x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    %alloc_0 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_0) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_1 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @gemm_0_1_3x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    %alloc_0 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_0) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_1 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @gemm_0_2_0x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    %alloc_0 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_0) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_1 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @gemm_0_2_1x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    %alloc_0 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_0) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_1 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @gemm_0_2_2x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    %alloc_0 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_0) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_1 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @gemm_0_2_3x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    %alloc_0 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_0) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_1 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @gemm_0_3_0x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    %alloc_0 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_0) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_1 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @gemm_0_3_1x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    %alloc_0 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_0) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_1 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @gemm_0_3_2x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    %alloc_0 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_0) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_1 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @gemm_0_3_3x16(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    %alloc_0 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_0) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_1 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_1) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @top(%arg0: memref<1024x64xbf16>, %arg1: memref<64x1024xbf16>, %arg2: memref<1024x1024xbf16>) attributes {dataflow, itypes = "___"} {
    return
  }
}
