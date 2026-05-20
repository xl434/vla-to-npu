module {
  func.func private @fill_zeros_bf16_32_32_vector(memref<32x32xbf16>)
  func.func private @matmul_scalar_bf16_bf16(memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>)
  func.func private @add_bf16_vector(memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>)
  func.func @"gemm_0_0_0-gemm_1_0_0"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    %alloc_0 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_1 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_5 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_5) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    memref.copy %alloc_6, %arg6 {to = "local_C"} : memref<32x32xbf16> to memref<32x32xbf16>
    return
  }
  func.func @"gemm_0_0_1-gemm_1_0_1"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    %alloc_0 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_1 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_5 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_5) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    memref.copy %alloc_6, %arg6 {to = "local_C"} : memref<32x32xbf16> to memref<32x32xbf16>
    return
  }
  func.func @"gemm_0_0_2-gemm_1_0_2"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    %alloc_0 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_1 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_5 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_5) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    memref.copy %alloc_6, %arg6 {to = "local_C"} : memref<32x32xbf16> to memref<32x32xbf16>
    return
  }
  func.func @"gemm_0_0_3-gemm_1_0_3"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    %alloc_0 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_1 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_5 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_5) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    memref.copy %alloc_6, %arg6 {to = "local_C"} : memref<32x32xbf16> to memref<32x32xbf16>
    return
  }
  func.func @"gemm_0_1_0-gemm_1_1_0"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    %alloc_0 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_1 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_5 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_5) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    memref.copy %alloc_6, %arg6 {to = "local_C"} : memref<32x32xbf16> to memref<32x32xbf16>
    return
  }
  func.func @"gemm_0_1_1-gemm_1_1_1"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    %alloc_0 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_1 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_5 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_5) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    memref.copy %alloc_6, %arg6 {to = "local_C"} : memref<32x32xbf16> to memref<32x32xbf16>
    return
  }
  func.func @"gemm_0_1_2-gemm_1_1_2"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    %alloc_0 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_1 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_5 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_5) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    memref.copy %alloc_6, %arg6 {to = "local_C"} : memref<32x32xbf16> to memref<32x32xbf16>
    return
  }
  func.func @"gemm_0_1_3-gemm_1_1_3"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    %alloc_0 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_1 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_5 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_5) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    memref.copy %alloc_6, %arg6 {to = "local_C"} : memref<32x32xbf16> to memref<32x32xbf16>
    return
  }
  func.func @"gemm_0_2_0-gemm_1_2_0"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    %alloc_0 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_1 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_5 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_5) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    memref.copy %alloc_6, %arg6 {to = "local_C"} : memref<32x32xbf16> to memref<32x32xbf16>
    return
  }
  func.func @"gemm_0_2_1-gemm_1_2_1"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    %alloc_0 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_1 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_5 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_5) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    memref.copy %alloc_6, %arg6 {to = "local_C"} : memref<32x32xbf16> to memref<32x32xbf16>
    return
  }
  func.func @"gemm_0_2_2-gemm_1_2_2"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    %alloc_0 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_1 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_5 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_5) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    memref.copy %alloc_6, %arg6 {to = "local_C"} : memref<32x32xbf16> to memref<32x32xbf16>
    return
  }
  func.func @"gemm_0_2_3-gemm_1_2_3"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    %alloc_0 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_1 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_5 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_5) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    memref.copy %alloc_6, %arg6 {to = "local_C"} : memref<32x32xbf16> to memref<32x32xbf16>
    return
  }
  func.func @"gemm_0_3_0-gemm_1_3_0"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    %alloc_0 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_1 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_5 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_5) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    memref.copy %alloc_6, %arg6 {to = "local_C"} : memref<32x32xbf16> to memref<32x32xbf16>
    return
  }
  func.func @"gemm_0_3_1-gemm_1_3_1"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    %alloc_0 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_1 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_5 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_5) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    memref.copy %alloc_6, %arg6 {to = "local_C"} : memref<32x32xbf16> to memref<32x32xbf16>
    return
  }
  func.func @"gemm_0_3_2-gemm_1_3_2"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    %alloc_0 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_1 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_5 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_5) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    memref.copy %alloc_6, %arg6 {to = "local_C"} : memref<32x32xbf16> to memref<32x32xbf16>
    return
  }
  func.func @"gemm_0_3_3-gemm_1_3_3"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    %alloc_0 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_1 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_5 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_5) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    memref.copy %alloc_6, %arg6 {to = "local_C"} : memref<32x32xbf16> to memref<32x32xbf16>
    return
  }
  func.func @top(%arg0: memref<128x64xbf16>, %arg1: memref<64x128xbf16>, %arg2: memref<128x128xbf16>) attributes {dataflow, itypes = "___"} {
    %0 = allo.stream_construct() {name = "pipe_0_0_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %1 = allo.stream_construct() {name = "pipe_0_0_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %2 = allo.stream_construct() {name = "pipe_0_0_2"} : !allo.stream<memref<32x32xbf16>, 2>
    %3 = allo.stream_construct() {name = "pipe_0_0_3"} : !allo.stream<memref<32x32xbf16>, 2>
    %4 = allo.stream_construct() {name = "pipe_0_1_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %5 = allo.stream_construct() {name = "pipe_0_1_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %6 = allo.stream_construct() {name = "pipe_0_1_2"} : !allo.stream<memref<32x32xbf16>, 2>
    %7 = allo.stream_construct() {name = "pipe_0_1_3"} : !allo.stream<memref<32x32xbf16>, 2>
    %8 = allo.stream_construct() {name = "pipe_0_2_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %9 = allo.stream_construct() {name = "pipe_0_2_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %10 = allo.stream_construct() {name = "pipe_0_2_2"} : !allo.stream<memref<32x32xbf16>, 2>
    %11 = allo.stream_construct() {name = "pipe_0_2_3"} : !allo.stream<memref<32x32xbf16>, 2>
    %12 = allo.stream_construct() {name = "pipe_0_3_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %13 = allo.stream_construct() {name = "pipe_0_3_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %14 = allo.stream_construct() {name = "pipe_0_3_2"} : !allo.stream<memref<32x32xbf16>, 2>
    %15 = allo.stream_construct() {name = "pipe_0_3_3"} : !allo.stream<memref<32x32xbf16>, 2>
    return
  }
}
