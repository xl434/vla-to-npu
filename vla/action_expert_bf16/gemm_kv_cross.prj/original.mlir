module {
  func.func private @fill_zeros_bf16_64_64_vector(memref<64x64xbf16>)
  func.func private @matmul_scalar_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func private @add_bf16_vector(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func @"gemm_0_0_0-gemm_1_0_0-gemm_2_0_0-gemm_3_0_0-gemm_4_0_0"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_5 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_5) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_7 = arith.constant 0.000000e+00 : bf16
    %alloc_8 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_6, %alloc_8 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_9 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_9) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg9, %arg10, %alloc_9) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_10 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_9, %alloc_8, %alloc_10) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_11 = arith.constant 0.000000e+00 : bf16
    %alloc_12 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_10, %alloc_12 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_13 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_13) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg14, %arg15, %alloc_13) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_14 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_13, %alloc_12, %alloc_14) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_15 = arith.constant 0.000000e+00 : bf16
    %alloc_16 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_14, %alloc_16 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_17 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_17) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg19, %arg20, %alloc_17) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_18 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_17, %alloc_16, %alloc_18) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_18, %arg21 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @"gemm_0_0_1-gemm_1_0_1-gemm_2_0_1-gemm_3_0_1-gemm_4_0_1"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_5 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_5) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_7 = arith.constant 0.000000e+00 : bf16
    %alloc_8 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_6, %alloc_8 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_9 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_9) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg9, %arg10, %alloc_9) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_10 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_9, %alloc_8, %alloc_10) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_11 = arith.constant 0.000000e+00 : bf16
    %alloc_12 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_10, %alloc_12 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_13 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_13) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg14, %arg15, %alloc_13) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_14 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_13, %alloc_12, %alloc_14) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_15 = arith.constant 0.000000e+00 : bf16
    %alloc_16 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_14, %alloc_16 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_17 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_17) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg19, %arg20, %alloc_17) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_18 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_17, %alloc_16, %alloc_18) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_18, %arg21 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @"gemm_0_0_2-gemm_1_0_2-gemm_2_0_2-gemm_3_0_2-gemm_4_0_2"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_5 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_5) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_7 = arith.constant 0.000000e+00 : bf16
    %alloc_8 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_6, %alloc_8 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_9 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_9) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg9, %arg10, %alloc_9) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_10 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_9, %alloc_8, %alloc_10) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_11 = arith.constant 0.000000e+00 : bf16
    %alloc_12 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_10, %alloc_12 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_13 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_13) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg14, %arg15, %alloc_13) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_14 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_13, %alloc_12, %alloc_14) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_15 = arith.constant 0.000000e+00 : bf16
    %alloc_16 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_14, %alloc_16 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_17 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_17) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg19, %arg20, %alloc_17) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_18 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_17, %alloc_16, %alloc_18) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_18, %arg21 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @"gemm_0_0_3-gemm_1_0_3-gemm_2_0_3-gemm_3_0_3-gemm_4_0_3"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_5 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_5) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_7 = arith.constant 0.000000e+00 : bf16
    %alloc_8 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_6, %alloc_8 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_9 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_9) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg9, %arg10, %alloc_9) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_10 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_9, %alloc_8, %alloc_10) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_11 = arith.constant 0.000000e+00 : bf16
    %alloc_12 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_10, %alloc_12 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_13 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_13) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg14, %arg15, %alloc_13) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_14 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_13, %alloc_12, %alloc_14) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_15 = arith.constant 0.000000e+00 : bf16
    %alloc_16 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_14, %alloc_16 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_17 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_17) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg19, %arg20, %alloc_17) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_18 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_17, %alloc_16, %alloc_18) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_18, %arg21 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @"gemm_0_0_4-gemm_1_0_4-gemm_2_0_4-gemm_3_0_4-gemm_4_0_4"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_5 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_5) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_7 = arith.constant 0.000000e+00 : bf16
    %alloc_8 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_6, %alloc_8 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_9 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_9) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg9, %arg10, %alloc_9) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_10 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_9, %alloc_8, %alloc_10) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_11 = arith.constant 0.000000e+00 : bf16
    %alloc_12 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_10, %alloc_12 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_13 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_13) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg14, %arg15, %alloc_13) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_14 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_13, %alloc_12, %alloc_14) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_15 = arith.constant 0.000000e+00 : bf16
    %alloc_16 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_14, %alloc_16 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_17 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_17) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg19, %arg20, %alloc_17) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_18 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_17, %alloc_16, %alloc_18) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_18, %arg21 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @"gemm_0_1_0-gemm_1_1_0-gemm_2_1_0-gemm_3_1_0-gemm_4_1_0"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_5 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_5) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_7 = arith.constant 0.000000e+00 : bf16
    %alloc_8 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_6, %alloc_8 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_9 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_9) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg9, %arg10, %alloc_9) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_10 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_9, %alloc_8, %alloc_10) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_11 = arith.constant 0.000000e+00 : bf16
    %alloc_12 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_10, %alloc_12 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_13 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_13) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg14, %arg15, %alloc_13) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_14 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_13, %alloc_12, %alloc_14) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_15 = arith.constant 0.000000e+00 : bf16
    %alloc_16 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_14, %alloc_16 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_17 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_17) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg19, %arg20, %alloc_17) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_18 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_17, %alloc_16, %alloc_18) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_18, %arg21 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @"gemm_0_1_1-gemm_1_1_1-gemm_2_1_1-gemm_3_1_1-gemm_4_1_1"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_5 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_5) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_7 = arith.constant 0.000000e+00 : bf16
    %alloc_8 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_6, %alloc_8 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_9 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_9) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg9, %arg10, %alloc_9) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_10 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_9, %alloc_8, %alloc_10) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_11 = arith.constant 0.000000e+00 : bf16
    %alloc_12 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_10, %alloc_12 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_13 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_13) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg14, %arg15, %alloc_13) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_14 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_13, %alloc_12, %alloc_14) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_15 = arith.constant 0.000000e+00 : bf16
    %alloc_16 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_14, %alloc_16 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_17 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_17) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg19, %arg20, %alloc_17) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_18 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_17, %alloc_16, %alloc_18) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_18, %arg21 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @"gemm_0_1_2-gemm_1_1_2-gemm_2_1_2-gemm_3_1_2-gemm_4_1_2"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_5 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_5) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_7 = arith.constant 0.000000e+00 : bf16
    %alloc_8 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_6, %alloc_8 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_9 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_9) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg9, %arg10, %alloc_9) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_10 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_9, %alloc_8, %alloc_10) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_11 = arith.constant 0.000000e+00 : bf16
    %alloc_12 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_10, %alloc_12 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_13 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_13) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg14, %arg15, %alloc_13) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_14 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_13, %alloc_12, %alloc_14) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_15 = arith.constant 0.000000e+00 : bf16
    %alloc_16 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_14, %alloc_16 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_17 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_17) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg19, %arg20, %alloc_17) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_18 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_17, %alloc_16, %alloc_18) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_18, %arg21 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @"gemm_0_1_3-gemm_1_1_3-gemm_2_1_3-gemm_3_1_3-gemm_4_1_3"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_5 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_5) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_7 = arith.constant 0.000000e+00 : bf16
    %alloc_8 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_6, %alloc_8 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_9 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_9) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg9, %arg10, %alloc_9) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_10 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_9, %alloc_8, %alloc_10) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_11 = arith.constant 0.000000e+00 : bf16
    %alloc_12 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_10, %alloc_12 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_13 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_13) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg14, %arg15, %alloc_13) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_14 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_13, %alloc_12, %alloc_14) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_15 = arith.constant 0.000000e+00 : bf16
    %alloc_16 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_14, %alloc_16 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_17 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_17) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg19, %arg20, %alloc_17) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_18 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_17, %alloc_16, %alloc_18) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_18, %arg21 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @"gemm_0_1_4-gemm_1_1_4-gemm_2_1_4-gemm_3_1_4-gemm_4_1_4"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_5 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_5) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_7 = arith.constant 0.000000e+00 : bf16
    %alloc_8 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_6, %alloc_8 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_9 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_9) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg9, %arg10, %alloc_9) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_10 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_9, %alloc_8, %alloc_10) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_11 = arith.constant 0.000000e+00 : bf16
    %alloc_12 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_10, %alloc_12 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_13 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_13) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg14, %arg15, %alloc_13) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_14 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_13, %alloc_12, %alloc_14) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_15 = arith.constant 0.000000e+00 : bf16
    %alloc_16 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_14, %alloc_16 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_17 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_17) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg19, %arg20, %alloc_17) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_18 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_17, %alloc_16, %alloc_18) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_18, %arg21 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @top(%arg0: memref<128x320xbf16>, %arg1: memref<320x320xbf16>, %arg2: memref<128x320xbf16>) attributes {dataflow, itypes = "___"} {
    %0 = allo.stream_construct() {name = "pipe_0_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1 = allo.stream_construct() {name = "pipe_0_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %2 = allo.stream_construct() {name = "pipe_0_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %3 = allo.stream_construct() {name = "pipe_0_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %4 = allo.stream_construct() {name = "pipe_0_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %5 = allo.stream_construct() {name = "pipe_0_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %6 = allo.stream_construct() {name = "pipe_0_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %7 = allo.stream_construct() {name = "pipe_0_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %8 = allo.stream_construct() {name = "pipe_0_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %9 = allo.stream_construct() {name = "pipe_0_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %10 = allo.stream_construct() {name = "pipe_1_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %11 = allo.stream_construct() {name = "pipe_1_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %12 = allo.stream_construct() {name = "pipe_1_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %13 = allo.stream_construct() {name = "pipe_1_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %14 = allo.stream_construct() {name = "pipe_1_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %15 = allo.stream_construct() {name = "pipe_1_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %16 = allo.stream_construct() {name = "pipe_1_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %17 = allo.stream_construct() {name = "pipe_1_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %18 = allo.stream_construct() {name = "pipe_1_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %19 = allo.stream_construct() {name = "pipe_1_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %20 = allo.stream_construct() {name = "pipe_2_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %21 = allo.stream_construct() {name = "pipe_2_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %22 = allo.stream_construct() {name = "pipe_2_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %23 = allo.stream_construct() {name = "pipe_2_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %24 = allo.stream_construct() {name = "pipe_2_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %25 = allo.stream_construct() {name = "pipe_2_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %26 = allo.stream_construct() {name = "pipe_2_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %27 = allo.stream_construct() {name = "pipe_2_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %28 = allo.stream_construct() {name = "pipe_2_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %29 = allo.stream_construct() {name = "pipe_2_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %30 = allo.stream_construct() {name = "pipe_3_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %31 = allo.stream_construct() {name = "pipe_3_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %32 = allo.stream_construct() {name = "pipe_3_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %33 = allo.stream_construct() {name = "pipe_3_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %34 = allo.stream_construct() {name = "pipe_3_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %35 = allo.stream_construct() {name = "pipe_3_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %36 = allo.stream_construct() {name = "pipe_3_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %37 = allo.stream_construct() {name = "pipe_3_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %38 = allo.stream_construct() {name = "pipe_3_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %39 = allo.stream_construct() {name = "pipe_3_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    return
  }
}
