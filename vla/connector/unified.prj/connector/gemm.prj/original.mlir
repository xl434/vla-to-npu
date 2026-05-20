module {
  func.func private @fill_zeros_bf16_32_32_vector(memref<32x32xbf16>)
  func.func private @matmul_scalar_bf16_bf16(memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>)
  func.func private @add_bf16_vector(memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>)
  func.func @"gemm_0_0_0-gemm_1_0_0-gemm_2_0_0-gemm_3_0_0"(%arg0: memref<32x192xbf16>, %arg1: memref<192x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x192xbf16>, %arg5: memref<192x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>, %arg8: !allo.stream<memref<32x32xbf16>, 2>, %arg9: memref<32x192xbf16>, %arg10: memref<192x32xbf16>, %arg11: memref<32x32xbf16>, %arg12: !allo.stream<memref<32x32xbf16>, 2>, %arg13: !allo.stream<memref<32x32xbf16>, 2>, %arg14: memref<32x192xbf16>, %arg15: memref<192x32xbf16>, %arg16: memref<32x32xbf16>, %arg17: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    %alloc_0 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_1 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_5 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_5) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_7 = arith.constant 0.000000e+00 : bf16
    %alloc_8 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_6, %alloc_8 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_9 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_9) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg9, %arg10, %alloc_9) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_10 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_9, %alloc_8, %alloc_10) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_11 = arith.constant 0.000000e+00 : bf16
    %alloc_12 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_10, %alloc_12 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_13 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_13) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg14, %arg15, %alloc_13) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_14 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_13, %alloc_12, %alloc_14) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    memref.copy %alloc_14, %arg16 {to = "local_C"} : memref<32x32xbf16> to memref<32x32xbf16>
    return
  }
  func.func @"gemm_0_0_1-gemm_1_0_1-gemm_2_0_1-gemm_3_0_1"(%arg0: memref<32x192xbf16>, %arg1: memref<192x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x192xbf16>, %arg5: memref<192x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>, %arg8: !allo.stream<memref<32x32xbf16>, 2>, %arg9: memref<32x192xbf16>, %arg10: memref<192x32xbf16>, %arg11: memref<32x32xbf16>, %arg12: !allo.stream<memref<32x32xbf16>, 2>, %arg13: !allo.stream<memref<32x32xbf16>, 2>, %arg14: memref<32x192xbf16>, %arg15: memref<192x32xbf16>, %arg16: memref<32x32xbf16>, %arg17: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    %alloc_0 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_1 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_5 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_5) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_7 = arith.constant 0.000000e+00 : bf16
    %alloc_8 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_6, %alloc_8 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_9 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_9) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg9, %arg10, %alloc_9) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_10 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_9, %alloc_8, %alloc_10) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    %cst_11 = arith.constant 0.000000e+00 : bf16
    %alloc_12 = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    memref.copy %alloc_10, %alloc_12 {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_13 = memref.alloc() : memref<32x32xbf16>
    call @fill_zeros_bf16_32_32_vector(%alloc_13) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg14, %arg15, %alloc_13) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
    %alloc_14 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    call @add_bf16_vector(%alloc_13, %alloc_12, %alloc_14) {lib = "add_bf16_vector"} : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    memref.copy %alloc_14, %arg16 {to = "local_C"} : memref<32x32xbf16> to memref<32x32xbf16>
    return
  }
  func.func @top(%arg0: memref<32x768xbf16>, %arg1: memref<768x64xbf16>, %arg2: memref<32x64xbf16>) attributes {dataflow, itypes = "___"} {
    %0 = allo.stream_construct() {name = "pipe_0_0_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %1 = allo.stream_construct() {name = "pipe_0_0_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %2 = allo.stream_construct() {name = "pipe_1_0_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %3 = allo.stream_construct() {name = "pipe_1_0_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %4 = allo.stream_construct() {name = "pipe_2_0_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %5 = allo.stream_construct() {name = "pipe_2_0_1"} : !allo.stream<memref<32x32xbf16>, 2>
    return
  }
}
