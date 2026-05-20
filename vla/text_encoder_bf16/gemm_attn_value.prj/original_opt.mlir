module {
  func.func private @fill_zeros_bf16_32_32_vector(memref<32x32xbf16>)
  func.func private @matmul_scalar_bf16_bf16(memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>)
  func.func private @add_bf16_vector(memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>)
  func.func private @matmul_bf16_bf16(memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>)
  func.func @"gemm_0_0_0-gemm_1_0_0-gemm_2_0_0-gemm_3_0_0"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>, %arg8: !allo.stream<memref<32x32xbf16>, 2>, %arg9: memref<32x32xbf16>, %arg10: memref<32x32xbf16>, %arg11: memref<32x32xbf16>, %arg12: !allo.stream<memref<32x32xbf16>, 2>, %arg13: !allo.stream<memref<32x32xbf16>, 2>, %arg14: memref<32x32xbf16>, %arg15: memref<32x32xbf16>, %arg16: memref<32x32xbf16>, %arg17: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_32_32_vector(%arg16) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    return
  }
  func.func @"gemm_0_0_1-gemm_1_0_1-gemm_2_0_1-gemm_3_0_1"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>, %arg8: !allo.stream<memref<32x32xbf16>, 2>, %arg9: memref<32x32xbf16>, %arg10: memref<32x32xbf16>, %arg11: memref<32x32xbf16>, %arg12: !allo.stream<memref<32x32xbf16>, 2>, %arg13: !allo.stream<memref<32x32xbf16>, 2>, %arg14: memref<32x32xbf16>, %arg15: memref<32x32xbf16>, %arg16: memref<32x32xbf16>, %arg17: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_32_32_vector(%arg16) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    return
  }
  func.func @"gemm_0_1_0-gemm_1_1_0-gemm_2_1_0-gemm_3_1_0"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>, %arg8: !allo.stream<memref<32x32xbf16>, 2>, %arg9: memref<32x32xbf16>, %arg10: memref<32x32xbf16>, %arg11: memref<32x32xbf16>, %arg12: !allo.stream<memref<32x32xbf16>, 2>, %arg13: !allo.stream<memref<32x32xbf16>, 2>, %arg14: memref<32x32xbf16>, %arg15: memref<32x32xbf16>, %arg16: memref<32x32xbf16>, %arg17: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_32_32_vector(%arg16) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    return
  }
  func.func @"gemm_0_1_1-gemm_1_1_1-gemm_2_1_1-gemm_3_1_1"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>, %arg8: !allo.stream<memref<32x32xbf16>, 2>, %arg9: memref<32x32xbf16>, %arg10: memref<32x32xbf16>, %arg11: memref<32x32xbf16>, %arg12: !allo.stream<memref<32x32xbf16>, 2>, %arg13: !allo.stream<memref<32x32xbf16>, 2>, %arg14: memref<32x32xbf16>, %arg15: memref<32x32xbf16>, %arg16: memref<32x32xbf16>, %arg17: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_32_32_vector(%arg16) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    return
  }
  func.func @"gemm_0_2_0-gemm_1_2_0-gemm_2_2_0-gemm_3_2_0"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>, %arg8: !allo.stream<memref<32x32xbf16>, 2>, %arg9: memref<32x32xbf16>, %arg10: memref<32x32xbf16>, %arg11: memref<32x32xbf16>, %arg12: !allo.stream<memref<32x32xbf16>, 2>, %arg13: !allo.stream<memref<32x32xbf16>, 2>, %arg14: memref<32x32xbf16>, %arg15: memref<32x32xbf16>, %arg16: memref<32x32xbf16>, %arg17: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_32_32_vector(%arg16) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    return
  }
  func.func @"gemm_0_2_1-gemm_1_2_1-gemm_2_2_1-gemm_3_2_1"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>, %arg8: !allo.stream<memref<32x32xbf16>, 2>, %arg9: memref<32x32xbf16>, %arg10: memref<32x32xbf16>, %arg11: memref<32x32xbf16>, %arg12: !allo.stream<memref<32x32xbf16>, 2>, %arg13: !allo.stream<memref<32x32xbf16>, 2>, %arg14: memref<32x32xbf16>, %arg15: memref<32x32xbf16>, %arg16: memref<32x32xbf16>, %arg17: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_32_32_vector(%arg16) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    return
  }
  func.func @"gemm_0_3_0-gemm_1_3_0-gemm_2_3_0-gemm_3_3_0"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>, %arg8: !allo.stream<memref<32x32xbf16>, 2>, %arg9: memref<32x32xbf16>, %arg10: memref<32x32xbf16>, %arg11: memref<32x32xbf16>, %arg12: !allo.stream<memref<32x32xbf16>, 2>, %arg13: !allo.stream<memref<32x32xbf16>, 2>, %arg14: memref<32x32xbf16>, %arg15: memref<32x32xbf16>, %arg16: memref<32x32xbf16>, %arg17: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_32_32_vector(%arg16) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    return
  }
  func.func @"gemm_0_3_1-gemm_1_3_1-gemm_2_3_1-gemm_3_3_1"(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>, %arg4: memref<32x32xbf16>, %arg5: memref<32x32xbf16>, %arg6: memref<32x32xbf16>, %arg7: !allo.stream<memref<32x32xbf16>, 2>, %arg8: !allo.stream<memref<32x32xbf16>, 2>, %arg9: memref<32x32xbf16>, %arg10: memref<32x32xbf16>, %arg11: memref<32x32xbf16>, %arg12: !allo.stream<memref<32x32xbf16>, 2>, %arg13: !allo.stream<memref<32x32xbf16>, 2>, %arg14: memref<32x32xbf16>, %arg15: memref<32x32xbf16>, %arg16: memref<32x32xbf16>, %arg17: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_32_32_vector(%arg16) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg16) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
    return
  }
  func.func @top(%arg0: memref<128x128xbf16>, %arg1: memref<128x64xbf16>, %arg2: memref<128x64xbf16>) attributes {dataflow, itypes = "___"} {
    %0 = allo.stream_construct() {name = "pipe_0_0_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %1 = allo.stream_construct() {name = "pipe_0_0_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %2 = allo.stream_construct() {name = "pipe_0_1_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %3 = allo.stream_construct() {name = "pipe_0_1_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %4 = allo.stream_construct() {name = "pipe_0_2_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %5 = allo.stream_construct() {name = "pipe_0_2_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %6 = allo.stream_construct() {name = "pipe_0_3_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %7 = allo.stream_construct() {name = "pipe_0_3_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %8 = allo.stream_construct() {name = "pipe_1_0_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %9 = allo.stream_construct() {name = "pipe_1_0_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %10 = allo.stream_construct() {name = "pipe_1_1_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %11 = allo.stream_construct() {name = "pipe_1_1_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %12 = allo.stream_construct() {name = "pipe_1_2_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %13 = allo.stream_construct() {name = "pipe_1_2_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %14 = allo.stream_construct() {name = "pipe_1_3_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %15 = allo.stream_construct() {name = "pipe_1_3_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %16 = allo.stream_construct() {name = "pipe_2_0_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %17 = allo.stream_construct() {name = "pipe_2_0_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %18 = allo.stream_construct() {name = "pipe_2_1_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %19 = allo.stream_construct() {name = "pipe_2_1_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %20 = allo.stream_construct() {name = "pipe_2_2_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %21 = allo.stream_construct() {name = "pipe_2_2_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %22 = allo.stream_construct() {name = "pipe_2_3_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %23 = allo.stream_construct() {name = "pipe_2_3_1"} : !allo.stream<memref<32x32xbf16>, 2>
    return
  }
}
