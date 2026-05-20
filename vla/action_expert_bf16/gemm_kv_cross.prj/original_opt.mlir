module {
  func.func private @fill_zeros_bf16_64_64_vector(memref<64x64xbf16>)
  func.func private @matmul_scalar_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func private @add_bf16_vector(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func private @matmul_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func @"gemm_0_0_0-gemm_1_0_0-gemm_2_0_0-gemm_3_0_0-gemm_4_0_0"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg21) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_0_1-gemm_1_0_1-gemm_2_0_1-gemm_3_0_1-gemm_4_0_1"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg21) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_0_2-gemm_1_0_2-gemm_2_0_2-gemm_3_0_2-gemm_4_0_2"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg21) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_0_3-gemm_1_0_3-gemm_2_0_3-gemm_3_0_3-gemm_4_0_3"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg21) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_0_4-gemm_1_0_4-gemm_2_0_4-gemm_3_0_4-gemm_4_0_4"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg21) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_1_0-gemm_1_1_0-gemm_2_1_0-gemm_3_1_0-gemm_4_1_0"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg21) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_1_1-gemm_1_1_1-gemm_2_1_1-gemm_3_1_1-gemm_4_1_1"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg21) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_1_2-gemm_1_1_2-gemm_2_1_2-gemm_3_1_2-gemm_4_1_2"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg21) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_1_3-gemm_1_1_3-gemm_2_1_3-gemm_3_1_3-gemm_4_1_3"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg21) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_1_4-gemm_1_1_4-gemm_2_1_4-gemm_3_1_4-gemm_4_1_4"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg21) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg21) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
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
