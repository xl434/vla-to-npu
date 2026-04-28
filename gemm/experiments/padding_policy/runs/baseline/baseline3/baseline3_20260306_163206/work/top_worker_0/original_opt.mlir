module {
  func.func private @fill_zeros_bf16_64_64_vector(memref<64x64xbf16>)
  func.func private @matmul_scalar_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func private @add_bf16_vector(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func private @matmul_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func @"gemm_0_0_0-gemm_1_0_0-gemm_2_0_0-gemm_3_0_0-gemm_4_0_0-gemm_5_0_0-gemm_6_0_0-gemm_7_0_0-gemm_8_0_0-gemm_9_0_0-gemm_10_0_0-gemm_11_0_0x8"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg56) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_0_1-gemm_1_0_1-gemm_2_0_1-gemm_3_0_1-gemm_4_0_1-gemm_5_0_1-gemm_6_0_1-gemm_7_0_1-gemm_8_0_1-gemm_9_0_1-gemm_10_0_1-gemm_11_0_1x8"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg56) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_0_2-gemm_1_0_2-gemm_2_0_2-gemm_3_0_2-gemm_4_0_2-gemm_5_0_2-gemm_6_0_2-gemm_7_0_2-gemm_8_0_2-gemm_9_0_2-gemm_10_0_2-gemm_11_0_2x8"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg56) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_0_3-gemm_1_0_3-gemm_2_0_3-gemm_3_0_3-gemm_4_0_3-gemm_5_0_3-gemm_6_0_3-gemm_7_0_3-gemm_8_0_3-gemm_9_0_3-gemm_10_0_3-gemm_11_0_3x8"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg56) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_1_0-gemm_1_1_0-gemm_2_1_0-gemm_3_1_0-gemm_4_1_0-gemm_5_1_0-gemm_6_1_0-gemm_7_1_0-gemm_8_1_0-gemm_9_1_0-gemm_10_1_0-gemm_11_1_0x8"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg56) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_1_1-gemm_1_1_1-gemm_2_1_1-gemm_3_1_1-gemm_4_1_1-gemm_5_1_1-gemm_6_1_1-gemm_7_1_1-gemm_8_1_1-gemm_9_1_1-gemm_10_1_1-gemm_11_1_1x8"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg56) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_1_2-gemm_1_1_2-gemm_2_1_2-gemm_3_1_2-gemm_4_1_2-gemm_5_1_2-gemm_6_1_2-gemm_7_1_2-gemm_8_1_2-gemm_9_1_2-gemm_10_1_2-gemm_11_1_2x8"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg56) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_1_3-gemm_1_1_3-gemm_2_1_3-gemm_3_1_3-gemm_4_1_3-gemm_5_1_3-gemm_6_1_3-gemm_7_1_3-gemm_8_1_3-gemm_9_1_3-gemm_10_1_3-gemm_11_1_3x8"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg56) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_2_0-gemm_1_2_0-gemm_2_2_0-gemm_3_2_0-gemm_4_2_0-gemm_5_2_0-gemm_6_2_0-gemm_7_2_0-gemm_8_2_0-gemm_9_2_0-gemm_10_2_0-gemm_11_2_0x8"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg56) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_2_1-gemm_1_2_1-gemm_2_2_1-gemm_3_2_1-gemm_4_2_1-gemm_5_2_1-gemm_6_2_1-gemm_7_2_1-gemm_8_2_1-gemm_9_2_1-gemm_10_2_1-gemm_11_2_1x8"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg56) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_2_2-gemm_1_2_2-gemm_2_2_2-gemm_3_2_2-gemm_4_2_2-gemm_5_2_2-gemm_6_2_2-gemm_7_2_2-gemm_8_2_2-gemm_9_2_2-gemm_10_2_2-gemm_11_2_2x8"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg56) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_2_3-gemm_1_2_3-gemm_2_2_3-gemm_3_2_3-gemm_4_2_3-gemm_5_2_3-gemm_6_2_3-gemm_7_2_3-gemm_8_2_3-gemm_9_2_3-gemm_10_2_3-gemm_11_2_3x8"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg56) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_3_0-gemm_1_3_0-gemm_2_3_0-gemm_3_3_0-gemm_4_3_0-gemm_5_3_0-gemm_6_3_0-gemm_7_3_0-gemm_8_3_0-gemm_9_3_0-gemm_10_3_0-gemm_11_3_0x8"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg56) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_3_1-gemm_1_3_1-gemm_2_3_1-gemm_3_3_1-gemm_4_3_1-gemm_5_3_1-gemm_6_3_1-gemm_7_3_1-gemm_8_3_1-gemm_9_3_1-gemm_10_3_1-gemm_11_3_1x8"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg56) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_3_2-gemm_1_3_2-gemm_2_3_2-gemm_3_3_2-gemm_4_3_2-gemm_5_3_2-gemm_6_3_2-gemm_7_3_2-gemm_8_3_2-gemm_9_3_2-gemm_10_3_2-gemm_11_3_2x8"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg56) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_3_3-gemm_1_3_3-gemm_2_3_3-gemm_3_3_3-gemm_4_3_3-gemm_5_3_3-gemm_6_3_3-gemm_7_3_3-gemm_8_3_3-gemm_9_3_3-gemm_10_3_3-gemm_11_3_3x8"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg56) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @top(%arg0: memref<1024x768xbf16>, %arg1: memref<768x512xbf16>, %arg2: memref<1024x512xbf16>) attributes {dataflow, itypes = "___"} {
    %0 = allo.stream_construct() {name = "pipe_0_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1 = allo.stream_construct() {name = "pipe_0_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %2 = allo.stream_construct() {name = "pipe_0_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %3 = allo.stream_construct() {name = "pipe_0_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %4 = allo.stream_construct() {name = "pipe_0_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %5 = allo.stream_construct() {name = "pipe_0_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %6 = allo.stream_construct() {name = "pipe_0_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %7 = allo.stream_construct() {name = "pipe_0_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %8 = allo.stream_construct() {name = "pipe_0_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %9 = allo.stream_construct() {name = "pipe_0_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %10 = allo.stream_construct() {name = "pipe_0_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %11 = allo.stream_construct() {name = "pipe_0_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %12 = allo.stream_construct() {name = "pipe_0_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %13 = allo.stream_construct() {name = "pipe_0_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %14 = allo.stream_construct() {name = "pipe_0_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %15 = allo.stream_construct() {name = "pipe_0_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %16 = allo.stream_construct() {name = "pipe_0_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %17 = allo.stream_construct() {name = "pipe_0_2_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %18 = allo.stream_construct() {name = "pipe_0_2_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %19 = allo.stream_construct() {name = "pipe_0_2_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %20 = allo.stream_construct() {name = "pipe_0_2_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %21 = allo.stream_construct() {name = "pipe_0_2_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %22 = allo.stream_construct() {name = "pipe_0_2_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %23 = allo.stream_construct() {name = "pipe_0_2_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %24 = allo.stream_construct() {name = "pipe_0_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %25 = allo.stream_construct() {name = "pipe_0_3_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %26 = allo.stream_construct() {name = "pipe_0_3_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %27 = allo.stream_construct() {name = "pipe_0_3_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %28 = allo.stream_construct() {name = "pipe_0_3_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %29 = allo.stream_construct() {name = "pipe_0_3_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %30 = allo.stream_construct() {name = "pipe_0_3_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %31 = allo.stream_construct() {name = "pipe_0_3_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %32 = allo.stream_construct() {name = "pipe_0_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %33 = allo.stream_construct() {name = "pipe_0_4_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %34 = allo.stream_construct() {name = "pipe_0_4_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %35 = allo.stream_construct() {name = "pipe_0_4_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %36 = allo.stream_construct() {name = "pipe_0_4_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %37 = allo.stream_construct() {name = "pipe_0_4_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %38 = allo.stream_construct() {name = "pipe_0_4_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %39 = allo.stream_construct() {name = "pipe_0_4_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %40 = allo.stream_construct() {name = "pipe_0_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %41 = allo.stream_construct() {name = "pipe_0_5_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %42 = allo.stream_construct() {name = "pipe_0_5_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %43 = allo.stream_construct() {name = "pipe_0_5_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %44 = allo.stream_construct() {name = "pipe_0_5_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %45 = allo.stream_construct() {name = "pipe_0_5_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %46 = allo.stream_construct() {name = "pipe_0_5_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %47 = allo.stream_construct() {name = "pipe_0_5_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %48 = allo.stream_construct() {name = "pipe_0_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %49 = allo.stream_construct() {name = "pipe_0_6_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %50 = allo.stream_construct() {name = "pipe_0_6_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %51 = allo.stream_construct() {name = "pipe_0_6_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %52 = allo.stream_construct() {name = "pipe_0_6_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %53 = allo.stream_construct() {name = "pipe_0_6_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %54 = allo.stream_construct() {name = "pipe_0_6_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %55 = allo.stream_construct() {name = "pipe_0_6_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %56 = allo.stream_construct() {name = "pipe_0_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %57 = allo.stream_construct() {name = "pipe_0_7_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %58 = allo.stream_construct() {name = "pipe_0_7_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %59 = allo.stream_construct() {name = "pipe_0_7_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %60 = allo.stream_construct() {name = "pipe_0_7_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %61 = allo.stream_construct() {name = "pipe_0_7_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %62 = allo.stream_construct() {name = "pipe_0_7_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %63 = allo.stream_construct() {name = "pipe_0_7_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %64 = allo.stream_construct() {name = "pipe_0_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %65 = allo.stream_construct() {name = "pipe_0_8_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %66 = allo.stream_construct() {name = "pipe_0_8_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %67 = allo.stream_construct() {name = "pipe_0_8_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %68 = allo.stream_construct() {name = "pipe_0_8_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %69 = allo.stream_construct() {name = "pipe_0_8_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %70 = allo.stream_construct() {name = "pipe_0_8_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %71 = allo.stream_construct() {name = "pipe_0_8_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %72 = allo.stream_construct() {name = "pipe_0_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %73 = allo.stream_construct() {name = "pipe_0_9_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %74 = allo.stream_construct() {name = "pipe_0_9_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %75 = allo.stream_construct() {name = "pipe_0_9_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %76 = allo.stream_construct() {name = "pipe_0_9_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %77 = allo.stream_construct() {name = "pipe_0_9_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %78 = allo.stream_construct() {name = "pipe_0_9_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %79 = allo.stream_construct() {name = "pipe_0_9_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %80 = allo.stream_construct() {name = "pipe_0_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %81 = allo.stream_construct() {name = "pipe_0_10_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %82 = allo.stream_construct() {name = "pipe_0_10_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %83 = allo.stream_construct() {name = "pipe_0_10_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %84 = allo.stream_construct() {name = "pipe_0_10_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %85 = allo.stream_construct() {name = "pipe_0_10_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %86 = allo.stream_construct() {name = "pipe_0_10_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %87 = allo.stream_construct() {name = "pipe_0_10_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %88 = allo.stream_construct() {name = "pipe_0_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %89 = allo.stream_construct() {name = "pipe_0_11_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %90 = allo.stream_construct() {name = "pipe_0_11_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %91 = allo.stream_construct() {name = "pipe_0_11_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %92 = allo.stream_construct() {name = "pipe_0_11_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %93 = allo.stream_construct() {name = "pipe_0_11_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %94 = allo.stream_construct() {name = "pipe_0_11_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %95 = allo.stream_construct() {name = "pipe_0_11_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %96 = allo.stream_construct() {name = "pipe_0_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %97 = allo.stream_construct() {name = "pipe_0_12_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %98 = allo.stream_construct() {name = "pipe_0_12_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %99 = allo.stream_construct() {name = "pipe_0_12_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %100 = allo.stream_construct() {name = "pipe_0_12_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %101 = allo.stream_construct() {name = "pipe_0_12_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %102 = allo.stream_construct() {name = "pipe_0_12_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %103 = allo.stream_construct() {name = "pipe_0_12_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %104 = allo.stream_construct() {name = "pipe_0_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %105 = allo.stream_construct() {name = "pipe_0_13_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %106 = allo.stream_construct() {name = "pipe_0_13_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %107 = allo.stream_construct() {name = "pipe_0_13_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %108 = allo.stream_construct() {name = "pipe_0_13_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %109 = allo.stream_construct() {name = "pipe_0_13_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %110 = allo.stream_construct() {name = "pipe_0_13_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %111 = allo.stream_construct() {name = "pipe_0_13_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %112 = allo.stream_construct() {name = "pipe_0_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %113 = allo.stream_construct() {name = "pipe_0_14_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %114 = allo.stream_construct() {name = "pipe_0_14_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %115 = allo.stream_construct() {name = "pipe_0_14_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %116 = allo.stream_construct() {name = "pipe_0_14_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %117 = allo.stream_construct() {name = "pipe_0_14_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %118 = allo.stream_construct() {name = "pipe_0_14_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %119 = allo.stream_construct() {name = "pipe_0_14_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %120 = allo.stream_construct() {name = "pipe_0_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %121 = allo.stream_construct() {name = "pipe_0_15_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %122 = allo.stream_construct() {name = "pipe_0_15_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %123 = allo.stream_construct() {name = "pipe_0_15_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %124 = allo.stream_construct() {name = "pipe_0_15_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %125 = allo.stream_construct() {name = "pipe_0_15_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %126 = allo.stream_construct() {name = "pipe_0_15_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %127 = allo.stream_construct() {name = "pipe_0_15_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %128 = allo.stream_construct() {name = "pipe_1_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %129 = allo.stream_construct() {name = "pipe_1_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %130 = allo.stream_construct() {name = "pipe_1_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %131 = allo.stream_construct() {name = "pipe_1_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %132 = allo.stream_construct() {name = "pipe_1_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %133 = allo.stream_construct() {name = "pipe_1_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %134 = allo.stream_construct() {name = "pipe_1_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %135 = allo.stream_construct() {name = "pipe_1_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %136 = allo.stream_construct() {name = "pipe_1_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %137 = allo.stream_construct() {name = "pipe_1_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %138 = allo.stream_construct() {name = "pipe_1_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %139 = allo.stream_construct() {name = "pipe_1_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %140 = allo.stream_construct() {name = "pipe_1_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %141 = allo.stream_construct() {name = "pipe_1_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %142 = allo.stream_construct() {name = "pipe_1_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %143 = allo.stream_construct() {name = "pipe_1_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %144 = allo.stream_construct() {name = "pipe_1_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %145 = allo.stream_construct() {name = "pipe_1_2_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %146 = allo.stream_construct() {name = "pipe_1_2_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %147 = allo.stream_construct() {name = "pipe_1_2_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %148 = allo.stream_construct() {name = "pipe_1_2_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %149 = allo.stream_construct() {name = "pipe_1_2_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %150 = allo.stream_construct() {name = "pipe_1_2_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %151 = allo.stream_construct() {name = "pipe_1_2_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %152 = allo.stream_construct() {name = "pipe_1_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %153 = allo.stream_construct() {name = "pipe_1_3_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %154 = allo.stream_construct() {name = "pipe_1_3_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %155 = allo.stream_construct() {name = "pipe_1_3_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %156 = allo.stream_construct() {name = "pipe_1_3_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %157 = allo.stream_construct() {name = "pipe_1_3_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %158 = allo.stream_construct() {name = "pipe_1_3_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %159 = allo.stream_construct() {name = "pipe_1_3_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %160 = allo.stream_construct() {name = "pipe_1_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %161 = allo.stream_construct() {name = "pipe_1_4_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %162 = allo.stream_construct() {name = "pipe_1_4_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %163 = allo.stream_construct() {name = "pipe_1_4_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %164 = allo.stream_construct() {name = "pipe_1_4_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %165 = allo.stream_construct() {name = "pipe_1_4_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %166 = allo.stream_construct() {name = "pipe_1_4_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %167 = allo.stream_construct() {name = "pipe_1_4_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %168 = allo.stream_construct() {name = "pipe_1_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %169 = allo.stream_construct() {name = "pipe_1_5_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %170 = allo.stream_construct() {name = "pipe_1_5_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %171 = allo.stream_construct() {name = "pipe_1_5_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %172 = allo.stream_construct() {name = "pipe_1_5_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %173 = allo.stream_construct() {name = "pipe_1_5_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %174 = allo.stream_construct() {name = "pipe_1_5_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %175 = allo.stream_construct() {name = "pipe_1_5_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %176 = allo.stream_construct() {name = "pipe_1_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %177 = allo.stream_construct() {name = "pipe_1_6_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %178 = allo.stream_construct() {name = "pipe_1_6_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %179 = allo.stream_construct() {name = "pipe_1_6_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %180 = allo.stream_construct() {name = "pipe_1_6_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %181 = allo.stream_construct() {name = "pipe_1_6_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %182 = allo.stream_construct() {name = "pipe_1_6_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %183 = allo.stream_construct() {name = "pipe_1_6_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %184 = allo.stream_construct() {name = "pipe_1_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %185 = allo.stream_construct() {name = "pipe_1_7_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %186 = allo.stream_construct() {name = "pipe_1_7_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %187 = allo.stream_construct() {name = "pipe_1_7_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %188 = allo.stream_construct() {name = "pipe_1_7_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %189 = allo.stream_construct() {name = "pipe_1_7_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %190 = allo.stream_construct() {name = "pipe_1_7_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %191 = allo.stream_construct() {name = "pipe_1_7_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %192 = allo.stream_construct() {name = "pipe_1_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %193 = allo.stream_construct() {name = "pipe_1_8_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %194 = allo.stream_construct() {name = "pipe_1_8_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %195 = allo.stream_construct() {name = "pipe_1_8_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %196 = allo.stream_construct() {name = "pipe_1_8_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %197 = allo.stream_construct() {name = "pipe_1_8_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %198 = allo.stream_construct() {name = "pipe_1_8_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %199 = allo.stream_construct() {name = "pipe_1_8_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %200 = allo.stream_construct() {name = "pipe_1_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %201 = allo.stream_construct() {name = "pipe_1_9_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %202 = allo.stream_construct() {name = "pipe_1_9_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %203 = allo.stream_construct() {name = "pipe_1_9_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %204 = allo.stream_construct() {name = "pipe_1_9_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %205 = allo.stream_construct() {name = "pipe_1_9_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %206 = allo.stream_construct() {name = "pipe_1_9_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %207 = allo.stream_construct() {name = "pipe_1_9_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %208 = allo.stream_construct() {name = "pipe_1_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %209 = allo.stream_construct() {name = "pipe_1_10_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %210 = allo.stream_construct() {name = "pipe_1_10_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %211 = allo.stream_construct() {name = "pipe_1_10_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %212 = allo.stream_construct() {name = "pipe_1_10_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %213 = allo.stream_construct() {name = "pipe_1_10_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %214 = allo.stream_construct() {name = "pipe_1_10_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %215 = allo.stream_construct() {name = "pipe_1_10_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %216 = allo.stream_construct() {name = "pipe_1_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %217 = allo.stream_construct() {name = "pipe_1_11_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %218 = allo.stream_construct() {name = "pipe_1_11_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %219 = allo.stream_construct() {name = "pipe_1_11_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %220 = allo.stream_construct() {name = "pipe_1_11_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %221 = allo.stream_construct() {name = "pipe_1_11_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %222 = allo.stream_construct() {name = "pipe_1_11_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %223 = allo.stream_construct() {name = "pipe_1_11_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %224 = allo.stream_construct() {name = "pipe_1_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %225 = allo.stream_construct() {name = "pipe_1_12_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %226 = allo.stream_construct() {name = "pipe_1_12_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %227 = allo.stream_construct() {name = "pipe_1_12_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %228 = allo.stream_construct() {name = "pipe_1_12_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %229 = allo.stream_construct() {name = "pipe_1_12_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %230 = allo.stream_construct() {name = "pipe_1_12_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %231 = allo.stream_construct() {name = "pipe_1_12_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %232 = allo.stream_construct() {name = "pipe_1_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %233 = allo.stream_construct() {name = "pipe_1_13_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %234 = allo.stream_construct() {name = "pipe_1_13_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %235 = allo.stream_construct() {name = "pipe_1_13_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %236 = allo.stream_construct() {name = "pipe_1_13_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %237 = allo.stream_construct() {name = "pipe_1_13_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %238 = allo.stream_construct() {name = "pipe_1_13_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %239 = allo.stream_construct() {name = "pipe_1_13_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %240 = allo.stream_construct() {name = "pipe_1_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %241 = allo.stream_construct() {name = "pipe_1_14_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %242 = allo.stream_construct() {name = "pipe_1_14_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %243 = allo.stream_construct() {name = "pipe_1_14_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %244 = allo.stream_construct() {name = "pipe_1_14_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %245 = allo.stream_construct() {name = "pipe_1_14_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %246 = allo.stream_construct() {name = "pipe_1_14_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %247 = allo.stream_construct() {name = "pipe_1_14_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %248 = allo.stream_construct() {name = "pipe_1_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %249 = allo.stream_construct() {name = "pipe_1_15_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %250 = allo.stream_construct() {name = "pipe_1_15_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %251 = allo.stream_construct() {name = "pipe_1_15_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %252 = allo.stream_construct() {name = "pipe_1_15_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %253 = allo.stream_construct() {name = "pipe_1_15_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %254 = allo.stream_construct() {name = "pipe_1_15_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %255 = allo.stream_construct() {name = "pipe_1_15_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %256 = allo.stream_construct() {name = "pipe_2_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %257 = allo.stream_construct() {name = "pipe_2_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %258 = allo.stream_construct() {name = "pipe_2_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %259 = allo.stream_construct() {name = "pipe_2_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %260 = allo.stream_construct() {name = "pipe_2_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %261 = allo.stream_construct() {name = "pipe_2_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %262 = allo.stream_construct() {name = "pipe_2_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %263 = allo.stream_construct() {name = "pipe_2_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %264 = allo.stream_construct() {name = "pipe_2_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %265 = allo.stream_construct() {name = "pipe_2_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %266 = allo.stream_construct() {name = "pipe_2_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %267 = allo.stream_construct() {name = "pipe_2_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %268 = allo.stream_construct() {name = "pipe_2_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %269 = allo.stream_construct() {name = "pipe_2_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %270 = allo.stream_construct() {name = "pipe_2_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %271 = allo.stream_construct() {name = "pipe_2_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %272 = allo.stream_construct() {name = "pipe_2_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %273 = allo.stream_construct() {name = "pipe_2_2_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %274 = allo.stream_construct() {name = "pipe_2_2_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %275 = allo.stream_construct() {name = "pipe_2_2_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %276 = allo.stream_construct() {name = "pipe_2_2_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %277 = allo.stream_construct() {name = "pipe_2_2_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %278 = allo.stream_construct() {name = "pipe_2_2_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %279 = allo.stream_construct() {name = "pipe_2_2_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %280 = allo.stream_construct() {name = "pipe_2_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %281 = allo.stream_construct() {name = "pipe_2_3_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %282 = allo.stream_construct() {name = "pipe_2_3_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %283 = allo.stream_construct() {name = "pipe_2_3_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %284 = allo.stream_construct() {name = "pipe_2_3_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %285 = allo.stream_construct() {name = "pipe_2_3_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %286 = allo.stream_construct() {name = "pipe_2_3_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %287 = allo.stream_construct() {name = "pipe_2_3_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %288 = allo.stream_construct() {name = "pipe_2_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %289 = allo.stream_construct() {name = "pipe_2_4_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %290 = allo.stream_construct() {name = "pipe_2_4_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %291 = allo.stream_construct() {name = "pipe_2_4_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %292 = allo.stream_construct() {name = "pipe_2_4_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %293 = allo.stream_construct() {name = "pipe_2_4_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %294 = allo.stream_construct() {name = "pipe_2_4_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %295 = allo.stream_construct() {name = "pipe_2_4_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %296 = allo.stream_construct() {name = "pipe_2_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %297 = allo.stream_construct() {name = "pipe_2_5_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %298 = allo.stream_construct() {name = "pipe_2_5_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %299 = allo.stream_construct() {name = "pipe_2_5_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %300 = allo.stream_construct() {name = "pipe_2_5_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %301 = allo.stream_construct() {name = "pipe_2_5_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %302 = allo.stream_construct() {name = "pipe_2_5_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %303 = allo.stream_construct() {name = "pipe_2_5_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %304 = allo.stream_construct() {name = "pipe_2_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %305 = allo.stream_construct() {name = "pipe_2_6_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %306 = allo.stream_construct() {name = "pipe_2_6_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %307 = allo.stream_construct() {name = "pipe_2_6_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %308 = allo.stream_construct() {name = "pipe_2_6_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %309 = allo.stream_construct() {name = "pipe_2_6_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %310 = allo.stream_construct() {name = "pipe_2_6_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %311 = allo.stream_construct() {name = "pipe_2_6_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %312 = allo.stream_construct() {name = "pipe_2_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %313 = allo.stream_construct() {name = "pipe_2_7_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %314 = allo.stream_construct() {name = "pipe_2_7_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %315 = allo.stream_construct() {name = "pipe_2_7_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %316 = allo.stream_construct() {name = "pipe_2_7_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %317 = allo.stream_construct() {name = "pipe_2_7_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %318 = allo.stream_construct() {name = "pipe_2_7_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %319 = allo.stream_construct() {name = "pipe_2_7_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %320 = allo.stream_construct() {name = "pipe_2_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %321 = allo.stream_construct() {name = "pipe_2_8_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %322 = allo.stream_construct() {name = "pipe_2_8_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %323 = allo.stream_construct() {name = "pipe_2_8_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %324 = allo.stream_construct() {name = "pipe_2_8_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %325 = allo.stream_construct() {name = "pipe_2_8_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %326 = allo.stream_construct() {name = "pipe_2_8_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %327 = allo.stream_construct() {name = "pipe_2_8_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %328 = allo.stream_construct() {name = "pipe_2_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %329 = allo.stream_construct() {name = "pipe_2_9_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %330 = allo.stream_construct() {name = "pipe_2_9_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %331 = allo.stream_construct() {name = "pipe_2_9_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %332 = allo.stream_construct() {name = "pipe_2_9_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %333 = allo.stream_construct() {name = "pipe_2_9_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %334 = allo.stream_construct() {name = "pipe_2_9_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %335 = allo.stream_construct() {name = "pipe_2_9_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %336 = allo.stream_construct() {name = "pipe_2_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %337 = allo.stream_construct() {name = "pipe_2_10_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %338 = allo.stream_construct() {name = "pipe_2_10_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %339 = allo.stream_construct() {name = "pipe_2_10_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %340 = allo.stream_construct() {name = "pipe_2_10_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %341 = allo.stream_construct() {name = "pipe_2_10_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %342 = allo.stream_construct() {name = "pipe_2_10_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %343 = allo.stream_construct() {name = "pipe_2_10_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %344 = allo.stream_construct() {name = "pipe_2_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %345 = allo.stream_construct() {name = "pipe_2_11_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %346 = allo.stream_construct() {name = "pipe_2_11_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %347 = allo.stream_construct() {name = "pipe_2_11_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %348 = allo.stream_construct() {name = "pipe_2_11_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %349 = allo.stream_construct() {name = "pipe_2_11_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %350 = allo.stream_construct() {name = "pipe_2_11_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %351 = allo.stream_construct() {name = "pipe_2_11_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %352 = allo.stream_construct() {name = "pipe_2_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %353 = allo.stream_construct() {name = "pipe_2_12_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %354 = allo.stream_construct() {name = "pipe_2_12_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %355 = allo.stream_construct() {name = "pipe_2_12_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %356 = allo.stream_construct() {name = "pipe_2_12_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %357 = allo.stream_construct() {name = "pipe_2_12_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %358 = allo.stream_construct() {name = "pipe_2_12_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %359 = allo.stream_construct() {name = "pipe_2_12_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %360 = allo.stream_construct() {name = "pipe_2_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %361 = allo.stream_construct() {name = "pipe_2_13_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %362 = allo.stream_construct() {name = "pipe_2_13_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %363 = allo.stream_construct() {name = "pipe_2_13_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %364 = allo.stream_construct() {name = "pipe_2_13_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %365 = allo.stream_construct() {name = "pipe_2_13_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %366 = allo.stream_construct() {name = "pipe_2_13_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %367 = allo.stream_construct() {name = "pipe_2_13_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %368 = allo.stream_construct() {name = "pipe_2_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %369 = allo.stream_construct() {name = "pipe_2_14_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %370 = allo.stream_construct() {name = "pipe_2_14_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %371 = allo.stream_construct() {name = "pipe_2_14_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %372 = allo.stream_construct() {name = "pipe_2_14_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %373 = allo.stream_construct() {name = "pipe_2_14_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %374 = allo.stream_construct() {name = "pipe_2_14_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %375 = allo.stream_construct() {name = "pipe_2_14_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %376 = allo.stream_construct() {name = "pipe_2_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %377 = allo.stream_construct() {name = "pipe_2_15_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %378 = allo.stream_construct() {name = "pipe_2_15_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %379 = allo.stream_construct() {name = "pipe_2_15_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %380 = allo.stream_construct() {name = "pipe_2_15_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %381 = allo.stream_construct() {name = "pipe_2_15_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %382 = allo.stream_construct() {name = "pipe_2_15_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %383 = allo.stream_construct() {name = "pipe_2_15_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %384 = allo.stream_construct() {name = "pipe_3_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %385 = allo.stream_construct() {name = "pipe_3_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %386 = allo.stream_construct() {name = "pipe_3_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %387 = allo.stream_construct() {name = "pipe_3_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %388 = allo.stream_construct() {name = "pipe_3_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %389 = allo.stream_construct() {name = "pipe_3_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %390 = allo.stream_construct() {name = "pipe_3_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %391 = allo.stream_construct() {name = "pipe_3_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %392 = allo.stream_construct() {name = "pipe_3_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %393 = allo.stream_construct() {name = "pipe_3_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %394 = allo.stream_construct() {name = "pipe_3_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %395 = allo.stream_construct() {name = "pipe_3_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %396 = allo.stream_construct() {name = "pipe_3_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %397 = allo.stream_construct() {name = "pipe_3_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %398 = allo.stream_construct() {name = "pipe_3_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %399 = allo.stream_construct() {name = "pipe_3_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %400 = allo.stream_construct() {name = "pipe_3_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %401 = allo.stream_construct() {name = "pipe_3_2_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %402 = allo.stream_construct() {name = "pipe_3_2_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %403 = allo.stream_construct() {name = "pipe_3_2_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %404 = allo.stream_construct() {name = "pipe_3_2_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %405 = allo.stream_construct() {name = "pipe_3_2_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %406 = allo.stream_construct() {name = "pipe_3_2_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %407 = allo.stream_construct() {name = "pipe_3_2_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %408 = allo.stream_construct() {name = "pipe_3_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %409 = allo.stream_construct() {name = "pipe_3_3_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %410 = allo.stream_construct() {name = "pipe_3_3_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %411 = allo.stream_construct() {name = "pipe_3_3_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %412 = allo.stream_construct() {name = "pipe_3_3_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %413 = allo.stream_construct() {name = "pipe_3_3_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %414 = allo.stream_construct() {name = "pipe_3_3_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %415 = allo.stream_construct() {name = "pipe_3_3_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %416 = allo.stream_construct() {name = "pipe_3_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %417 = allo.stream_construct() {name = "pipe_3_4_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %418 = allo.stream_construct() {name = "pipe_3_4_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %419 = allo.stream_construct() {name = "pipe_3_4_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %420 = allo.stream_construct() {name = "pipe_3_4_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %421 = allo.stream_construct() {name = "pipe_3_4_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %422 = allo.stream_construct() {name = "pipe_3_4_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %423 = allo.stream_construct() {name = "pipe_3_4_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %424 = allo.stream_construct() {name = "pipe_3_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %425 = allo.stream_construct() {name = "pipe_3_5_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %426 = allo.stream_construct() {name = "pipe_3_5_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %427 = allo.stream_construct() {name = "pipe_3_5_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %428 = allo.stream_construct() {name = "pipe_3_5_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %429 = allo.stream_construct() {name = "pipe_3_5_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %430 = allo.stream_construct() {name = "pipe_3_5_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %431 = allo.stream_construct() {name = "pipe_3_5_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %432 = allo.stream_construct() {name = "pipe_3_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %433 = allo.stream_construct() {name = "pipe_3_6_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %434 = allo.stream_construct() {name = "pipe_3_6_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %435 = allo.stream_construct() {name = "pipe_3_6_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %436 = allo.stream_construct() {name = "pipe_3_6_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %437 = allo.stream_construct() {name = "pipe_3_6_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %438 = allo.stream_construct() {name = "pipe_3_6_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %439 = allo.stream_construct() {name = "pipe_3_6_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %440 = allo.stream_construct() {name = "pipe_3_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %441 = allo.stream_construct() {name = "pipe_3_7_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %442 = allo.stream_construct() {name = "pipe_3_7_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %443 = allo.stream_construct() {name = "pipe_3_7_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %444 = allo.stream_construct() {name = "pipe_3_7_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %445 = allo.stream_construct() {name = "pipe_3_7_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %446 = allo.stream_construct() {name = "pipe_3_7_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %447 = allo.stream_construct() {name = "pipe_3_7_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %448 = allo.stream_construct() {name = "pipe_3_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %449 = allo.stream_construct() {name = "pipe_3_8_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %450 = allo.stream_construct() {name = "pipe_3_8_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %451 = allo.stream_construct() {name = "pipe_3_8_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %452 = allo.stream_construct() {name = "pipe_3_8_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %453 = allo.stream_construct() {name = "pipe_3_8_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %454 = allo.stream_construct() {name = "pipe_3_8_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %455 = allo.stream_construct() {name = "pipe_3_8_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %456 = allo.stream_construct() {name = "pipe_3_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %457 = allo.stream_construct() {name = "pipe_3_9_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %458 = allo.stream_construct() {name = "pipe_3_9_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %459 = allo.stream_construct() {name = "pipe_3_9_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %460 = allo.stream_construct() {name = "pipe_3_9_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %461 = allo.stream_construct() {name = "pipe_3_9_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %462 = allo.stream_construct() {name = "pipe_3_9_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %463 = allo.stream_construct() {name = "pipe_3_9_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %464 = allo.stream_construct() {name = "pipe_3_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %465 = allo.stream_construct() {name = "pipe_3_10_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %466 = allo.stream_construct() {name = "pipe_3_10_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %467 = allo.stream_construct() {name = "pipe_3_10_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %468 = allo.stream_construct() {name = "pipe_3_10_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %469 = allo.stream_construct() {name = "pipe_3_10_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %470 = allo.stream_construct() {name = "pipe_3_10_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %471 = allo.stream_construct() {name = "pipe_3_10_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %472 = allo.stream_construct() {name = "pipe_3_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %473 = allo.stream_construct() {name = "pipe_3_11_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %474 = allo.stream_construct() {name = "pipe_3_11_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %475 = allo.stream_construct() {name = "pipe_3_11_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %476 = allo.stream_construct() {name = "pipe_3_11_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %477 = allo.stream_construct() {name = "pipe_3_11_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %478 = allo.stream_construct() {name = "pipe_3_11_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %479 = allo.stream_construct() {name = "pipe_3_11_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %480 = allo.stream_construct() {name = "pipe_3_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %481 = allo.stream_construct() {name = "pipe_3_12_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %482 = allo.stream_construct() {name = "pipe_3_12_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %483 = allo.stream_construct() {name = "pipe_3_12_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %484 = allo.stream_construct() {name = "pipe_3_12_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %485 = allo.stream_construct() {name = "pipe_3_12_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %486 = allo.stream_construct() {name = "pipe_3_12_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %487 = allo.stream_construct() {name = "pipe_3_12_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %488 = allo.stream_construct() {name = "pipe_3_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %489 = allo.stream_construct() {name = "pipe_3_13_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %490 = allo.stream_construct() {name = "pipe_3_13_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %491 = allo.stream_construct() {name = "pipe_3_13_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %492 = allo.stream_construct() {name = "pipe_3_13_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %493 = allo.stream_construct() {name = "pipe_3_13_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %494 = allo.stream_construct() {name = "pipe_3_13_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %495 = allo.stream_construct() {name = "pipe_3_13_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %496 = allo.stream_construct() {name = "pipe_3_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %497 = allo.stream_construct() {name = "pipe_3_14_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %498 = allo.stream_construct() {name = "pipe_3_14_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %499 = allo.stream_construct() {name = "pipe_3_14_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %500 = allo.stream_construct() {name = "pipe_3_14_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %501 = allo.stream_construct() {name = "pipe_3_14_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %502 = allo.stream_construct() {name = "pipe_3_14_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %503 = allo.stream_construct() {name = "pipe_3_14_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %504 = allo.stream_construct() {name = "pipe_3_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %505 = allo.stream_construct() {name = "pipe_3_15_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %506 = allo.stream_construct() {name = "pipe_3_15_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %507 = allo.stream_construct() {name = "pipe_3_15_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %508 = allo.stream_construct() {name = "pipe_3_15_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %509 = allo.stream_construct() {name = "pipe_3_15_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %510 = allo.stream_construct() {name = "pipe_3_15_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %511 = allo.stream_construct() {name = "pipe_3_15_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %512 = allo.stream_construct() {name = "pipe_4_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %513 = allo.stream_construct() {name = "pipe_4_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %514 = allo.stream_construct() {name = "pipe_4_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %515 = allo.stream_construct() {name = "pipe_4_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %516 = allo.stream_construct() {name = "pipe_4_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %517 = allo.stream_construct() {name = "pipe_4_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %518 = allo.stream_construct() {name = "pipe_4_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %519 = allo.stream_construct() {name = "pipe_4_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %520 = allo.stream_construct() {name = "pipe_4_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %521 = allo.stream_construct() {name = "pipe_4_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %522 = allo.stream_construct() {name = "pipe_4_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %523 = allo.stream_construct() {name = "pipe_4_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %524 = allo.stream_construct() {name = "pipe_4_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %525 = allo.stream_construct() {name = "pipe_4_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %526 = allo.stream_construct() {name = "pipe_4_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %527 = allo.stream_construct() {name = "pipe_4_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %528 = allo.stream_construct() {name = "pipe_4_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %529 = allo.stream_construct() {name = "pipe_4_2_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %530 = allo.stream_construct() {name = "pipe_4_2_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %531 = allo.stream_construct() {name = "pipe_4_2_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %532 = allo.stream_construct() {name = "pipe_4_2_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %533 = allo.stream_construct() {name = "pipe_4_2_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %534 = allo.stream_construct() {name = "pipe_4_2_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %535 = allo.stream_construct() {name = "pipe_4_2_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %536 = allo.stream_construct() {name = "pipe_4_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %537 = allo.stream_construct() {name = "pipe_4_3_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %538 = allo.stream_construct() {name = "pipe_4_3_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %539 = allo.stream_construct() {name = "pipe_4_3_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %540 = allo.stream_construct() {name = "pipe_4_3_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %541 = allo.stream_construct() {name = "pipe_4_3_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %542 = allo.stream_construct() {name = "pipe_4_3_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %543 = allo.stream_construct() {name = "pipe_4_3_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %544 = allo.stream_construct() {name = "pipe_4_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %545 = allo.stream_construct() {name = "pipe_4_4_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %546 = allo.stream_construct() {name = "pipe_4_4_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %547 = allo.stream_construct() {name = "pipe_4_4_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %548 = allo.stream_construct() {name = "pipe_4_4_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %549 = allo.stream_construct() {name = "pipe_4_4_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %550 = allo.stream_construct() {name = "pipe_4_4_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %551 = allo.stream_construct() {name = "pipe_4_4_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %552 = allo.stream_construct() {name = "pipe_4_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %553 = allo.stream_construct() {name = "pipe_4_5_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %554 = allo.stream_construct() {name = "pipe_4_5_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %555 = allo.stream_construct() {name = "pipe_4_5_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %556 = allo.stream_construct() {name = "pipe_4_5_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %557 = allo.stream_construct() {name = "pipe_4_5_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %558 = allo.stream_construct() {name = "pipe_4_5_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %559 = allo.stream_construct() {name = "pipe_4_5_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %560 = allo.stream_construct() {name = "pipe_4_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %561 = allo.stream_construct() {name = "pipe_4_6_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %562 = allo.stream_construct() {name = "pipe_4_6_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %563 = allo.stream_construct() {name = "pipe_4_6_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %564 = allo.stream_construct() {name = "pipe_4_6_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %565 = allo.stream_construct() {name = "pipe_4_6_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %566 = allo.stream_construct() {name = "pipe_4_6_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %567 = allo.stream_construct() {name = "pipe_4_6_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %568 = allo.stream_construct() {name = "pipe_4_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %569 = allo.stream_construct() {name = "pipe_4_7_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %570 = allo.stream_construct() {name = "pipe_4_7_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %571 = allo.stream_construct() {name = "pipe_4_7_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %572 = allo.stream_construct() {name = "pipe_4_7_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %573 = allo.stream_construct() {name = "pipe_4_7_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %574 = allo.stream_construct() {name = "pipe_4_7_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %575 = allo.stream_construct() {name = "pipe_4_7_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %576 = allo.stream_construct() {name = "pipe_4_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %577 = allo.stream_construct() {name = "pipe_4_8_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %578 = allo.stream_construct() {name = "pipe_4_8_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %579 = allo.stream_construct() {name = "pipe_4_8_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %580 = allo.stream_construct() {name = "pipe_4_8_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %581 = allo.stream_construct() {name = "pipe_4_8_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %582 = allo.stream_construct() {name = "pipe_4_8_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %583 = allo.stream_construct() {name = "pipe_4_8_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %584 = allo.stream_construct() {name = "pipe_4_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %585 = allo.stream_construct() {name = "pipe_4_9_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %586 = allo.stream_construct() {name = "pipe_4_9_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %587 = allo.stream_construct() {name = "pipe_4_9_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %588 = allo.stream_construct() {name = "pipe_4_9_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %589 = allo.stream_construct() {name = "pipe_4_9_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %590 = allo.stream_construct() {name = "pipe_4_9_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %591 = allo.stream_construct() {name = "pipe_4_9_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %592 = allo.stream_construct() {name = "pipe_4_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %593 = allo.stream_construct() {name = "pipe_4_10_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %594 = allo.stream_construct() {name = "pipe_4_10_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %595 = allo.stream_construct() {name = "pipe_4_10_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %596 = allo.stream_construct() {name = "pipe_4_10_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %597 = allo.stream_construct() {name = "pipe_4_10_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %598 = allo.stream_construct() {name = "pipe_4_10_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %599 = allo.stream_construct() {name = "pipe_4_10_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %600 = allo.stream_construct() {name = "pipe_4_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %601 = allo.stream_construct() {name = "pipe_4_11_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %602 = allo.stream_construct() {name = "pipe_4_11_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %603 = allo.stream_construct() {name = "pipe_4_11_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %604 = allo.stream_construct() {name = "pipe_4_11_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %605 = allo.stream_construct() {name = "pipe_4_11_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %606 = allo.stream_construct() {name = "pipe_4_11_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %607 = allo.stream_construct() {name = "pipe_4_11_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %608 = allo.stream_construct() {name = "pipe_4_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %609 = allo.stream_construct() {name = "pipe_4_12_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %610 = allo.stream_construct() {name = "pipe_4_12_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %611 = allo.stream_construct() {name = "pipe_4_12_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %612 = allo.stream_construct() {name = "pipe_4_12_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %613 = allo.stream_construct() {name = "pipe_4_12_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %614 = allo.stream_construct() {name = "pipe_4_12_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %615 = allo.stream_construct() {name = "pipe_4_12_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %616 = allo.stream_construct() {name = "pipe_4_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %617 = allo.stream_construct() {name = "pipe_4_13_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %618 = allo.stream_construct() {name = "pipe_4_13_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %619 = allo.stream_construct() {name = "pipe_4_13_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %620 = allo.stream_construct() {name = "pipe_4_13_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %621 = allo.stream_construct() {name = "pipe_4_13_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %622 = allo.stream_construct() {name = "pipe_4_13_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %623 = allo.stream_construct() {name = "pipe_4_13_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %624 = allo.stream_construct() {name = "pipe_4_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %625 = allo.stream_construct() {name = "pipe_4_14_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %626 = allo.stream_construct() {name = "pipe_4_14_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %627 = allo.stream_construct() {name = "pipe_4_14_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %628 = allo.stream_construct() {name = "pipe_4_14_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %629 = allo.stream_construct() {name = "pipe_4_14_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %630 = allo.stream_construct() {name = "pipe_4_14_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %631 = allo.stream_construct() {name = "pipe_4_14_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %632 = allo.stream_construct() {name = "pipe_4_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %633 = allo.stream_construct() {name = "pipe_4_15_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %634 = allo.stream_construct() {name = "pipe_4_15_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %635 = allo.stream_construct() {name = "pipe_4_15_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %636 = allo.stream_construct() {name = "pipe_4_15_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %637 = allo.stream_construct() {name = "pipe_4_15_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %638 = allo.stream_construct() {name = "pipe_4_15_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %639 = allo.stream_construct() {name = "pipe_4_15_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %640 = allo.stream_construct() {name = "pipe_5_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %641 = allo.stream_construct() {name = "pipe_5_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %642 = allo.stream_construct() {name = "pipe_5_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %643 = allo.stream_construct() {name = "pipe_5_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %644 = allo.stream_construct() {name = "pipe_5_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %645 = allo.stream_construct() {name = "pipe_5_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %646 = allo.stream_construct() {name = "pipe_5_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %647 = allo.stream_construct() {name = "pipe_5_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %648 = allo.stream_construct() {name = "pipe_5_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %649 = allo.stream_construct() {name = "pipe_5_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %650 = allo.stream_construct() {name = "pipe_5_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %651 = allo.stream_construct() {name = "pipe_5_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %652 = allo.stream_construct() {name = "pipe_5_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %653 = allo.stream_construct() {name = "pipe_5_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %654 = allo.stream_construct() {name = "pipe_5_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %655 = allo.stream_construct() {name = "pipe_5_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %656 = allo.stream_construct() {name = "pipe_5_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %657 = allo.stream_construct() {name = "pipe_5_2_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %658 = allo.stream_construct() {name = "pipe_5_2_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %659 = allo.stream_construct() {name = "pipe_5_2_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %660 = allo.stream_construct() {name = "pipe_5_2_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %661 = allo.stream_construct() {name = "pipe_5_2_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %662 = allo.stream_construct() {name = "pipe_5_2_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %663 = allo.stream_construct() {name = "pipe_5_2_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %664 = allo.stream_construct() {name = "pipe_5_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %665 = allo.stream_construct() {name = "pipe_5_3_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %666 = allo.stream_construct() {name = "pipe_5_3_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %667 = allo.stream_construct() {name = "pipe_5_3_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %668 = allo.stream_construct() {name = "pipe_5_3_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %669 = allo.stream_construct() {name = "pipe_5_3_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %670 = allo.stream_construct() {name = "pipe_5_3_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %671 = allo.stream_construct() {name = "pipe_5_3_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %672 = allo.stream_construct() {name = "pipe_5_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %673 = allo.stream_construct() {name = "pipe_5_4_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %674 = allo.stream_construct() {name = "pipe_5_4_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %675 = allo.stream_construct() {name = "pipe_5_4_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %676 = allo.stream_construct() {name = "pipe_5_4_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %677 = allo.stream_construct() {name = "pipe_5_4_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %678 = allo.stream_construct() {name = "pipe_5_4_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %679 = allo.stream_construct() {name = "pipe_5_4_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %680 = allo.stream_construct() {name = "pipe_5_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %681 = allo.stream_construct() {name = "pipe_5_5_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %682 = allo.stream_construct() {name = "pipe_5_5_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %683 = allo.stream_construct() {name = "pipe_5_5_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %684 = allo.stream_construct() {name = "pipe_5_5_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %685 = allo.stream_construct() {name = "pipe_5_5_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %686 = allo.stream_construct() {name = "pipe_5_5_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %687 = allo.stream_construct() {name = "pipe_5_5_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %688 = allo.stream_construct() {name = "pipe_5_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %689 = allo.stream_construct() {name = "pipe_5_6_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %690 = allo.stream_construct() {name = "pipe_5_6_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %691 = allo.stream_construct() {name = "pipe_5_6_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %692 = allo.stream_construct() {name = "pipe_5_6_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %693 = allo.stream_construct() {name = "pipe_5_6_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %694 = allo.stream_construct() {name = "pipe_5_6_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %695 = allo.stream_construct() {name = "pipe_5_6_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %696 = allo.stream_construct() {name = "pipe_5_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %697 = allo.stream_construct() {name = "pipe_5_7_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %698 = allo.stream_construct() {name = "pipe_5_7_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %699 = allo.stream_construct() {name = "pipe_5_7_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %700 = allo.stream_construct() {name = "pipe_5_7_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %701 = allo.stream_construct() {name = "pipe_5_7_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %702 = allo.stream_construct() {name = "pipe_5_7_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %703 = allo.stream_construct() {name = "pipe_5_7_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %704 = allo.stream_construct() {name = "pipe_5_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %705 = allo.stream_construct() {name = "pipe_5_8_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %706 = allo.stream_construct() {name = "pipe_5_8_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %707 = allo.stream_construct() {name = "pipe_5_8_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %708 = allo.stream_construct() {name = "pipe_5_8_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %709 = allo.stream_construct() {name = "pipe_5_8_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %710 = allo.stream_construct() {name = "pipe_5_8_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %711 = allo.stream_construct() {name = "pipe_5_8_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %712 = allo.stream_construct() {name = "pipe_5_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %713 = allo.stream_construct() {name = "pipe_5_9_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %714 = allo.stream_construct() {name = "pipe_5_9_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %715 = allo.stream_construct() {name = "pipe_5_9_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %716 = allo.stream_construct() {name = "pipe_5_9_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %717 = allo.stream_construct() {name = "pipe_5_9_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %718 = allo.stream_construct() {name = "pipe_5_9_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %719 = allo.stream_construct() {name = "pipe_5_9_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %720 = allo.stream_construct() {name = "pipe_5_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %721 = allo.stream_construct() {name = "pipe_5_10_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %722 = allo.stream_construct() {name = "pipe_5_10_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %723 = allo.stream_construct() {name = "pipe_5_10_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %724 = allo.stream_construct() {name = "pipe_5_10_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %725 = allo.stream_construct() {name = "pipe_5_10_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %726 = allo.stream_construct() {name = "pipe_5_10_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %727 = allo.stream_construct() {name = "pipe_5_10_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %728 = allo.stream_construct() {name = "pipe_5_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %729 = allo.stream_construct() {name = "pipe_5_11_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %730 = allo.stream_construct() {name = "pipe_5_11_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %731 = allo.stream_construct() {name = "pipe_5_11_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %732 = allo.stream_construct() {name = "pipe_5_11_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %733 = allo.stream_construct() {name = "pipe_5_11_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %734 = allo.stream_construct() {name = "pipe_5_11_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %735 = allo.stream_construct() {name = "pipe_5_11_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %736 = allo.stream_construct() {name = "pipe_5_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %737 = allo.stream_construct() {name = "pipe_5_12_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %738 = allo.stream_construct() {name = "pipe_5_12_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %739 = allo.stream_construct() {name = "pipe_5_12_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %740 = allo.stream_construct() {name = "pipe_5_12_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %741 = allo.stream_construct() {name = "pipe_5_12_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %742 = allo.stream_construct() {name = "pipe_5_12_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %743 = allo.stream_construct() {name = "pipe_5_12_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %744 = allo.stream_construct() {name = "pipe_5_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %745 = allo.stream_construct() {name = "pipe_5_13_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %746 = allo.stream_construct() {name = "pipe_5_13_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %747 = allo.stream_construct() {name = "pipe_5_13_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %748 = allo.stream_construct() {name = "pipe_5_13_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %749 = allo.stream_construct() {name = "pipe_5_13_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %750 = allo.stream_construct() {name = "pipe_5_13_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %751 = allo.stream_construct() {name = "pipe_5_13_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %752 = allo.stream_construct() {name = "pipe_5_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %753 = allo.stream_construct() {name = "pipe_5_14_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %754 = allo.stream_construct() {name = "pipe_5_14_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %755 = allo.stream_construct() {name = "pipe_5_14_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %756 = allo.stream_construct() {name = "pipe_5_14_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %757 = allo.stream_construct() {name = "pipe_5_14_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %758 = allo.stream_construct() {name = "pipe_5_14_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %759 = allo.stream_construct() {name = "pipe_5_14_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %760 = allo.stream_construct() {name = "pipe_5_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %761 = allo.stream_construct() {name = "pipe_5_15_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %762 = allo.stream_construct() {name = "pipe_5_15_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %763 = allo.stream_construct() {name = "pipe_5_15_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %764 = allo.stream_construct() {name = "pipe_5_15_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %765 = allo.stream_construct() {name = "pipe_5_15_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %766 = allo.stream_construct() {name = "pipe_5_15_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %767 = allo.stream_construct() {name = "pipe_5_15_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %768 = allo.stream_construct() {name = "pipe_6_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %769 = allo.stream_construct() {name = "pipe_6_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %770 = allo.stream_construct() {name = "pipe_6_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %771 = allo.stream_construct() {name = "pipe_6_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %772 = allo.stream_construct() {name = "pipe_6_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %773 = allo.stream_construct() {name = "pipe_6_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %774 = allo.stream_construct() {name = "pipe_6_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %775 = allo.stream_construct() {name = "pipe_6_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %776 = allo.stream_construct() {name = "pipe_6_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %777 = allo.stream_construct() {name = "pipe_6_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %778 = allo.stream_construct() {name = "pipe_6_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %779 = allo.stream_construct() {name = "pipe_6_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %780 = allo.stream_construct() {name = "pipe_6_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %781 = allo.stream_construct() {name = "pipe_6_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %782 = allo.stream_construct() {name = "pipe_6_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %783 = allo.stream_construct() {name = "pipe_6_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %784 = allo.stream_construct() {name = "pipe_6_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %785 = allo.stream_construct() {name = "pipe_6_2_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %786 = allo.stream_construct() {name = "pipe_6_2_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %787 = allo.stream_construct() {name = "pipe_6_2_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %788 = allo.stream_construct() {name = "pipe_6_2_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %789 = allo.stream_construct() {name = "pipe_6_2_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %790 = allo.stream_construct() {name = "pipe_6_2_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %791 = allo.stream_construct() {name = "pipe_6_2_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %792 = allo.stream_construct() {name = "pipe_6_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %793 = allo.stream_construct() {name = "pipe_6_3_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %794 = allo.stream_construct() {name = "pipe_6_3_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %795 = allo.stream_construct() {name = "pipe_6_3_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %796 = allo.stream_construct() {name = "pipe_6_3_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %797 = allo.stream_construct() {name = "pipe_6_3_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %798 = allo.stream_construct() {name = "pipe_6_3_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %799 = allo.stream_construct() {name = "pipe_6_3_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %800 = allo.stream_construct() {name = "pipe_6_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %801 = allo.stream_construct() {name = "pipe_6_4_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %802 = allo.stream_construct() {name = "pipe_6_4_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %803 = allo.stream_construct() {name = "pipe_6_4_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %804 = allo.stream_construct() {name = "pipe_6_4_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %805 = allo.stream_construct() {name = "pipe_6_4_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %806 = allo.stream_construct() {name = "pipe_6_4_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %807 = allo.stream_construct() {name = "pipe_6_4_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %808 = allo.stream_construct() {name = "pipe_6_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %809 = allo.stream_construct() {name = "pipe_6_5_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %810 = allo.stream_construct() {name = "pipe_6_5_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %811 = allo.stream_construct() {name = "pipe_6_5_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %812 = allo.stream_construct() {name = "pipe_6_5_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %813 = allo.stream_construct() {name = "pipe_6_5_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %814 = allo.stream_construct() {name = "pipe_6_5_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %815 = allo.stream_construct() {name = "pipe_6_5_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %816 = allo.stream_construct() {name = "pipe_6_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %817 = allo.stream_construct() {name = "pipe_6_6_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %818 = allo.stream_construct() {name = "pipe_6_6_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %819 = allo.stream_construct() {name = "pipe_6_6_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %820 = allo.stream_construct() {name = "pipe_6_6_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %821 = allo.stream_construct() {name = "pipe_6_6_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %822 = allo.stream_construct() {name = "pipe_6_6_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %823 = allo.stream_construct() {name = "pipe_6_6_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %824 = allo.stream_construct() {name = "pipe_6_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %825 = allo.stream_construct() {name = "pipe_6_7_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %826 = allo.stream_construct() {name = "pipe_6_7_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %827 = allo.stream_construct() {name = "pipe_6_7_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %828 = allo.stream_construct() {name = "pipe_6_7_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %829 = allo.stream_construct() {name = "pipe_6_7_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %830 = allo.stream_construct() {name = "pipe_6_7_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %831 = allo.stream_construct() {name = "pipe_6_7_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %832 = allo.stream_construct() {name = "pipe_6_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %833 = allo.stream_construct() {name = "pipe_6_8_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %834 = allo.stream_construct() {name = "pipe_6_8_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %835 = allo.stream_construct() {name = "pipe_6_8_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %836 = allo.stream_construct() {name = "pipe_6_8_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %837 = allo.stream_construct() {name = "pipe_6_8_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %838 = allo.stream_construct() {name = "pipe_6_8_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %839 = allo.stream_construct() {name = "pipe_6_8_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %840 = allo.stream_construct() {name = "pipe_6_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %841 = allo.stream_construct() {name = "pipe_6_9_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %842 = allo.stream_construct() {name = "pipe_6_9_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %843 = allo.stream_construct() {name = "pipe_6_9_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %844 = allo.stream_construct() {name = "pipe_6_9_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %845 = allo.stream_construct() {name = "pipe_6_9_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %846 = allo.stream_construct() {name = "pipe_6_9_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %847 = allo.stream_construct() {name = "pipe_6_9_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %848 = allo.stream_construct() {name = "pipe_6_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %849 = allo.stream_construct() {name = "pipe_6_10_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %850 = allo.stream_construct() {name = "pipe_6_10_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %851 = allo.stream_construct() {name = "pipe_6_10_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %852 = allo.stream_construct() {name = "pipe_6_10_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %853 = allo.stream_construct() {name = "pipe_6_10_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %854 = allo.stream_construct() {name = "pipe_6_10_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %855 = allo.stream_construct() {name = "pipe_6_10_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %856 = allo.stream_construct() {name = "pipe_6_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %857 = allo.stream_construct() {name = "pipe_6_11_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %858 = allo.stream_construct() {name = "pipe_6_11_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %859 = allo.stream_construct() {name = "pipe_6_11_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %860 = allo.stream_construct() {name = "pipe_6_11_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %861 = allo.stream_construct() {name = "pipe_6_11_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %862 = allo.stream_construct() {name = "pipe_6_11_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %863 = allo.stream_construct() {name = "pipe_6_11_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %864 = allo.stream_construct() {name = "pipe_6_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %865 = allo.stream_construct() {name = "pipe_6_12_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %866 = allo.stream_construct() {name = "pipe_6_12_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %867 = allo.stream_construct() {name = "pipe_6_12_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %868 = allo.stream_construct() {name = "pipe_6_12_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %869 = allo.stream_construct() {name = "pipe_6_12_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %870 = allo.stream_construct() {name = "pipe_6_12_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %871 = allo.stream_construct() {name = "pipe_6_12_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %872 = allo.stream_construct() {name = "pipe_6_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %873 = allo.stream_construct() {name = "pipe_6_13_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %874 = allo.stream_construct() {name = "pipe_6_13_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %875 = allo.stream_construct() {name = "pipe_6_13_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %876 = allo.stream_construct() {name = "pipe_6_13_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %877 = allo.stream_construct() {name = "pipe_6_13_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %878 = allo.stream_construct() {name = "pipe_6_13_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %879 = allo.stream_construct() {name = "pipe_6_13_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %880 = allo.stream_construct() {name = "pipe_6_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %881 = allo.stream_construct() {name = "pipe_6_14_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %882 = allo.stream_construct() {name = "pipe_6_14_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %883 = allo.stream_construct() {name = "pipe_6_14_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %884 = allo.stream_construct() {name = "pipe_6_14_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %885 = allo.stream_construct() {name = "pipe_6_14_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %886 = allo.stream_construct() {name = "pipe_6_14_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %887 = allo.stream_construct() {name = "pipe_6_14_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %888 = allo.stream_construct() {name = "pipe_6_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %889 = allo.stream_construct() {name = "pipe_6_15_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %890 = allo.stream_construct() {name = "pipe_6_15_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %891 = allo.stream_construct() {name = "pipe_6_15_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %892 = allo.stream_construct() {name = "pipe_6_15_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %893 = allo.stream_construct() {name = "pipe_6_15_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %894 = allo.stream_construct() {name = "pipe_6_15_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %895 = allo.stream_construct() {name = "pipe_6_15_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %896 = allo.stream_construct() {name = "pipe_7_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %897 = allo.stream_construct() {name = "pipe_7_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %898 = allo.stream_construct() {name = "pipe_7_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %899 = allo.stream_construct() {name = "pipe_7_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %900 = allo.stream_construct() {name = "pipe_7_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %901 = allo.stream_construct() {name = "pipe_7_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %902 = allo.stream_construct() {name = "pipe_7_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %903 = allo.stream_construct() {name = "pipe_7_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %904 = allo.stream_construct() {name = "pipe_7_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %905 = allo.stream_construct() {name = "pipe_7_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %906 = allo.stream_construct() {name = "pipe_7_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %907 = allo.stream_construct() {name = "pipe_7_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %908 = allo.stream_construct() {name = "pipe_7_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %909 = allo.stream_construct() {name = "pipe_7_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %910 = allo.stream_construct() {name = "pipe_7_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %911 = allo.stream_construct() {name = "pipe_7_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %912 = allo.stream_construct() {name = "pipe_7_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %913 = allo.stream_construct() {name = "pipe_7_2_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %914 = allo.stream_construct() {name = "pipe_7_2_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %915 = allo.stream_construct() {name = "pipe_7_2_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %916 = allo.stream_construct() {name = "pipe_7_2_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %917 = allo.stream_construct() {name = "pipe_7_2_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %918 = allo.stream_construct() {name = "pipe_7_2_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %919 = allo.stream_construct() {name = "pipe_7_2_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %920 = allo.stream_construct() {name = "pipe_7_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %921 = allo.stream_construct() {name = "pipe_7_3_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %922 = allo.stream_construct() {name = "pipe_7_3_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %923 = allo.stream_construct() {name = "pipe_7_3_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %924 = allo.stream_construct() {name = "pipe_7_3_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %925 = allo.stream_construct() {name = "pipe_7_3_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %926 = allo.stream_construct() {name = "pipe_7_3_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %927 = allo.stream_construct() {name = "pipe_7_3_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %928 = allo.stream_construct() {name = "pipe_7_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %929 = allo.stream_construct() {name = "pipe_7_4_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %930 = allo.stream_construct() {name = "pipe_7_4_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %931 = allo.stream_construct() {name = "pipe_7_4_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %932 = allo.stream_construct() {name = "pipe_7_4_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %933 = allo.stream_construct() {name = "pipe_7_4_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %934 = allo.stream_construct() {name = "pipe_7_4_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %935 = allo.stream_construct() {name = "pipe_7_4_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %936 = allo.stream_construct() {name = "pipe_7_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %937 = allo.stream_construct() {name = "pipe_7_5_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %938 = allo.stream_construct() {name = "pipe_7_5_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %939 = allo.stream_construct() {name = "pipe_7_5_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %940 = allo.stream_construct() {name = "pipe_7_5_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %941 = allo.stream_construct() {name = "pipe_7_5_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %942 = allo.stream_construct() {name = "pipe_7_5_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %943 = allo.stream_construct() {name = "pipe_7_5_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %944 = allo.stream_construct() {name = "pipe_7_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %945 = allo.stream_construct() {name = "pipe_7_6_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %946 = allo.stream_construct() {name = "pipe_7_6_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %947 = allo.stream_construct() {name = "pipe_7_6_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %948 = allo.stream_construct() {name = "pipe_7_6_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %949 = allo.stream_construct() {name = "pipe_7_6_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %950 = allo.stream_construct() {name = "pipe_7_6_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %951 = allo.stream_construct() {name = "pipe_7_6_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %952 = allo.stream_construct() {name = "pipe_7_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %953 = allo.stream_construct() {name = "pipe_7_7_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %954 = allo.stream_construct() {name = "pipe_7_7_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %955 = allo.stream_construct() {name = "pipe_7_7_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %956 = allo.stream_construct() {name = "pipe_7_7_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %957 = allo.stream_construct() {name = "pipe_7_7_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %958 = allo.stream_construct() {name = "pipe_7_7_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %959 = allo.stream_construct() {name = "pipe_7_7_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %960 = allo.stream_construct() {name = "pipe_7_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %961 = allo.stream_construct() {name = "pipe_7_8_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %962 = allo.stream_construct() {name = "pipe_7_8_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %963 = allo.stream_construct() {name = "pipe_7_8_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %964 = allo.stream_construct() {name = "pipe_7_8_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %965 = allo.stream_construct() {name = "pipe_7_8_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %966 = allo.stream_construct() {name = "pipe_7_8_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %967 = allo.stream_construct() {name = "pipe_7_8_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %968 = allo.stream_construct() {name = "pipe_7_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %969 = allo.stream_construct() {name = "pipe_7_9_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %970 = allo.stream_construct() {name = "pipe_7_9_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %971 = allo.stream_construct() {name = "pipe_7_9_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %972 = allo.stream_construct() {name = "pipe_7_9_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %973 = allo.stream_construct() {name = "pipe_7_9_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %974 = allo.stream_construct() {name = "pipe_7_9_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %975 = allo.stream_construct() {name = "pipe_7_9_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %976 = allo.stream_construct() {name = "pipe_7_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %977 = allo.stream_construct() {name = "pipe_7_10_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %978 = allo.stream_construct() {name = "pipe_7_10_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %979 = allo.stream_construct() {name = "pipe_7_10_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %980 = allo.stream_construct() {name = "pipe_7_10_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %981 = allo.stream_construct() {name = "pipe_7_10_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %982 = allo.stream_construct() {name = "pipe_7_10_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %983 = allo.stream_construct() {name = "pipe_7_10_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %984 = allo.stream_construct() {name = "pipe_7_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %985 = allo.stream_construct() {name = "pipe_7_11_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %986 = allo.stream_construct() {name = "pipe_7_11_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %987 = allo.stream_construct() {name = "pipe_7_11_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %988 = allo.stream_construct() {name = "pipe_7_11_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %989 = allo.stream_construct() {name = "pipe_7_11_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %990 = allo.stream_construct() {name = "pipe_7_11_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %991 = allo.stream_construct() {name = "pipe_7_11_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %992 = allo.stream_construct() {name = "pipe_7_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %993 = allo.stream_construct() {name = "pipe_7_12_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %994 = allo.stream_construct() {name = "pipe_7_12_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %995 = allo.stream_construct() {name = "pipe_7_12_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %996 = allo.stream_construct() {name = "pipe_7_12_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %997 = allo.stream_construct() {name = "pipe_7_12_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %998 = allo.stream_construct() {name = "pipe_7_12_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %999 = allo.stream_construct() {name = "pipe_7_12_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1000 = allo.stream_construct() {name = "pipe_7_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1001 = allo.stream_construct() {name = "pipe_7_13_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1002 = allo.stream_construct() {name = "pipe_7_13_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1003 = allo.stream_construct() {name = "pipe_7_13_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1004 = allo.stream_construct() {name = "pipe_7_13_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1005 = allo.stream_construct() {name = "pipe_7_13_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1006 = allo.stream_construct() {name = "pipe_7_13_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1007 = allo.stream_construct() {name = "pipe_7_13_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1008 = allo.stream_construct() {name = "pipe_7_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1009 = allo.stream_construct() {name = "pipe_7_14_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1010 = allo.stream_construct() {name = "pipe_7_14_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1011 = allo.stream_construct() {name = "pipe_7_14_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1012 = allo.stream_construct() {name = "pipe_7_14_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1013 = allo.stream_construct() {name = "pipe_7_14_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1014 = allo.stream_construct() {name = "pipe_7_14_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1015 = allo.stream_construct() {name = "pipe_7_14_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1016 = allo.stream_construct() {name = "pipe_7_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1017 = allo.stream_construct() {name = "pipe_7_15_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1018 = allo.stream_construct() {name = "pipe_7_15_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1019 = allo.stream_construct() {name = "pipe_7_15_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1020 = allo.stream_construct() {name = "pipe_7_15_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1021 = allo.stream_construct() {name = "pipe_7_15_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1022 = allo.stream_construct() {name = "pipe_7_15_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1023 = allo.stream_construct() {name = "pipe_7_15_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1024 = allo.stream_construct() {name = "pipe_8_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1025 = allo.stream_construct() {name = "pipe_8_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1026 = allo.stream_construct() {name = "pipe_8_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1027 = allo.stream_construct() {name = "pipe_8_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1028 = allo.stream_construct() {name = "pipe_8_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1029 = allo.stream_construct() {name = "pipe_8_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1030 = allo.stream_construct() {name = "pipe_8_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1031 = allo.stream_construct() {name = "pipe_8_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1032 = allo.stream_construct() {name = "pipe_8_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1033 = allo.stream_construct() {name = "pipe_8_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1034 = allo.stream_construct() {name = "pipe_8_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1035 = allo.stream_construct() {name = "pipe_8_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1036 = allo.stream_construct() {name = "pipe_8_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1037 = allo.stream_construct() {name = "pipe_8_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1038 = allo.stream_construct() {name = "pipe_8_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1039 = allo.stream_construct() {name = "pipe_8_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1040 = allo.stream_construct() {name = "pipe_8_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1041 = allo.stream_construct() {name = "pipe_8_2_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1042 = allo.stream_construct() {name = "pipe_8_2_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1043 = allo.stream_construct() {name = "pipe_8_2_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1044 = allo.stream_construct() {name = "pipe_8_2_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1045 = allo.stream_construct() {name = "pipe_8_2_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1046 = allo.stream_construct() {name = "pipe_8_2_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1047 = allo.stream_construct() {name = "pipe_8_2_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1048 = allo.stream_construct() {name = "pipe_8_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1049 = allo.stream_construct() {name = "pipe_8_3_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1050 = allo.stream_construct() {name = "pipe_8_3_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1051 = allo.stream_construct() {name = "pipe_8_3_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1052 = allo.stream_construct() {name = "pipe_8_3_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1053 = allo.stream_construct() {name = "pipe_8_3_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1054 = allo.stream_construct() {name = "pipe_8_3_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1055 = allo.stream_construct() {name = "pipe_8_3_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1056 = allo.stream_construct() {name = "pipe_8_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1057 = allo.stream_construct() {name = "pipe_8_4_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1058 = allo.stream_construct() {name = "pipe_8_4_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1059 = allo.stream_construct() {name = "pipe_8_4_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1060 = allo.stream_construct() {name = "pipe_8_4_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1061 = allo.stream_construct() {name = "pipe_8_4_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1062 = allo.stream_construct() {name = "pipe_8_4_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1063 = allo.stream_construct() {name = "pipe_8_4_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1064 = allo.stream_construct() {name = "pipe_8_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1065 = allo.stream_construct() {name = "pipe_8_5_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1066 = allo.stream_construct() {name = "pipe_8_5_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1067 = allo.stream_construct() {name = "pipe_8_5_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1068 = allo.stream_construct() {name = "pipe_8_5_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1069 = allo.stream_construct() {name = "pipe_8_5_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1070 = allo.stream_construct() {name = "pipe_8_5_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1071 = allo.stream_construct() {name = "pipe_8_5_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1072 = allo.stream_construct() {name = "pipe_8_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1073 = allo.stream_construct() {name = "pipe_8_6_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1074 = allo.stream_construct() {name = "pipe_8_6_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1075 = allo.stream_construct() {name = "pipe_8_6_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1076 = allo.stream_construct() {name = "pipe_8_6_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1077 = allo.stream_construct() {name = "pipe_8_6_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1078 = allo.stream_construct() {name = "pipe_8_6_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1079 = allo.stream_construct() {name = "pipe_8_6_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1080 = allo.stream_construct() {name = "pipe_8_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1081 = allo.stream_construct() {name = "pipe_8_7_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1082 = allo.stream_construct() {name = "pipe_8_7_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1083 = allo.stream_construct() {name = "pipe_8_7_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1084 = allo.stream_construct() {name = "pipe_8_7_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1085 = allo.stream_construct() {name = "pipe_8_7_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1086 = allo.stream_construct() {name = "pipe_8_7_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1087 = allo.stream_construct() {name = "pipe_8_7_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1088 = allo.stream_construct() {name = "pipe_8_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1089 = allo.stream_construct() {name = "pipe_8_8_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1090 = allo.stream_construct() {name = "pipe_8_8_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1091 = allo.stream_construct() {name = "pipe_8_8_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1092 = allo.stream_construct() {name = "pipe_8_8_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1093 = allo.stream_construct() {name = "pipe_8_8_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1094 = allo.stream_construct() {name = "pipe_8_8_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1095 = allo.stream_construct() {name = "pipe_8_8_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1096 = allo.stream_construct() {name = "pipe_8_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1097 = allo.stream_construct() {name = "pipe_8_9_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1098 = allo.stream_construct() {name = "pipe_8_9_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1099 = allo.stream_construct() {name = "pipe_8_9_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1100 = allo.stream_construct() {name = "pipe_8_9_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1101 = allo.stream_construct() {name = "pipe_8_9_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1102 = allo.stream_construct() {name = "pipe_8_9_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1103 = allo.stream_construct() {name = "pipe_8_9_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1104 = allo.stream_construct() {name = "pipe_8_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1105 = allo.stream_construct() {name = "pipe_8_10_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1106 = allo.stream_construct() {name = "pipe_8_10_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1107 = allo.stream_construct() {name = "pipe_8_10_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1108 = allo.stream_construct() {name = "pipe_8_10_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1109 = allo.stream_construct() {name = "pipe_8_10_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1110 = allo.stream_construct() {name = "pipe_8_10_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1111 = allo.stream_construct() {name = "pipe_8_10_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1112 = allo.stream_construct() {name = "pipe_8_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1113 = allo.stream_construct() {name = "pipe_8_11_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1114 = allo.stream_construct() {name = "pipe_8_11_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1115 = allo.stream_construct() {name = "pipe_8_11_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1116 = allo.stream_construct() {name = "pipe_8_11_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1117 = allo.stream_construct() {name = "pipe_8_11_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1118 = allo.stream_construct() {name = "pipe_8_11_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1119 = allo.stream_construct() {name = "pipe_8_11_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1120 = allo.stream_construct() {name = "pipe_8_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1121 = allo.stream_construct() {name = "pipe_8_12_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1122 = allo.stream_construct() {name = "pipe_8_12_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1123 = allo.stream_construct() {name = "pipe_8_12_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1124 = allo.stream_construct() {name = "pipe_8_12_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1125 = allo.stream_construct() {name = "pipe_8_12_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1126 = allo.stream_construct() {name = "pipe_8_12_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1127 = allo.stream_construct() {name = "pipe_8_12_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1128 = allo.stream_construct() {name = "pipe_8_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1129 = allo.stream_construct() {name = "pipe_8_13_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1130 = allo.stream_construct() {name = "pipe_8_13_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1131 = allo.stream_construct() {name = "pipe_8_13_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1132 = allo.stream_construct() {name = "pipe_8_13_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1133 = allo.stream_construct() {name = "pipe_8_13_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1134 = allo.stream_construct() {name = "pipe_8_13_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1135 = allo.stream_construct() {name = "pipe_8_13_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1136 = allo.stream_construct() {name = "pipe_8_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1137 = allo.stream_construct() {name = "pipe_8_14_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1138 = allo.stream_construct() {name = "pipe_8_14_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1139 = allo.stream_construct() {name = "pipe_8_14_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1140 = allo.stream_construct() {name = "pipe_8_14_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1141 = allo.stream_construct() {name = "pipe_8_14_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1142 = allo.stream_construct() {name = "pipe_8_14_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1143 = allo.stream_construct() {name = "pipe_8_14_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1144 = allo.stream_construct() {name = "pipe_8_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1145 = allo.stream_construct() {name = "pipe_8_15_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1146 = allo.stream_construct() {name = "pipe_8_15_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1147 = allo.stream_construct() {name = "pipe_8_15_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1148 = allo.stream_construct() {name = "pipe_8_15_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1149 = allo.stream_construct() {name = "pipe_8_15_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1150 = allo.stream_construct() {name = "pipe_8_15_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1151 = allo.stream_construct() {name = "pipe_8_15_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1152 = allo.stream_construct() {name = "pipe_9_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1153 = allo.stream_construct() {name = "pipe_9_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1154 = allo.stream_construct() {name = "pipe_9_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1155 = allo.stream_construct() {name = "pipe_9_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1156 = allo.stream_construct() {name = "pipe_9_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1157 = allo.stream_construct() {name = "pipe_9_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1158 = allo.stream_construct() {name = "pipe_9_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1159 = allo.stream_construct() {name = "pipe_9_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1160 = allo.stream_construct() {name = "pipe_9_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1161 = allo.stream_construct() {name = "pipe_9_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1162 = allo.stream_construct() {name = "pipe_9_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1163 = allo.stream_construct() {name = "pipe_9_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1164 = allo.stream_construct() {name = "pipe_9_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1165 = allo.stream_construct() {name = "pipe_9_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1166 = allo.stream_construct() {name = "pipe_9_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1167 = allo.stream_construct() {name = "pipe_9_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1168 = allo.stream_construct() {name = "pipe_9_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1169 = allo.stream_construct() {name = "pipe_9_2_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1170 = allo.stream_construct() {name = "pipe_9_2_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1171 = allo.stream_construct() {name = "pipe_9_2_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1172 = allo.stream_construct() {name = "pipe_9_2_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1173 = allo.stream_construct() {name = "pipe_9_2_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1174 = allo.stream_construct() {name = "pipe_9_2_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1175 = allo.stream_construct() {name = "pipe_9_2_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1176 = allo.stream_construct() {name = "pipe_9_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1177 = allo.stream_construct() {name = "pipe_9_3_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1178 = allo.stream_construct() {name = "pipe_9_3_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1179 = allo.stream_construct() {name = "pipe_9_3_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1180 = allo.stream_construct() {name = "pipe_9_3_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1181 = allo.stream_construct() {name = "pipe_9_3_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1182 = allo.stream_construct() {name = "pipe_9_3_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1183 = allo.stream_construct() {name = "pipe_9_3_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1184 = allo.stream_construct() {name = "pipe_9_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1185 = allo.stream_construct() {name = "pipe_9_4_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1186 = allo.stream_construct() {name = "pipe_9_4_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1187 = allo.stream_construct() {name = "pipe_9_4_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1188 = allo.stream_construct() {name = "pipe_9_4_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1189 = allo.stream_construct() {name = "pipe_9_4_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1190 = allo.stream_construct() {name = "pipe_9_4_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1191 = allo.stream_construct() {name = "pipe_9_4_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1192 = allo.stream_construct() {name = "pipe_9_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1193 = allo.stream_construct() {name = "pipe_9_5_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1194 = allo.stream_construct() {name = "pipe_9_5_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1195 = allo.stream_construct() {name = "pipe_9_5_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1196 = allo.stream_construct() {name = "pipe_9_5_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1197 = allo.stream_construct() {name = "pipe_9_5_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1198 = allo.stream_construct() {name = "pipe_9_5_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1199 = allo.stream_construct() {name = "pipe_9_5_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1200 = allo.stream_construct() {name = "pipe_9_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1201 = allo.stream_construct() {name = "pipe_9_6_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1202 = allo.stream_construct() {name = "pipe_9_6_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1203 = allo.stream_construct() {name = "pipe_9_6_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1204 = allo.stream_construct() {name = "pipe_9_6_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1205 = allo.stream_construct() {name = "pipe_9_6_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1206 = allo.stream_construct() {name = "pipe_9_6_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1207 = allo.stream_construct() {name = "pipe_9_6_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1208 = allo.stream_construct() {name = "pipe_9_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1209 = allo.stream_construct() {name = "pipe_9_7_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1210 = allo.stream_construct() {name = "pipe_9_7_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1211 = allo.stream_construct() {name = "pipe_9_7_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1212 = allo.stream_construct() {name = "pipe_9_7_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1213 = allo.stream_construct() {name = "pipe_9_7_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1214 = allo.stream_construct() {name = "pipe_9_7_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1215 = allo.stream_construct() {name = "pipe_9_7_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1216 = allo.stream_construct() {name = "pipe_9_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1217 = allo.stream_construct() {name = "pipe_9_8_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1218 = allo.stream_construct() {name = "pipe_9_8_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1219 = allo.stream_construct() {name = "pipe_9_8_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1220 = allo.stream_construct() {name = "pipe_9_8_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1221 = allo.stream_construct() {name = "pipe_9_8_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1222 = allo.stream_construct() {name = "pipe_9_8_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1223 = allo.stream_construct() {name = "pipe_9_8_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1224 = allo.stream_construct() {name = "pipe_9_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1225 = allo.stream_construct() {name = "pipe_9_9_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1226 = allo.stream_construct() {name = "pipe_9_9_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1227 = allo.stream_construct() {name = "pipe_9_9_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1228 = allo.stream_construct() {name = "pipe_9_9_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1229 = allo.stream_construct() {name = "pipe_9_9_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1230 = allo.stream_construct() {name = "pipe_9_9_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1231 = allo.stream_construct() {name = "pipe_9_9_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1232 = allo.stream_construct() {name = "pipe_9_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1233 = allo.stream_construct() {name = "pipe_9_10_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1234 = allo.stream_construct() {name = "pipe_9_10_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1235 = allo.stream_construct() {name = "pipe_9_10_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1236 = allo.stream_construct() {name = "pipe_9_10_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1237 = allo.stream_construct() {name = "pipe_9_10_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1238 = allo.stream_construct() {name = "pipe_9_10_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1239 = allo.stream_construct() {name = "pipe_9_10_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1240 = allo.stream_construct() {name = "pipe_9_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1241 = allo.stream_construct() {name = "pipe_9_11_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1242 = allo.stream_construct() {name = "pipe_9_11_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1243 = allo.stream_construct() {name = "pipe_9_11_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1244 = allo.stream_construct() {name = "pipe_9_11_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1245 = allo.stream_construct() {name = "pipe_9_11_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1246 = allo.stream_construct() {name = "pipe_9_11_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1247 = allo.stream_construct() {name = "pipe_9_11_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1248 = allo.stream_construct() {name = "pipe_9_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1249 = allo.stream_construct() {name = "pipe_9_12_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1250 = allo.stream_construct() {name = "pipe_9_12_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1251 = allo.stream_construct() {name = "pipe_9_12_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1252 = allo.stream_construct() {name = "pipe_9_12_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1253 = allo.stream_construct() {name = "pipe_9_12_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1254 = allo.stream_construct() {name = "pipe_9_12_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1255 = allo.stream_construct() {name = "pipe_9_12_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1256 = allo.stream_construct() {name = "pipe_9_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1257 = allo.stream_construct() {name = "pipe_9_13_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1258 = allo.stream_construct() {name = "pipe_9_13_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1259 = allo.stream_construct() {name = "pipe_9_13_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1260 = allo.stream_construct() {name = "pipe_9_13_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1261 = allo.stream_construct() {name = "pipe_9_13_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1262 = allo.stream_construct() {name = "pipe_9_13_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1263 = allo.stream_construct() {name = "pipe_9_13_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1264 = allo.stream_construct() {name = "pipe_9_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1265 = allo.stream_construct() {name = "pipe_9_14_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1266 = allo.stream_construct() {name = "pipe_9_14_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1267 = allo.stream_construct() {name = "pipe_9_14_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1268 = allo.stream_construct() {name = "pipe_9_14_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1269 = allo.stream_construct() {name = "pipe_9_14_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1270 = allo.stream_construct() {name = "pipe_9_14_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1271 = allo.stream_construct() {name = "pipe_9_14_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1272 = allo.stream_construct() {name = "pipe_9_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1273 = allo.stream_construct() {name = "pipe_9_15_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1274 = allo.stream_construct() {name = "pipe_9_15_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1275 = allo.stream_construct() {name = "pipe_9_15_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1276 = allo.stream_construct() {name = "pipe_9_15_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1277 = allo.stream_construct() {name = "pipe_9_15_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1278 = allo.stream_construct() {name = "pipe_9_15_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1279 = allo.stream_construct() {name = "pipe_9_15_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1280 = allo.stream_construct() {name = "pipe_10_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1281 = allo.stream_construct() {name = "pipe_10_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1282 = allo.stream_construct() {name = "pipe_10_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1283 = allo.stream_construct() {name = "pipe_10_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1284 = allo.stream_construct() {name = "pipe_10_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1285 = allo.stream_construct() {name = "pipe_10_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1286 = allo.stream_construct() {name = "pipe_10_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1287 = allo.stream_construct() {name = "pipe_10_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1288 = allo.stream_construct() {name = "pipe_10_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1289 = allo.stream_construct() {name = "pipe_10_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1290 = allo.stream_construct() {name = "pipe_10_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1291 = allo.stream_construct() {name = "pipe_10_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1292 = allo.stream_construct() {name = "pipe_10_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1293 = allo.stream_construct() {name = "pipe_10_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1294 = allo.stream_construct() {name = "pipe_10_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1295 = allo.stream_construct() {name = "pipe_10_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1296 = allo.stream_construct() {name = "pipe_10_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1297 = allo.stream_construct() {name = "pipe_10_2_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1298 = allo.stream_construct() {name = "pipe_10_2_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1299 = allo.stream_construct() {name = "pipe_10_2_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1300 = allo.stream_construct() {name = "pipe_10_2_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1301 = allo.stream_construct() {name = "pipe_10_2_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1302 = allo.stream_construct() {name = "pipe_10_2_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1303 = allo.stream_construct() {name = "pipe_10_2_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1304 = allo.stream_construct() {name = "pipe_10_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1305 = allo.stream_construct() {name = "pipe_10_3_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1306 = allo.stream_construct() {name = "pipe_10_3_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1307 = allo.stream_construct() {name = "pipe_10_3_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1308 = allo.stream_construct() {name = "pipe_10_3_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1309 = allo.stream_construct() {name = "pipe_10_3_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1310 = allo.stream_construct() {name = "pipe_10_3_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1311 = allo.stream_construct() {name = "pipe_10_3_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1312 = allo.stream_construct() {name = "pipe_10_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1313 = allo.stream_construct() {name = "pipe_10_4_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1314 = allo.stream_construct() {name = "pipe_10_4_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1315 = allo.stream_construct() {name = "pipe_10_4_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1316 = allo.stream_construct() {name = "pipe_10_4_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1317 = allo.stream_construct() {name = "pipe_10_4_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1318 = allo.stream_construct() {name = "pipe_10_4_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1319 = allo.stream_construct() {name = "pipe_10_4_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1320 = allo.stream_construct() {name = "pipe_10_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1321 = allo.stream_construct() {name = "pipe_10_5_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1322 = allo.stream_construct() {name = "pipe_10_5_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1323 = allo.stream_construct() {name = "pipe_10_5_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1324 = allo.stream_construct() {name = "pipe_10_5_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1325 = allo.stream_construct() {name = "pipe_10_5_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1326 = allo.stream_construct() {name = "pipe_10_5_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1327 = allo.stream_construct() {name = "pipe_10_5_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1328 = allo.stream_construct() {name = "pipe_10_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1329 = allo.stream_construct() {name = "pipe_10_6_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1330 = allo.stream_construct() {name = "pipe_10_6_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1331 = allo.stream_construct() {name = "pipe_10_6_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1332 = allo.stream_construct() {name = "pipe_10_6_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1333 = allo.stream_construct() {name = "pipe_10_6_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1334 = allo.stream_construct() {name = "pipe_10_6_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1335 = allo.stream_construct() {name = "pipe_10_6_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1336 = allo.stream_construct() {name = "pipe_10_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1337 = allo.stream_construct() {name = "pipe_10_7_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1338 = allo.stream_construct() {name = "pipe_10_7_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1339 = allo.stream_construct() {name = "pipe_10_7_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1340 = allo.stream_construct() {name = "pipe_10_7_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1341 = allo.stream_construct() {name = "pipe_10_7_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1342 = allo.stream_construct() {name = "pipe_10_7_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1343 = allo.stream_construct() {name = "pipe_10_7_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1344 = allo.stream_construct() {name = "pipe_10_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1345 = allo.stream_construct() {name = "pipe_10_8_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1346 = allo.stream_construct() {name = "pipe_10_8_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1347 = allo.stream_construct() {name = "pipe_10_8_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1348 = allo.stream_construct() {name = "pipe_10_8_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1349 = allo.stream_construct() {name = "pipe_10_8_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1350 = allo.stream_construct() {name = "pipe_10_8_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1351 = allo.stream_construct() {name = "pipe_10_8_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1352 = allo.stream_construct() {name = "pipe_10_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1353 = allo.stream_construct() {name = "pipe_10_9_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1354 = allo.stream_construct() {name = "pipe_10_9_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1355 = allo.stream_construct() {name = "pipe_10_9_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1356 = allo.stream_construct() {name = "pipe_10_9_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1357 = allo.stream_construct() {name = "pipe_10_9_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1358 = allo.stream_construct() {name = "pipe_10_9_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1359 = allo.stream_construct() {name = "pipe_10_9_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1360 = allo.stream_construct() {name = "pipe_10_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1361 = allo.stream_construct() {name = "pipe_10_10_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1362 = allo.stream_construct() {name = "pipe_10_10_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1363 = allo.stream_construct() {name = "pipe_10_10_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1364 = allo.stream_construct() {name = "pipe_10_10_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1365 = allo.stream_construct() {name = "pipe_10_10_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1366 = allo.stream_construct() {name = "pipe_10_10_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1367 = allo.stream_construct() {name = "pipe_10_10_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1368 = allo.stream_construct() {name = "pipe_10_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1369 = allo.stream_construct() {name = "pipe_10_11_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1370 = allo.stream_construct() {name = "pipe_10_11_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1371 = allo.stream_construct() {name = "pipe_10_11_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1372 = allo.stream_construct() {name = "pipe_10_11_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1373 = allo.stream_construct() {name = "pipe_10_11_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1374 = allo.stream_construct() {name = "pipe_10_11_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1375 = allo.stream_construct() {name = "pipe_10_11_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1376 = allo.stream_construct() {name = "pipe_10_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1377 = allo.stream_construct() {name = "pipe_10_12_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1378 = allo.stream_construct() {name = "pipe_10_12_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1379 = allo.stream_construct() {name = "pipe_10_12_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1380 = allo.stream_construct() {name = "pipe_10_12_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1381 = allo.stream_construct() {name = "pipe_10_12_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1382 = allo.stream_construct() {name = "pipe_10_12_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1383 = allo.stream_construct() {name = "pipe_10_12_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1384 = allo.stream_construct() {name = "pipe_10_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1385 = allo.stream_construct() {name = "pipe_10_13_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1386 = allo.stream_construct() {name = "pipe_10_13_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1387 = allo.stream_construct() {name = "pipe_10_13_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1388 = allo.stream_construct() {name = "pipe_10_13_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1389 = allo.stream_construct() {name = "pipe_10_13_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1390 = allo.stream_construct() {name = "pipe_10_13_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1391 = allo.stream_construct() {name = "pipe_10_13_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1392 = allo.stream_construct() {name = "pipe_10_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1393 = allo.stream_construct() {name = "pipe_10_14_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1394 = allo.stream_construct() {name = "pipe_10_14_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1395 = allo.stream_construct() {name = "pipe_10_14_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1396 = allo.stream_construct() {name = "pipe_10_14_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1397 = allo.stream_construct() {name = "pipe_10_14_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1398 = allo.stream_construct() {name = "pipe_10_14_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1399 = allo.stream_construct() {name = "pipe_10_14_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1400 = allo.stream_construct() {name = "pipe_10_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1401 = allo.stream_construct() {name = "pipe_10_15_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1402 = allo.stream_construct() {name = "pipe_10_15_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1403 = allo.stream_construct() {name = "pipe_10_15_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1404 = allo.stream_construct() {name = "pipe_10_15_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1405 = allo.stream_construct() {name = "pipe_10_15_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1406 = allo.stream_construct() {name = "pipe_10_15_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1407 = allo.stream_construct() {name = "pipe_10_15_7"} : !allo.stream<memref<64x64xbf16>, 2>
    return
  }
}
