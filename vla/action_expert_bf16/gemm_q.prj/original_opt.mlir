module {
  func.func private @fill_zeros_bf16_32_64_vector(memref<32x64xbf16>)
  func.func private @matmul_scalar_bf16_bf16(memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>)
  func.func private @add_bf16_vector(memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>)
  func.func private @matmul_bf16_bf16(memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>)
  func.func @"gemm_0_0_12-gemm_1_0_12-gemm_2_0_12-gemm_3_0_12-gemm_4_0_12-gemm_5_0_12-gemm_6_0_12-gemm_7_0_12-gemm_8_0_12-gemm_9_0_12-gemm_10_0_12-gemm_11_0_12"(%arg0: memref<32x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<32x64xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: memref<32x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<32x64xbf16>, %arg7: !allo.stream<memref<32x64xbf16>, 2>, %arg8: !allo.stream<memref<32x64xbf16>, 2>, %arg9: memref<32x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<32x64xbf16>, %arg12: !allo.stream<memref<32x64xbf16>, 2>, %arg13: !allo.stream<memref<32x64xbf16>, 2>, %arg14: memref<32x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<32x64xbf16>, %arg17: !allo.stream<memref<32x64xbf16>, 2>, %arg18: !allo.stream<memref<32x64xbf16>, 2>, %arg19: memref<32x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<32x64xbf16>, %arg22: !allo.stream<memref<32x64xbf16>, 2>, %arg23: !allo.stream<memref<32x64xbf16>, 2>, %arg24: memref<32x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<32x64xbf16>, %arg27: !allo.stream<memref<32x64xbf16>, 2>, %arg28: !allo.stream<memref<32x64xbf16>, 2>, %arg29: memref<32x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<32x64xbf16>, %arg32: !allo.stream<memref<32x64xbf16>, 2>, %arg33: !allo.stream<memref<32x64xbf16>, 2>, %arg34: memref<32x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<32x64xbf16>, %arg37: !allo.stream<memref<32x64xbf16>, 2>, %arg38: !allo.stream<memref<32x64xbf16>, 2>, %arg39: memref<32x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<32x64xbf16>, %arg42: !allo.stream<memref<32x64xbf16>, 2>, %arg43: !allo.stream<memref<32x64xbf16>, 2>, %arg44: memref<32x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<32x64xbf16>, %arg47: !allo.stream<memref<32x64xbf16>, 2>, %arg48: !allo.stream<memref<32x64xbf16>, 2>, %arg49: memref<32x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<32x64xbf16>, %arg52: !allo.stream<memref<32x64xbf16>, 2>, %arg53: !allo.stream<memref<32x64xbf16>, 2>, %arg54: memref<32x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<32x64xbf16>, %arg57: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_32_64_vector(%arg56) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_0_13-gemm_1_0_13-gemm_2_0_13-gemm_3_0_13-gemm_4_0_13-gemm_5_0_13-gemm_6_0_13-gemm_7_0_13-gemm_8_0_13-gemm_9_0_13-gemm_10_0_13-gemm_11_0_13"(%arg0: memref<32x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<32x64xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: memref<32x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<32x64xbf16>, %arg7: !allo.stream<memref<32x64xbf16>, 2>, %arg8: !allo.stream<memref<32x64xbf16>, 2>, %arg9: memref<32x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<32x64xbf16>, %arg12: !allo.stream<memref<32x64xbf16>, 2>, %arg13: !allo.stream<memref<32x64xbf16>, 2>, %arg14: memref<32x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<32x64xbf16>, %arg17: !allo.stream<memref<32x64xbf16>, 2>, %arg18: !allo.stream<memref<32x64xbf16>, 2>, %arg19: memref<32x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<32x64xbf16>, %arg22: !allo.stream<memref<32x64xbf16>, 2>, %arg23: !allo.stream<memref<32x64xbf16>, 2>, %arg24: memref<32x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<32x64xbf16>, %arg27: !allo.stream<memref<32x64xbf16>, 2>, %arg28: !allo.stream<memref<32x64xbf16>, 2>, %arg29: memref<32x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<32x64xbf16>, %arg32: !allo.stream<memref<32x64xbf16>, 2>, %arg33: !allo.stream<memref<32x64xbf16>, 2>, %arg34: memref<32x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<32x64xbf16>, %arg37: !allo.stream<memref<32x64xbf16>, 2>, %arg38: !allo.stream<memref<32x64xbf16>, 2>, %arg39: memref<32x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<32x64xbf16>, %arg42: !allo.stream<memref<32x64xbf16>, 2>, %arg43: !allo.stream<memref<32x64xbf16>, 2>, %arg44: memref<32x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<32x64xbf16>, %arg47: !allo.stream<memref<32x64xbf16>, 2>, %arg48: !allo.stream<memref<32x64xbf16>, 2>, %arg49: memref<32x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<32x64xbf16>, %arg52: !allo.stream<memref<32x64xbf16>, 2>, %arg53: !allo.stream<memref<32x64xbf16>, 2>, %arg54: memref<32x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<32x64xbf16>, %arg57: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_32_64_vector(%arg56) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_0_14-gemm_1_0_14-gemm_2_0_14-gemm_3_0_14-gemm_4_0_14-gemm_5_0_14-gemm_6_0_14-gemm_7_0_14-gemm_8_0_14-gemm_9_0_14-gemm_10_0_14-gemm_11_0_14"(%arg0: memref<32x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<32x64xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: memref<32x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<32x64xbf16>, %arg7: !allo.stream<memref<32x64xbf16>, 2>, %arg8: !allo.stream<memref<32x64xbf16>, 2>, %arg9: memref<32x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<32x64xbf16>, %arg12: !allo.stream<memref<32x64xbf16>, 2>, %arg13: !allo.stream<memref<32x64xbf16>, 2>, %arg14: memref<32x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<32x64xbf16>, %arg17: !allo.stream<memref<32x64xbf16>, 2>, %arg18: !allo.stream<memref<32x64xbf16>, 2>, %arg19: memref<32x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<32x64xbf16>, %arg22: !allo.stream<memref<32x64xbf16>, 2>, %arg23: !allo.stream<memref<32x64xbf16>, 2>, %arg24: memref<32x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<32x64xbf16>, %arg27: !allo.stream<memref<32x64xbf16>, 2>, %arg28: !allo.stream<memref<32x64xbf16>, 2>, %arg29: memref<32x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<32x64xbf16>, %arg32: !allo.stream<memref<32x64xbf16>, 2>, %arg33: !allo.stream<memref<32x64xbf16>, 2>, %arg34: memref<32x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<32x64xbf16>, %arg37: !allo.stream<memref<32x64xbf16>, 2>, %arg38: !allo.stream<memref<32x64xbf16>, 2>, %arg39: memref<32x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<32x64xbf16>, %arg42: !allo.stream<memref<32x64xbf16>, 2>, %arg43: !allo.stream<memref<32x64xbf16>, 2>, %arg44: memref<32x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<32x64xbf16>, %arg47: !allo.stream<memref<32x64xbf16>, 2>, %arg48: !allo.stream<memref<32x64xbf16>, 2>, %arg49: memref<32x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<32x64xbf16>, %arg52: !allo.stream<memref<32x64xbf16>, 2>, %arg53: !allo.stream<memref<32x64xbf16>, 2>, %arg54: memref<32x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<32x64xbf16>, %arg57: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_32_64_vector(%arg56) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_0_0-gemm_1_0_0-gemm_2_0_0-gemm_3_0_0-gemm_4_0_0-gemm_5_0_0-gemm_6_0_0-gemm_7_0_0-gemm_8_0_0-gemm_9_0_0-gemm_10_0_0-gemm_11_0_0x3"(%arg0: memref<32x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<32x64xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: memref<32x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<32x64xbf16>, %arg7: !allo.stream<memref<32x64xbf16>, 2>, %arg8: !allo.stream<memref<32x64xbf16>, 2>, %arg9: memref<32x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<32x64xbf16>, %arg12: !allo.stream<memref<32x64xbf16>, 2>, %arg13: !allo.stream<memref<32x64xbf16>, 2>, %arg14: memref<32x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<32x64xbf16>, %arg17: !allo.stream<memref<32x64xbf16>, 2>, %arg18: !allo.stream<memref<32x64xbf16>, 2>, %arg19: memref<32x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<32x64xbf16>, %arg22: !allo.stream<memref<32x64xbf16>, 2>, %arg23: !allo.stream<memref<32x64xbf16>, 2>, %arg24: memref<32x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<32x64xbf16>, %arg27: !allo.stream<memref<32x64xbf16>, 2>, %arg28: !allo.stream<memref<32x64xbf16>, 2>, %arg29: memref<32x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<32x64xbf16>, %arg32: !allo.stream<memref<32x64xbf16>, 2>, %arg33: !allo.stream<memref<32x64xbf16>, 2>, %arg34: memref<32x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<32x64xbf16>, %arg37: !allo.stream<memref<32x64xbf16>, 2>, %arg38: !allo.stream<memref<32x64xbf16>, 2>, %arg39: memref<32x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<32x64xbf16>, %arg42: !allo.stream<memref<32x64xbf16>, 2>, %arg43: !allo.stream<memref<32x64xbf16>, 2>, %arg44: memref<32x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<32x64xbf16>, %arg47: !allo.stream<memref<32x64xbf16>, 2>, %arg48: !allo.stream<memref<32x64xbf16>, 2>, %arg49: memref<32x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<32x64xbf16>, %arg52: !allo.stream<memref<32x64xbf16>, 2>, %arg53: !allo.stream<memref<32x64xbf16>, 2>, %arg54: memref<32x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<32x64xbf16>, %arg57: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_32_64_vector(%arg56) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_0_1-gemm_1_0_1-gemm_2_0_1-gemm_3_0_1-gemm_4_0_1-gemm_5_0_1-gemm_6_0_1-gemm_7_0_1-gemm_8_0_1-gemm_9_0_1-gemm_10_0_1-gemm_11_0_1x3"(%arg0: memref<32x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<32x64xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: memref<32x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<32x64xbf16>, %arg7: !allo.stream<memref<32x64xbf16>, 2>, %arg8: !allo.stream<memref<32x64xbf16>, 2>, %arg9: memref<32x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<32x64xbf16>, %arg12: !allo.stream<memref<32x64xbf16>, 2>, %arg13: !allo.stream<memref<32x64xbf16>, 2>, %arg14: memref<32x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<32x64xbf16>, %arg17: !allo.stream<memref<32x64xbf16>, 2>, %arg18: !allo.stream<memref<32x64xbf16>, 2>, %arg19: memref<32x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<32x64xbf16>, %arg22: !allo.stream<memref<32x64xbf16>, 2>, %arg23: !allo.stream<memref<32x64xbf16>, 2>, %arg24: memref<32x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<32x64xbf16>, %arg27: !allo.stream<memref<32x64xbf16>, 2>, %arg28: !allo.stream<memref<32x64xbf16>, 2>, %arg29: memref<32x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<32x64xbf16>, %arg32: !allo.stream<memref<32x64xbf16>, 2>, %arg33: !allo.stream<memref<32x64xbf16>, 2>, %arg34: memref<32x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<32x64xbf16>, %arg37: !allo.stream<memref<32x64xbf16>, 2>, %arg38: !allo.stream<memref<32x64xbf16>, 2>, %arg39: memref<32x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<32x64xbf16>, %arg42: !allo.stream<memref<32x64xbf16>, 2>, %arg43: !allo.stream<memref<32x64xbf16>, 2>, %arg44: memref<32x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<32x64xbf16>, %arg47: !allo.stream<memref<32x64xbf16>, 2>, %arg48: !allo.stream<memref<32x64xbf16>, 2>, %arg49: memref<32x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<32x64xbf16>, %arg52: !allo.stream<memref<32x64xbf16>, 2>, %arg53: !allo.stream<memref<32x64xbf16>, 2>, %arg54: memref<32x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<32x64xbf16>, %arg57: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_32_64_vector(%arg56) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_0_2-gemm_1_0_2-gemm_2_0_2-gemm_3_0_2-gemm_4_0_2-gemm_5_0_2-gemm_6_0_2-gemm_7_0_2-gemm_8_0_2-gemm_9_0_2-gemm_10_0_2-gemm_11_0_2x3"(%arg0: memref<32x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<32x64xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: memref<32x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<32x64xbf16>, %arg7: !allo.stream<memref<32x64xbf16>, 2>, %arg8: !allo.stream<memref<32x64xbf16>, 2>, %arg9: memref<32x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<32x64xbf16>, %arg12: !allo.stream<memref<32x64xbf16>, 2>, %arg13: !allo.stream<memref<32x64xbf16>, 2>, %arg14: memref<32x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<32x64xbf16>, %arg17: !allo.stream<memref<32x64xbf16>, 2>, %arg18: !allo.stream<memref<32x64xbf16>, 2>, %arg19: memref<32x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<32x64xbf16>, %arg22: !allo.stream<memref<32x64xbf16>, 2>, %arg23: !allo.stream<memref<32x64xbf16>, 2>, %arg24: memref<32x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<32x64xbf16>, %arg27: !allo.stream<memref<32x64xbf16>, 2>, %arg28: !allo.stream<memref<32x64xbf16>, 2>, %arg29: memref<32x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<32x64xbf16>, %arg32: !allo.stream<memref<32x64xbf16>, 2>, %arg33: !allo.stream<memref<32x64xbf16>, 2>, %arg34: memref<32x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<32x64xbf16>, %arg37: !allo.stream<memref<32x64xbf16>, 2>, %arg38: !allo.stream<memref<32x64xbf16>, 2>, %arg39: memref<32x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<32x64xbf16>, %arg42: !allo.stream<memref<32x64xbf16>, 2>, %arg43: !allo.stream<memref<32x64xbf16>, 2>, %arg44: memref<32x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<32x64xbf16>, %arg47: !allo.stream<memref<32x64xbf16>, 2>, %arg48: !allo.stream<memref<32x64xbf16>, 2>, %arg49: memref<32x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<32x64xbf16>, %arg52: !allo.stream<memref<32x64xbf16>, 2>, %arg53: !allo.stream<memref<32x64xbf16>, 2>, %arg54: memref<32x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<32x64xbf16>, %arg57: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_32_64_vector(%arg56) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_0_3-gemm_1_0_3-gemm_2_0_3-gemm_3_0_3-gemm_4_0_3-gemm_5_0_3-gemm_6_0_3-gemm_7_0_3-gemm_8_0_3-gemm_9_0_3-gemm_10_0_3-gemm_11_0_3x3"(%arg0: memref<32x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<32x64xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: memref<32x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<32x64xbf16>, %arg7: !allo.stream<memref<32x64xbf16>, 2>, %arg8: !allo.stream<memref<32x64xbf16>, 2>, %arg9: memref<32x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<32x64xbf16>, %arg12: !allo.stream<memref<32x64xbf16>, 2>, %arg13: !allo.stream<memref<32x64xbf16>, 2>, %arg14: memref<32x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<32x64xbf16>, %arg17: !allo.stream<memref<32x64xbf16>, 2>, %arg18: !allo.stream<memref<32x64xbf16>, 2>, %arg19: memref<32x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<32x64xbf16>, %arg22: !allo.stream<memref<32x64xbf16>, 2>, %arg23: !allo.stream<memref<32x64xbf16>, 2>, %arg24: memref<32x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<32x64xbf16>, %arg27: !allo.stream<memref<32x64xbf16>, 2>, %arg28: !allo.stream<memref<32x64xbf16>, 2>, %arg29: memref<32x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<32x64xbf16>, %arg32: !allo.stream<memref<32x64xbf16>, 2>, %arg33: !allo.stream<memref<32x64xbf16>, 2>, %arg34: memref<32x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<32x64xbf16>, %arg37: !allo.stream<memref<32x64xbf16>, 2>, %arg38: !allo.stream<memref<32x64xbf16>, 2>, %arg39: memref<32x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<32x64xbf16>, %arg42: !allo.stream<memref<32x64xbf16>, 2>, %arg43: !allo.stream<memref<32x64xbf16>, 2>, %arg44: memref<32x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<32x64xbf16>, %arg47: !allo.stream<memref<32x64xbf16>, 2>, %arg48: !allo.stream<memref<32x64xbf16>, 2>, %arg49: memref<32x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<32x64xbf16>, %arg52: !allo.stream<memref<32x64xbf16>, 2>, %arg53: !allo.stream<memref<32x64xbf16>, 2>, %arg54: memref<32x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<32x64xbf16>, %arg57: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_32_64_vector(%arg56) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg56) : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    return
  }
  func.func @top(%arg0: memref<32x768xbf16>, %arg1: memref<768x960xbf16>, %arg2: memref<32x960xbf16>) attributes {dataflow, itypes = "___"} {
    %0 = allo.stream_construct() {name = "pipe_0_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %1 = allo.stream_construct() {name = "pipe_0_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %2 = allo.stream_construct() {name = "pipe_0_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %3 = allo.stream_construct() {name = "pipe_0_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %4 = allo.stream_construct() {name = "pipe_0_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %5 = allo.stream_construct() {name = "pipe_0_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %6 = allo.stream_construct() {name = "pipe_0_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %7 = allo.stream_construct() {name = "pipe_0_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %8 = allo.stream_construct() {name = "pipe_0_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %9 = allo.stream_construct() {name = "pipe_0_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %10 = allo.stream_construct() {name = "pipe_0_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %11 = allo.stream_construct() {name = "pipe_0_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %12 = allo.stream_construct() {name = "pipe_0_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %13 = allo.stream_construct() {name = "pipe_0_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %14 = allo.stream_construct() {name = "pipe_0_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    %15 = allo.stream_construct() {name = "pipe_1_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %16 = allo.stream_construct() {name = "pipe_1_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %17 = allo.stream_construct() {name = "pipe_1_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %18 = allo.stream_construct() {name = "pipe_1_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %19 = allo.stream_construct() {name = "pipe_1_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %20 = allo.stream_construct() {name = "pipe_1_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %21 = allo.stream_construct() {name = "pipe_1_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %22 = allo.stream_construct() {name = "pipe_1_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %23 = allo.stream_construct() {name = "pipe_1_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %24 = allo.stream_construct() {name = "pipe_1_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %25 = allo.stream_construct() {name = "pipe_1_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %26 = allo.stream_construct() {name = "pipe_1_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %27 = allo.stream_construct() {name = "pipe_1_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %28 = allo.stream_construct() {name = "pipe_1_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %29 = allo.stream_construct() {name = "pipe_1_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    %30 = allo.stream_construct() {name = "pipe_2_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %31 = allo.stream_construct() {name = "pipe_2_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %32 = allo.stream_construct() {name = "pipe_2_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %33 = allo.stream_construct() {name = "pipe_2_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %34 = allo.stream_construct() {name = "pipe_2_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %35 = allo.stream_construct() {name = "pipe_2_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %36 = allo.stream_construct() {name = "pipe_2_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %37 = allo.stream_construct() {name = "pipe_2_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %38 = allo.stream_construct() {name = "pipe_2_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %39 = allo.stream_construct() {name = "pipe_2_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %40 = allo.stream_construct() {name = "pipe_2_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %41 = allo.stream_construct() {name = "pipe_2_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %42 = allo.stream_construct() {name = "pipe_2_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %43 = allo.stream_construct() {name = "pipe_2_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %44 = allo.stream_construct() {name = "pipe_2_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    %45 = allo.stream_construct() {name = "pipe_3_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %46 = allo.stream_construct() {name = "pipe_3_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %47 = allo.stream_construct() {name = "pipe_3_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %48 = allo.stream_construct() {name = "pipe_3_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %49 = allo.stream_construct() {name = "pipe_3_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %50 = allo.stream_construct() {name = "pipe_3_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %51 = allo.stream_construct() {name = "pipe_3_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %52 = allo.stream_construct() {name = "pipe_3_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %53 = allo.stream_construct() {name = "pipe_3_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %54 = allo.stream_construct() {name = "pipe_3_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %55 = allo.stream_construct() {name = "pipe_3_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %56 = allo.stream_construct() {name = "pipe_3_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %57 = allo.stream_construct() {name = "pipe_3_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %58 = allo.stream_construct() {name = "pipe_3_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %59 = allo.stream_construct() {name = "pipe_3_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    %60 = allo.stream_construct() {name = "pipe_4_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %61 = allo.stream_construct() {name = "pipe_4_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %62 = allo.stream_construct() {name = "pipe_4_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %63 = allo.stream_construct() {name = "pipe_4_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %64 = allo.stream_construct() {name = "pipe_4_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %65 = allo.stream_construct() {name = "pipe_4_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %66 = allo.stream_construct() {name = "pipe_4_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %67 = allo.stream_construct() {name = "pipe_4_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %68 = allo.stream_construct() {name = "pipe_4_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %69 = allo.stream_construct() {name = "pipe_4_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %70 = allo.stream_construct() {name = "pipe_4_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %71 = allo.stream_construct() {name = "pipe_4_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %72 = allo.stream_construct() {name = "pipe_4_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %73 = allo.stream_construct() {name = "pipe_4_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %74 = allo.stream_construct() {name = "pipe_4_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    %75 = allo.stream_construct() {name = "pipe_5_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %76 = allo.stream_construct() {name = "pipe_5_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %77 = allo.stream_construct() {name = "pipe_5_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %78 = allo.stream_construct() {name = "pipe_5_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %79 = allo.stream_construct() {name = "pipe_5_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %80 = allo.stream_construct() {name = "pipe_5_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %81 = allo.stream_construct() {name = "pipe_5_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %82 = allo.stream_construct() {name = "pipe_5_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %83 = allo.stream_construct() {name = "pipe_5_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %84 = allo.stream_construct() {name = "pipe_5_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %85 = allo.stream_construct() {name = "pipe_5_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %86 = allo.stream_construct() {name = "pipe_5_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %87 = allo.stream_construct() {name = "pipe_5_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %88 = allo.stream_construct() {name = "pipe_5_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %89 = allo.stream_construct() {name = "pipe_5_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    %90 = allo.stream_construct() {name = "pipe_6_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %91 = allo.stream_construct() {name = "pipe_6_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %92 = allo.stream_construct() {name = "pipe_6_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %93 = allo.stream_construct() {name = "pipe_6_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %94 = allo.stream_construct() {name = "pipe_6_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %95 = allo.stream_construct() {name = "pipe_6_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %96 = allo.stream_construct() {name = "pipe_6_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %97 = allo.stream_construct() {name = "pipe_6_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %98 = allo.stream_construct() {name = "pipe_6_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %99 = allo.stream_construct() {name = "pipe_6_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %100 = allo.stream_construct() {name = "pipe_6_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %101 = allo.stream_construct() {name = "pipe_6_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %102 = allo.stream_construct() {name = "pipe_6_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %103 = allo.stream_construct() {name = "pipe_6_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %104 = allo.stream_construct() {name = "pipe_6_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    %105 = allo.stream_construct() {name = "pipe_7_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %106 = allo.stream_construct() {name = "pipe_7_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %107 = allo.stream_construct() {name = "pipe_7_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %108 = allo.stream_construct() {name = "pipe_7_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %109 = allo.stream_construct() {name = "pipe_7_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %110 = allo.stream_construct() {name = "pipe_7_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %111 = allo.stream_construct() {name = "pipe_7_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %112 = allo.stream_construct() {name = "pipe_7_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %113 = allo.stream_construct() {name = "pipe_7_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %114 = allo.stream_construct() {name = "pipe_7_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %115 = allo.stream_construct() {name = "pipe_7_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %116 = allo.stream_construct() {name = "pipe_7_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %117 = allo.stream_construct() {name = "pipe_7_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %118 = allo.stream_construct() {name = "pipe_7_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %119 = allo.stream_construct() {name = "pipe_7_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    %120 = allo.stream_construct() {name = "pipe_8_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %121 = allo.stream_construct() {name = "pipe_8_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %122 = allo.stream_construct() {name = "pipe_8_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %123 = allo.stream_construct() {name = "pipe_8_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %124 = allo.stream_construct() {name = "pipe_8_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %125 = allo.stream_construct() {name = "pipe_8_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %126 = allo.stream_construct() {name = "pipe_8_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %127 = allo.stream_construct() {name = "pipe_8_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %128 = allo.stream_construct() {name = "pipe_8_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %129 = allo.stream_construct() {name = "pipe_8_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %130 = allo.stream_construct() {name = "pipe_8_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %131 = allo.stream_construct() {name = "pipe_8_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %132 = allo.stream_construct() {name = "pipe_8_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %133 = allo.stream_construct() {name = "pipe_8_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %134 = allo.stream_construct() {name = "pipe_8_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    %135 = allo.stream_construct() {name = "pipe_9_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %136 = allo.stream_construct() {name = "pipe_9_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %137 = allo.stream_construct() {name = "pipe_9_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %138 = allo.stream_construct() {name = "pipe_9_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %139 = allo.stream_construct() {name = "pipe_9_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %140 = allo.stream_construct() {name = "pipe_9_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %141 = allo.stream_construct() {name = "pipe_9_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %142 = allo.stream_construct() {name = "pipe_9_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %143 = allo.stream_construct() {name = "pipe_9_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %144 = allo.stream_construct() {name = "pipe_9_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %145 = allo.stream_construct() {name = "pipe_9_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %146 = allo.stream_construct() {name = "pipe_9_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %147 = allo.stream_construct() {name = "pipe_9_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %148 = allo.stream_construct() {name = "pipe_9_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %149 = allo.stream_construct() {name = "pipe_9_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    %150 = allo.stream_construct() {name = "pipe_10_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %151 = allo.stream_construct() {name = "pipe_10_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %152 = allo.stream_construct() {name = "pipe_10_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %153 = allo.stream_construct() {name = "pipe_10_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %154 = allo.stream_construct() {name = "pipe_10_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %155 = allo.stream_construct() {name = "pipe_10_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %156 = allo.stream_construct() {name = "pipe_10_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %157 = allo.stream_construct() {name = "pipe_10_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %158 = allo.stream_construct() {name = "pipe_10_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %159 = allo.stream_construct() {name = "pipe_10_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %160 = allo.stream_construct() {name = "pipe_10_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %161 = allo.stream_construct() {name = "pipe_10_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %162 = allo.stream_construct() {name = "pipe_10_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %163 = allo.stream_construct() {name = "pipe_10_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %164 = allo.stream_construct() {name = "pipe_10_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    return
  }
}
