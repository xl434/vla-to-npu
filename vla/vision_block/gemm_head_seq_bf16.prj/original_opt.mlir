module {
  func.func private @fill_zeros_bf16_64_64_vector(memref<64x64xbf16>)
  func.func private @matmul_scalar_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func private @add_bf16_vector(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func private @matmul_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func @"gemm_0_0_0-gemm_1_0_0-gemm_2_0_0-gemm_3_0_0-gemm_4_0_0-gemm_5_0_0-gemm_6_0_0-gemm_7_0_0-gemm_8_0_0-gemm_9_0_0-gemm_10_0_0-gemm_11_0_0-gemm_12_0_0-gemm_13_0_0-gemm_14_0_0-gemm_15_0_0x4"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>, %arg73: !allo.stream<memref<64x64xbf16>, 2>, %arg74: memref<64x64xbf16>, %arg75: memref<64x64xbf16>, %arg76: memref<64x64xbf16>, %arg77: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg76) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg59, %arg60, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg64, %arg65, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg69, %arg70, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg74, %arg75, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_1_0-gemm_1_1_0-gemm_2_1_0-gemm_3_1_0-gemm_4_1_0-gemm_5_1_0-gemm_6_1_0-gemm_7_1_0-gemm_8_1_0-gemm_9_1_0-gemm_10_1_0-gemm_11_1_0-gemm_12_1_0-gemm_13_1_0-gemm_14_1_0-gemm_15_1_0x4"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>, %arg73: !allo.stream<memref<64x64xbf16>, 2>, %arg74: memref<64x64xbf16>, %arg75: memref<64x64xbf16>, %arg76: memref<64x64xbf16>, %arg77: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg76) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg59, %arg60, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg64, %arg65, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg69, %arg70, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg74, %arg75, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_2_0-gemm_1_2_0-gemm_2_2_0-gemm_3_2_0-gemm_4_2_0-gemm_5_2_0-gemm_6_2_0-gemm_7_2_0-gemm_8_2_0-gemm_9_2_0-gemm_10_2_0-gemm_11_2_0-gemm_12_2_0-gemm_13_2_0-gemm_14_2_0-gemm_15_2_0x4"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>, %arg73: !allo.stream<memref<64x64xbf16>, 2>, %arg74: memref<64x64xbf16>, %arg75: memref<64x64xbf16>, %arg76: memref<64x64xbf16>, %arg77: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg76) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg59, %arg60, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg64, %arg65, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg69, %arg70, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg74, %arg75, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_3_0-gemm_1_3_0-gemm_2_3_0-gemm_3_3_0-gemm_4_3_0-gemm_5_3_0-gemm_6_3_0-gemm_7_3_0-gemm_8_3_0-gemm_9_3_0-gemm_10_3_0-gemm_11_3_0-gemm_12_3_0-gemm_13_3_0-gemm_14_3_0-gemm_15_3_0x4"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>, %arg73: !allo.stream<memref<64x64xbf16>, 2>, %arg74: memref<64x64xbf16>, %arg75: memref<64x64xbf16>, %arg76: memref<64x64xbf16>, %arg77: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg76) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg59, %arg60, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg64, %arg65, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg69, %arg70, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg74, %arg75, %arg76) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @top(%arg0: memref<1024x1024xbf16>, %arg1: memref<1024x64xbf16>, %arg2: memref<1024x64xbf16>) attributes {dataflow, itypes = "___"} {
    %0 = allo.stream_construct() {name = "pipe_0_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1 = allo.stream_construct() {name = "pipe_0_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %2 = allo.stream_construct() {name = "pipe_0_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %3 = allo.stream_construct() {name = "pipe_0_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %4 = allo.stream_construct() {name = "pipe_0_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %5 = allo.stream_construct() {name = "pipe_0_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %6 = allo.stream_construct() {name = "pipe_0_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %7 = allo.stream_construct() {name = "pipe_0_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %8 = allo.stream_construct() {name = "pipe_0_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %9 = allo.stream_construct() {name = "pipe_0_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %10 = allo.stream_construct() {name = "pipe_0_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %11 = allo.stream_construct() {name = "pipe_0_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %12 = allo.stream_construct() {name = "pipe_0_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %13 = allo.stream_construct() {name = "pipe_0_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %14 = allo.stream_construct() {name = "pipe_0_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %15 = allo.stream_construct() {name = "pipe_0_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %16 = allo.stream_construct() {name = "pipe_1_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %17 = allo.stream_construct() {name = "pipe_1_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %18 = allo.stream_construct() {name = "pipe_1_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %19 = allo.stream_construct() {name = "pipe_1_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %20 = allo.stream_construct() {name = "pipe_1_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %21 = allo.stream_construct() {name = "pipe_1_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %22 = allo.stream_construct() {name = "pipe_1_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %23 = allo.stream_construct() {name = "pipe_1_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %24 = allo.stream_construct() {name = "pipe_1_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %25 = allo.stream_construct() {name = "pipe_1_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %26 = allo.stream_construct() {name = "pipe_1_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %27 = allo.stream_construct() {name = "pipe_1_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %28 = allo.stream_construct() {name = "pipe_1_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %29 = allo.stream_construct() {name = "pipe_1_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %30 = allo.stream_construct() {name = "pipe_1_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %31 = allo.stream_construct() {name = "pipe_1_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %32 = allo.stream_construct() {name = "pipe_2_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %33 = allo.stream_construct() {name = "pipe_2_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %34 = allo.stream_construct() {name = "pipe_2_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %35 = allo.stream_construct() {name = "pipe_2_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %36 = allo.stream_construct() {name = "pipe_2_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %37 = allo.stream_construct() {name = "pipe_2_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %38 = allo.stream_construct() {name = "pipe_2_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %39 = allo.stream_construct() {name = "pipe_2_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %40 = allo.stream_construct() {name = "pipe_2_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %41 = allo.stream_construct() {name = "pipe_2_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %42 = allo.stream_construct() {name = "pipe_2_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %43 = allo.stream_construct() {name = "pipe_2_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %44 = allo.stream_construct() {name = "pipe_2_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %45 = allo.stream_construct() {name = "pipe_2_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %46 = allo.stream_construct() {name = "pipe_2_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %47 = allo.stream_construct() {name = "pipe_2_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %48 = allo.stream_construct() {name = "pipe_3_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %49 = allo.stream_construct() {name = "pipe_3_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %50 = allo.stream_construct() {name = "pipe_3_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %51 = allo.stream_construct() {name = "pipe_3_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %52 = allo.stream_construct() {name = "pipe_3_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %53 = allo.stream_construct() {name = "pipe_3_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %54 = allo.stream_construct() {name = "pipe_3_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %55 = allo.stream_construct() {name = "pipe_3_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %56 = allo.stream_construct() {name = "pipe_3_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %57 = allo.stream_construct() {name = "pipe_3_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %58 = allo.stream_construct() {name = "pipe_3_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %59 = allo.stream_construct() {name = "pipe_3_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %60 = allo.stream_construct() {name = "pipe_3_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %61 = allo.stream_construct() {name = "pipe_3_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %62 = allo.stream_construct() {name = "pipe_3_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %63 = allo.stream_construct() {name = "pipe_3_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %64 = allo.stream_construct() {name = "pipe_4_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %65 = allo.stream_construct() {name = "pipe_4_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %66 = allo.stream_construct() {name = "pipe_4_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %67 = allo.stream_construct() {name = "pipe_4_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %68 = allo.stream_construct() {name = "pipe_4_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %69 = allo.stream_construct() {name = "pipe_4_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %70 = allo.stream_construct() {name = "pipe_4_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %71 = allo.stream_construct() {name = "pipe_4_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %72 = allo.stream_construct() {name = "pipe_4_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %73 = allo.stream_construct() {name = "pipe_4_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %74 = allo.stream_construct() {name = "pipe_4_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %75 = allo.stream_construct() {name = "pipe_4_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %76 = allo.stream_construct() {name = "pipe_4_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %77 = allo.stream_construct() {name = "pipe_4_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %78 = allo.stream_construct() {name = "pipe_4_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %79 = allo.stream_construct() {name = "pipe_4_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %80 = allo.stream_construct() {name = "pipe_5_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %81 = allo.stream_construct() {name = "pipe_5_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %82 = allo.stream_construct() {name = "pipe_5_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %83 = allo.stream_construct() {name = "pipe_5_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %84 = allo.stream_construct() {name = "pipe_5_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %85 = allo.stream_construct() {name = "pipe_5_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %86 = allo.stream_construct() {name = "pipe_5_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %87 = allo.stream_construct() {name = "pipe_5_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %88 = allo.stream_construct() {name = "pipe_5_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %89 = allo.stream_construct() {name = "pipe_5_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %90 = allo.stream_construct() {name = "pipe_5_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %91 = allo.stream_construct() {name = "pipe_5_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %92 = allo.stream_construct() {name = "pipe_5_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %93 = allo.stream_construct() {name = "pipe_5_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %94 = allo.stream_construct() {name = "pipe_5_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %95 = allo.stream_construct() {name = "pipe_5_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %96 = allo.stream_construct() {name = "pipe_6_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %97 = allo.stream_construct() {name = "pipe_6_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %98 = allo.stream_construct() {name = "pipe_6_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %99 = allo.stream_construct() {name = "pipe_6_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %100 = allo.stream_construct() {name = "pipe_6_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %101 = allo.stream_construct() {name = "pipe_6_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %102 = allo.stream_construct() {name = "pipe_6_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %103 = allo.stream_construct() {name = "pipe_6_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %104 = allo.stream_construct() {name = "pipe_6_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %105 = allo.stream_construct() {name = "pipe_6_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %106 = allo.stream_construct() {name = "pipe_6_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %107 = allo.stream_construct() {name = "pipe_6_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %108 = allo.stream_construct() {name = "pipe_6_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %109 = allo.stream_construct() {name = "pipe_6_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %110 = allo.stream_construct() {name = "pipe_6_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %111 = allo.stream_construct() {name = "pipe_6_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %112 = allo.stream_construct() {name = "pipe_7_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %113 = allo.stream_construct() {name = "pipe_7_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %114 = allo.stream_construct() {name = "pipe_7_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %115 = allo.stream_construct() {name = "pipe_7_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %116 = allo.stream_construct() {name = "pipe_7_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %117 = allo.stream_construct() {name = "pipe_7_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %118 = allo.stream_construct() {name = "pipe_7_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %119 = allo.stream_construct() {name = "pipe_7_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %120 = allo.stream_construct() {name = "pipe_7_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %121 = allo.stream_construct() {name = "pipe_7_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %122 = allo.stream_construct() {name = "pipe_7_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %123 = allo.stream_construct() {name = "pipe_7_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %124 = allo.stream_construct() {name = "pipe_7_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %125 = allo.stream_construct() {name = "pipe_7_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %126 = allo.stream_construct() {name = "pipe_7_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %127 = allo.stream_construct() {name = "pipe_7_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %128 = allo.stream_construct() {name = "pipe_8_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %129 = allo.stream_construct() {name = "pipe_8_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %130 = allo.stream_construct() {name = "pipe_8_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %131 = allo.stream_construct() {name = "pipe_8_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %132 = allo.stream_construct() {name = "pipe_8_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %133 = allo.stream_construct() {name = "pipe_8_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %134 = allo.stream_construct() {name = "pipe_8_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %135 = allo.stream_construct() {name = "pipe_8_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %136 = allo.stream_construct() {name = "pipe_8_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %137 = allo.stream_construct() {name = "pipe_8_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %138 = allo.stream_construct() {name = "pipe_8_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %139 = allo.stream_construct() {name = "pipe_8_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %140 = allo.stream_construct() {name = "pipe_8_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %141 = allo.stream_construct() {name = "pipe_8_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %142 = allo.stream_construct() {name = "pipe_8_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %143 = allo.stream_construct() {name = "pipe_8_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %144 = allo.stream_construct() {name = "pipe_9_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %145 = allo.stream_construct() {name = "pipe_9_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %146 = allo.stream_construct() {name = "pipe_9_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %147 = allo.stream_construct() {name = "pipe_9_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %148 = allo.stream_construct() {name = "pipe_9_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %149 = allo.stream_construct() {name = "pipe_9_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %150 = allo.stream_construct() {name = "pipe_9_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %151 = allo.stream_construct() {name = "pipe_9_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %152 = allo.stream_construct() {name = "pipe_9_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %153 = allo.stream_construct() {name = "pipe_9_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %154 = allo.stream_construct() {name = "pipe_9_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %155 = allo.stream_construct() {name = "pipe_9_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %156 = allo.stream_construct() {name = "pipe_9_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %157 = allo.stream_construct() {name = "pipe_9_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %158 = allo.stream_construct() {name = "pipe_9_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %159 = allo.stream_construct() {name = "pipe_9_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %160 = allo.stream_construct() {name = "pipe_10_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %161 = allo.stream_construct() {name = "pipe_10_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %162 = allo.stream_construct() {name = "pipe_10_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %163 = allo.stream_construct() {name = "pipe_10_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %164 = allo.stream_construct() {name = "pipe_10_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %165 = allo.stream_construct() {name = "pipe_10_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %166 = allo.stream_construct() {name = "pipe_10_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %167 = allo.stream_construct() {name = "pipe_10_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %168 = allo.stream_construct() {name = "pipe_10_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %169 = allo.stream_construct() {name = "pipe_10_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %170 = allo.stream_construct() {name = "pipe_10_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %171 = allo.stream_construct() {name = "pipe_10_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %172 = allo.stream_construct() {name = "pipe_10_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %173 = allo.stream_construct() {name = "pipe_10_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %174 = allo.stream_construct() {name = "pipe_10_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %175 = allo.stream_construct() {name = "pipe_10_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %176 = allo.stream_construct() {name = "pipe_11_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %177 = allo.stream_construct() {name = "pipe_11_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %178 = allo.stream_construct() {name = "pipe_11_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %179 = allo.stream_construct() {name = "pipe_11_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %180 = allo.stream_construct() {name = "pipe_11_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %181 = allo.stream_construct() {name = "pipe_11_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %182 = allo.stream_construct() {name = "pipe_11_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %183 = allo.stream_construct() {name = "pipe_11_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %184 = allo.stream_construct() {name = "pipe_11_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %185 = allo.stream_construct() {name = "pipe_11_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %186 = allo.stream_construct() {name = "pipe_11_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %187 = allo.stream_construct() {name = "pipe_11_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %188 = allo.stream_construct() {name = "pipe_11_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %189 = allo.stream_construct() {name = "pipe_11_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %190 = allo.stream_construct() {name = "pipe_11_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %191 = allo.stream_construct() {name = "pipe_11_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %192 = allo.stream_construct() {name = "pipe_12_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %193 = allo.stream_construct() {name = "pipe_12_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %194 = allo.stream_construct() {name = "pipe_12_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %195 = allo.stream_construct() {name = "pipe_12_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %196 = allo.stream_construct() {name = "pipe_12_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %197 = allo.stream_construct() {name = "pipe_12_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %198 = allo.stream_construct() {name = "pipe_12_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %199 = allo.stream_construct() {name = "pipe_12_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %200 = allo.stream_construct() {name = "pipe_12_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %201 = allo.stream_construct() {name = "pipe_12_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %202 = allo.stream_construct() {name = "pipe_12_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %203 = allo.stream_construct() {name = "pipe_12_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %204 = allo.stream_construct() {name = "pipe_12_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %205 = allo.stream_construct() {name = "pipe_12_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %206 = allo.stream_construct() {name = "pipe_12_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %207 = allo.stream_construct() {name = "pipe_12_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %208 = allo.stream_construct() {name = "pipe_13_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %209 = allo.stream_construct() {name = "pipe_13_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %210 = allo.stream_construct() {name = "pipe_13_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %211 = allo.stream_construct() {name = "pipe_13_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %212 = allo.stream_construct() {name = "pipe_13_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %213 = allo.stream_construct() {name = "pipe_13_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %214 = allo.stream_construct() {name = "pipe_13_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %215 = allo.stream_construct() {name = "pipe_13_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %216 = allo.stream_construct() {name = "pipe_13_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %217 = allo.stream_construct() {name = "pipe_13_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %218 = allo.stream_construct() {name = "pipe_13_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %219 = allo.stream_construct() {name = "pipe_13_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %220 = allo.stream_construct() {name = "pipe_13_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %221 = allo.stream_construct() {name = "pipe_13_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %222 = allo.stream_construct() {name = "pipe_13_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %223 = allo.stream_construct() {name = "pipe_13_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %224 = allo.stream_construct() {name = "pipe_14_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %225 = allo.stream_construct() {name = "pipe_14_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %226 = allo.stream_construct() {name = "pipe_14_2_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %227 = allo.stream_construct() {name = "pipe_14_3_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %228 = allo.stream_construct() {name = "pipe_14_4_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %229 = allo.stream_construct() {name = "pipe_14_5_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %230 = allo.stream_construct() {name = "pipe_14_6_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %231 = allo.stream_construct() {name = "pipe_14_7_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %232 = allo.stream_construct() {name = "pipe_14_8_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %233 = allo.stream_construct() {name = "pipe_14_9_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %234 = allo.stream_construct() {name = "pipe_14_10_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %235 = allo.stream_construct() {name = "pipe_14_11_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %236 = allo.stream_construct() {name = "pipe_14_12_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %237 = allo.stream_construct() {name = "pipe_14_13_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %238 = allo.stream_construct() {name = "pipe_14_14_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %239 = allo.stream_construct() {name = "pipe_14_15_0"} : !allo.stream<memref<64x64xbf16>, 2>
    return
  }
}
