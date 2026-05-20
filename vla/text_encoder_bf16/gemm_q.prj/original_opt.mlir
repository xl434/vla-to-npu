module {
  func.func private @fill_zeros_bf16_64_64_vector(memref<64x64xbf16>)
  func.func private @matmul_scalar_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func private @add_bf16_vector(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func private @matmul_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func @"gemm_0_0_12-gemm_1_0_12-gemm_2_0_12-gemm_3_0_12-gemm_4_0_12-gemm_5_0_12-gemm_6_0_12-gemm_7_0_12-gemm_8_0_12-gemm_9_0_12-gemm_10_0_12-gemm_11_0_12-gemm_12_0_12-gemm_13_0_12-gemm_14_0_12"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg71) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg59, %arg60, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg64, %arg65, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg69, %arg70, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_0_13-gemm_1_0_13-gemm_2_0_13-gemm_3_0_13-gemm_4_0_13-gemm_5_0_13-gemm_6_0_13-gemm_7_0_13-gemm_8_0_13-gemm_9_0_13-gemm_10_0_13-gemm_11_0_13-gemm_12_0_13-gemm_13_0_13-gemm_14_0_13"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg71) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg59, %arg60, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg64, %arg65, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg69, %arg70, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_0_14-gemm_1_0_14-gemm_2_0_14-gemm_3_0_14-gemm_4_0_14-gemm_5_0_14-gemm_6_0_14-gemm_7_0_14-gemm_8_0_14-gemm_9_0_14-gemm_10_0_14-gemm_11_0_14-gemm_12_0_14-gemm_13_0_14-gemm_14_0_14"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg71) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg59, %arg60, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg64, %arg65, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg69, %arg70, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_1_12-gemm_1_1_12-gemm_2_1_12-gemm_3_1_12-gemm_4_1_12-gemm_5_1_12-gemm_6_1_12-gemm_7_1_12-gemm_8_1_12-gemm_9_1_12-gemm_10_1_12-gemm_11_1_12-gemm_12_1_12-gemm_13_1_12-gemm_14_1_12"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg71) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg59, %arg60, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg64, %arg65, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg69, %arg70, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_1_13-gemm_1_1_13-gemm_2_1_13-gemm_3_1_13-gemm_4_1_13-gemm_5_1_13-gemm_6_1_13-gemm_7_1_13-gemm_8_1_13-gemm_9_1_13-gemm_10_1_13-gemm_11_1_13-gemm_12_1_13-gemm_13_1_13-gemm_14_1_13"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg71) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg59, %arg60, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg64, %arg65, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg69, %arg70, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_1_14-gemm_1_1_14-gemm_2_1_14-gemm_3_1_14-gemm_4_1_14-gemm_5_1_14-gemm_6_1_14-gemm_7_1_14-gemm_8_1_14-gemm_9_1_14-gemm_10_1_14-gemm_11_1_14-gemm_12_1_14-gemm_13_1_14-gemm_14_1_14"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg71) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg59, %arg60, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg64, %arg65, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg69, %arg70, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_0_0-gemm_1_0_0-gemm_2_0_0-gemm_3_0_0-gemm_4_0_0-gemm_5_0_0-gemm_6_0_0-gemm_7_0_0-gemm_8_0_0-gemm_9_0_0-gemm_10_0_0-gemm_11_0_0-gemm_12_0_0-gemm_13_0_0-gemm_14_0_0x3"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg71) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg59, %arg60, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg64, %arg65, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg69, %arg70, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_0_1-gemm_1_0_1-gemm_2_0_1-gemm_3_0_1-gemm_4_0_1-gemm_5_0_1-gemm_6_0_1-gemm_7_0_1-gemm_8_0_1-gemm_9_0_1-gemm_10_0_1-gemm_11_0_1-gemm_12_0_1-gemm_13_0_1-gemm_14_0_1x3"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg71) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg59, %arg60, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg64, %arg65, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg69, %arg70, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_0_2-gemm_1_0_2-gemm_2_0_2-gemm_3_0_2-gemm_4_0_2-gemm_5_0_2-gemm_6_0_2-gemm_7_0_2-gemm_8_0_2-gemm_9_0_2-gemm_10_0_2-gemm_11_0_2-gemm_12_0_2-gemm_13_0_2-gemm_14_0_2x3"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg71) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg59, %arg60, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg64, %arg65, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg69, %arg70, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_0_3-gemm_1_0_3-gemm_2_0_3-gemm_3_0_3-gemm_4_0_3-gemm_5_0_3-gemm_6_0_3-gemm_7_0_3-gemm_8_0_3-gemm_9_0_3-gemm_10_0_3-gemm_11_0_3-gemm_12_0_3-gemm_13_0_3-gemm_14_0_3x3"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg71) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg59, %arg60, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg64, %arg65, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg69, %arg70, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_1_0-gemm_1_1_0-gemm_2_1_0-gemm_3_1_0-gemm_4_1_0-gemm_5_1_0-gemm_6_1_0-gemm_7_1_0-gemm_8_1_0-gemm_9_1_0-gemm_10_1_0-gemm_11_1_0-gemm_12_1_0-gemm_13_1_0-gemm_14_1_0x3"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg71) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg59, %arg60, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg64, %arg65, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg69, %arg70, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_1_1-gemm_1_1_1-gemm_2_1_1-gemm_3_1_1-gemm_4_1_1-gemm_5_1_1-gemm_6_1_1-gemm_7_1_1-gemm_8_1_1-gemm_9_1_1-gemm_10_1_1-gemm_11_1_1-gemm_12_1_1-gemm_13_1_1-gemm_14_1_1x3"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg71) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg59, %arg60, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg64, %arg65, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg69, %arg70, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_1_2-gemm_1_1_2-gemm_2_1_2-gemm_3_1_2-gemm_4_1_2-gemm_5_1_2-gemm_6_1_2-gemm_7_1_2-gemm_8_1_2-gemm_9_1_2-gemm_10_1_2-gemm_11_1_2-gemm_12_1_2-gemm_13_1_2-gemm_14_1_2x3"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg71) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg59, %arg60, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg64, %arg65, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg69, %arg70, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @"gemm_0_1_3-gemm_1_1_3-gemm_2_1_3-gemm_3_1_3-gemm_4_1_3-gemm_5_1_3-gemm_6_1_3-gemm_7_1_3-gemm_8_1_3-gemm_9_1_3-gemm_10_1_3-gemm_11_1_3-gemm_12_1_3-gemm_13_1_3-gemm_14_1_3x3"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    call @fill_zeros_bf16_64_64_vector(%arg71) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg0, %arg1, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg4, %arg5, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg9, %arg10, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg14, %arg15, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg19, %arg20, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg24, %arg25, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg29, %arg30, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg34, %arg35, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg39, %arg40, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg44, %arg45, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg49, %arg50, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg54, %arg55, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg59, %arg60, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg64, %arg65, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    call @matmul_bf16_bf16(%arg69, %arg70, %arg71) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @top(%arg0: memref<128x960xbf16>, %arg1: memref<960x960xbf16>, %arg2: memref<128x960xbf16>) attributes {dataflow, itypes = "___"} {
    %0 = allo.stream_construct() {name = "pipe_0_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1 = allo.stream_construct() {name = "pipe_0_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %2 = allo.stream_construct() {name = "pipe_0_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %3 = allo.stream_construct() {name = "pipe_0_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %4 = allo.stream_construct() {name = "pipe_0_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %5 = allo.stream_construct() {name = "pipe_0_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %6 = allo.stream_construct() {name = "pipe_0_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %7 = allo.stream_construct() {name = "pipe_0_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %8 = allo.stream_construct() {name = "pipe_0_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %9 = allo.stream_construct() {name = "pipe_0_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %10 = allo.stream_construct() {name = "pipe_0_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %11 = allo.stream_construct() {name = "pipe_0_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %12 = allo.stream_construct() {name = "pipe_0_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %13 = allo.stream_construct() {name = "pipe_0_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %14 = allo.stream_construct() {name = "pipe_0_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %15 = allo.stream_construct() {name = "pipe_0_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %16 = allo.stream_construct() {name = "pipe_0_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %17 = allo.stream_construct() {name = "pipe_0_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %18 = allo.stream_construct() {name = "pipe_0_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %19 = allo.stream_construct() {name = "pipe_0_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %20 = allo.stream_construct() {name = "pipe_0_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %21 = allo.stream_construct() {name = "pipe_0_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %22 = allo.stream_construct() {name = "pipe_0_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %23 = allo.stream_construct() {name = "pipe_0_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %24 = allo.stream_construct() {name = "pipe_0_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %25 = allo.stream_construct() {name = "pipe_0_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %26 = allo.stream_construct() {name = "pipe_0_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %27 = allo.stream_construct() {name = "pipe_0_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %28 = allo.stream_construct() {name = "pipe_0_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %29 = allo.stream_construct() {name = "pipe_0_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %30 = allo.stream_construct() {name = "pipe_1_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %31 = allo.stream_construct() {name = "pipe_1_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %32 = allo.stream_construct() {name = "pipe_1_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %33 = allo.stream_construct() {name = "pipe_1_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %34 = allo.stream_construct() {name = "pipe_1_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %35 = allo.stream_construct() {name = "pipe_1_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %36 = allo.stream_construct() {name = "pipe_1_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %37 = allo.stream_construct() {name = "pipe_1_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %38 = allo.stream_construct() {name = "pipe_1_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %39 = allo.stream_construct() {name = "pipe_1_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %40 = allo.stream_construct() {name = "pipe_1_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %41 = allo.stream_construct() {name = "pipe_1_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %42 = allo.stream_construct() {name = "pipe_1_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %43 = allo.stream_construct() {name = "pipe_1_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %44 = allo.stream_construct() {name = "pipe_1_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %45 = allo.stream_construct() {name = "pipe_1_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %46 = allo.stream_construct() {name = "pipe_1_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %47 = allo.stream_construct() {name = "pipe_1_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %48 = allo.stream_construct() {name = "pipe_1_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %49 = allo.stream_construct() {name = "pipe_1_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %50 = allo.stream_construct() {name = "pipe_1_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %51 = allo.stream_construct() {name = "pipe_1_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %52 = allo.stream_construct() {name = "pipe_1_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %53 = allo.stream_construct() {name = "pipe_1_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %54 = allo.stream_construct() {name = "pipe_1_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %55 = allo.stream_construct() {name = "pipe_1_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %56 = allo.stream_construct() {name = "pipe_1_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %57 = allo.stream_construct() {name = "pipe_1_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %58 = allo.stream_construct() {name = "pipe_1_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %59 = allo.stream_construct() {name = "pipe_1_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %60 = allo.stream_construct() {name = "pipe_2_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %61 = allo.stream_construct() {name = "pipe_2_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %62 = allo.stream_construct() {name = "pipe_2_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %63 = allo.stream_construct() {name = "pipe_2_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %64 = allo.stream_construct() {name = "pipe_2_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %65 = allo.stream_construct() {name = "pipe_2_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %66 = allo.stream_construct() {name = "pipe_2_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %67 = allo.stream_construct() {name = "pipe_2_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %68 = allo.stream_construct() {name = "pipe_2_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %69 = allo.stream_construct() {name = "pipe_2_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %70 = allo.stream_construct() {name = "pipe_2_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %71 = allo.stream_construct() {name = "pipe_2_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %72 = allo.stream_construct() {name = "pipe_2_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %73 = allo.stream_construct() {name = "pipe_2_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %74 = allo.stream_construct() {name = "pipe_2_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %75 = allo.stream_construct() {name = "pipe_2_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %76 = allo.stream_construct() {name = "pipe_2_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %77 = allo.stream_construct() {name = "pipe_2_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %78 = allo.stream_construct() {name = "pipe_2_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %79 = allo.stream_construct() {name = "pipe_2_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %80 = allo.stream_construct() {name = "pipe_2_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %81 = allo.stream_construct() {name = "pipe_2_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %82 = allo.stream_construct() {name = "pipe_2_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %83 = allo.stream_construct() {name = "pipe_2_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %84 = allo.stream_construct() {name = "pipe_2_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %85 = allo.stream_construct() {name = "pipe_2_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %86 = allo.stream_construct() {name = "pipe_2_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %87 = allo.stream_construct() {name = "pipe_2_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %88 = allo.stream_construct() {name = "pipe_2_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %89 = allo.stream_construct() {name = "pipe_2_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %90 = allo.stream_construct() {name = "pipe_3_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %91 = allo.stream_construct() {name = "pipe_3_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %92 = allo.stream_construct() {name = "pipe_3_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %93 = allo.stream_construct() {name = "pipe_3_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %94 = allo.stream_construct() {name = "pipe_3_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %95 = allo.stream_construct() {name = "pipe_3_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %96 = allo.stream_construct() {name = "pipe_3_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %97 = allo.stream_construct() {name = "pipe_3_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %98 = allo.stream_construct() {name = "pipe_3_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %99 = allo.stream_construct() {name = "pipe_3_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %100 = allo.stream_construct() {name = "pipe_3_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %101 = allo.stream_construct() {name = "pipe_3_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %102 = allo.stream_construct() {name = "pipe_3_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %103 = allo.stream_construct() {name = "pipe_3_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %104 = allo.stream_construct() {name = "pipe_3_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %105 = allo.stream_construct() {name = "pipe_3_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %106 = allo.stream_construct() {name = "pipe_3_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %107 = allo.stream_construct() {name = "pipe_3_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %108 = allo.stream_construct() {name = "pipe_3_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %109 = allo.stream_construct() {name = "pipe_3_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %110 = allo.stream_construct() {name = "pipe_3_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %111 = allo.stream_construct() {name = "pipe_3_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %112 = allo.stream_construct() {name = "pipe_3_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %113 = allo.stream_construct() {name = "pipe_3_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %114 = allo.stream_construct() {name = "pipe_3_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %115 = allo.stream_construct() {name = "pipe_3_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %116 = allo.stream_construct() {name = "pipe_3_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %117 = allo.stream_construct() {name = "pipe_3_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %118 = allo.stream_construct() {name = "pipe_3_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %119 = allo.stream_construct() {name = "pipe_3_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %120 = allo.stream_construct() {name = "pipe_4_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %121 = allo.stream_construct() {name = "pipe_4_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %122 = allo.stream_construct() {name = "pipe_4_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %123 = allo.stream_construct() {name = "pipe_4_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %124 = allo.stream_construct() {name = "pipe_4_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %125 = allo.stream_construct() {name = "pipe_4_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %126 = allo.stream_construct() {name = "pipe_4_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %127 = allo.stream_construct() {name = "pipe_4_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %128 = allo.stream_construct() {name = "pipe_4_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %129 = allo.stream_construct() {name = "pipe_4_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %130 = allo.stream_construct() {name = "pipe_4_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %131 = allo.stream_construct() {name = "pipe_4_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %132 = allo.stream_construct() {name = "pipe_4_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %133 = allo.stream_construct() {name = "pipe_4_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %134 = allo.stream_construct() {name = "pipe_4_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %135 = allo.stream_construct() {name = "pipe_4_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %136 = allo.stream_construct() {name = "pipe_4_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %137 = allo.stream_construct() {name = "pipe_4_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %138 = allo.stream_construct() {name = "pipe_4_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %139 = allo.stream_construct() {name = "pipe_4_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %140 = allo.stream_construct() {name = "pipe_4_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %141 = allo.stream_construct() {name = "pipe_4_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %142 = allo.stream_construct() {name = "pipe_4_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %143 = allo.stream_construct() {name = "pipe_4_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %144 = allo.stream_construct() {name = "pipe_4_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %145 = allo.stream_construct() {name = "pipe_4_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %146 = allo.stream_construct() {name = "pipe_4_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %147 = allo.stream_construct() {name = "pipe_4_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %148 = allo.stream_construct() {name = "pipe_4_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %149 = allo.stream_construct() {name = "pipe_4_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %150 = allo.stream_construct() {name = "pipe_5_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %151 = allo.stream_construct() {name = "pipe_5_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %152 = allo.stream_construct() {name = "pipe_5_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %153 = allo.stream_construct() {name = "pipe_5_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %154 = allo.stream_construct() {name = "pipe_5_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %155 = allo.stream_construct() {name = "pipe_5_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %156 = allo.stream_construct() {name = "pipe_5_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %157 = allo.stream_construct() {name = "pipe_5_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %158 = allo.stream_construct() {name = "pipe_5_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %159 = allo.stream_construct() {name = "pipe_5_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %160 = allo.stream_construct() {name = "pipe_5_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %161 = allo.stream_construct() {name = "pipe_5_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %162 = allo.stream_construct() {name = "pipe_5_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %163 = allo.stream_construct() {name = "pipe_5_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %164 = allo.stream_construct() {name = "pipe_5_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %165 = allo.stream_construct() {name = "pipe_5_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %166 = allo.stream_construct() {name = "pipe_5_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %167 = allo.stream_construct() {name = "pipe_5_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %168 = allo.stream_construct() {name = "pipe_5_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %169 = allo.stream_construct() {name = "pipe_5_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %170 = allo.stream_construct() {name = "pipe_5_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %171 = allo.stream_construct() {name = "pipe_5_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %172 = allo.stream_construct() {name = "pipe_5_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %173 = allo.stream_construct() {name = "pipe_5_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %174 = allo.stream_construct() {name = "pipe_5_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %175 = allo.stream_construct() {name = "pipe_5_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %176 = allo.stream_construct() {name = "pipe_5_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %177 = allo.stream_construct() {name = "pipe_5_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %178 = allo.stream_construct() {name = "pipe_5_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %179 = allo.stream_construct() {name = "pipe_5_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %180 = allo.stream_construct() {name = "pipe_6_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %181 = allo.stream_construct() {name = "pipe_6_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %182 = allo.stream_construct() {name = "pipe_6_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %183 = allo.stream_construct() {name = "pipe_6_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %184 = allo.stream_construct() {name = "pipe_6_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %185 = allo.stream_construct() {name = "pipe_6_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %186 = allo.stream_construct() {name = "pipe_6_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %187 = allo.stream_construct() {name = "pipe_6_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %188 = allo.stream_construct() {name = "pipe_6_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %189 = allo.stream_construct() {name = "pipe_6_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %190 = allo.stream_construct() {name = "pipe_6_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %191 = allo.stream_construct() {name = "pipe_6_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %192 = allo.stream_construct() {name = "pipe_6_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %193 = allo.stream_construct() {name = "pipe_6_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %194 = allo.stream_construct() {name = "pipe_6_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %195 = allo.stream_construct() {name = "pipe_6_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %196 = allo.stream_construct() {name = "pipe_6_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %197 = allo.stream_construct() {name = "pipe_6_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %198 = allo.stream_construct() {name = "pipe_6_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %199 = allo.stream_construct() {name = "pipe_6_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %200 = allo.stream_construct() {name = "pipe_6_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %201 = allo.stream_construct() {name = "pipe_6_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %202 = allo.stream_construct() {name = "pipe_6_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %203 = allo.stream_construct() {name = "pipe_6_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %204 = allo.stream_construct() {name = "pipe_6_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %205 = allo.stream_construct() {name = "pipe_6_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %206 = allo.stream_construct() {name = "pipe_6_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %207 = allo.stream_construct() {name = "pipe_6_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %208 = allo.stream_construct() {name = "pipe_6_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %209 = allo.stream_construct() {name = "pipe_6_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %210 = allo.stream_construct() {name = "pipe_7_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %211 = allo.stream_construct() {name = "pipe_7_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %212 = allo.stream_construct() {name = "pipe_7_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %213 = allo.stream_construct() {name = "pipe_7_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %214 = allo.stream_construct() {name = "pipe_7_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %215 = allo.stream_construct() {name = "pipe_7_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %216 = allo.stream_construct() {name = "pipe_7_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %217 = allo.stream_construct() {name = "pipe_7_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %218 = allo.stream_construct() {name = "pipe_7_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %219 = allo.stream_construct() {name = "pipe_7_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %220 = allo.stream_construct() {name = "pipe_7_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %221 = allo.stream_construct() {name = "pipe_7_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %222 = allo.stream_construct() {name = "pipe_7_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %223 = allo.stream_construct() {name = "pipe_7_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %224 = allo.stream_construct() {name = "pipe_7_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %225 = allo.stream_construct() {name = "pipe_7_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %226 = allo.stream_construct() {name = "pipe_7_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %227 = allo.stream_construct() {name = "pipe_7_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %228 = allo.stream_construct() {name = "pipe_7_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %229 = allo.stream_construct() {name = "pipe_7_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %230 = allo.stream_construct() {name = "pipe_7_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %231 = allo.stream_construct() {name = "pipe_7_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %232 = allo.stream_construct() {name = "pipe_7_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %233 = allo.stream_construct() {name = "pipe_7_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %234 = allo.stream_construct() {name = "pipe_7_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %235 = allo.stream_construct() {name = "pipe_7_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %236 = allo.stream_construct() {name = "pipe_7_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %237 = allo.stream_construct() {name = "pipe_7_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %238 = allo.stream_construct() {name = "pipe_7_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %239 = allo.stream_construct() {name = "pipe_7_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %240 = allo.stream_construct() {name = "pipe_8_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %241 = allo.stream_construct() {name = "pipe_8_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %242 = allo.stream_construct() {name = "pipe_8_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %243 = allo.stream_construct() {name = "pipe_8_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %244 = allo.stream_construct() {name = "pipe_8_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %245 = allo.stream_construct() {name = "pipe_8_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %246 = allo.stream_construct() {name = "pipe_8_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %247 = allo.stream_construct() {name = "pipe_8_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %248 = allo.stream_construct() {name = "pipe_8_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %249 = allo.stream_construct() {name = "pipe_8_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %250 = allo.stream_construct() {name = "pipe_8_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %251 = allo.stream_construct() {name = "pipe_8_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %252 = allo.stream_construct() {name = "pipe_8_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %253 = allo.stream_construct() {name = "pipe_8_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %254 = allo.stream_construct() {name = "pipe_8_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %255 = allo.stream_construct() {name = "pipe_8_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %256 = allo.stream_construct() {name = "pipe_8_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %257 = allo.stream_construct() {name = "pipe_8_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %258 = allo.stream_construct() {name = "pipe_8_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %259 = allo.stream_construct() {name = "pipe_8_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %260 = allo.stream_construct() {name = "pipe_8_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %261 = allo.stream_construct() {name = "pipe_8_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %262 = allo.stream_construct() {name = "pipe_8_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %263 = allo.stream_construct() {name = "pipe_8_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %264 = allo.stream_construct() {name = "pipe_8_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %265 = allo.stream_construct() {name = "pipe_8_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %266 = allo.stream_construct() {name = "pipe_8_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %267 = allo.stream_construct() {name = "pipe_8_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %268 = allo.stream_construct() {name = "pipe_8_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %269 = allo.stream_construct() {name = "pipe_8_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %270 = allo.stream_construct() {name = "pipe_9_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %271 = allo.stream_construct() {name = "pipe_9_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %272 = allo.stream_construct() {name = "pipe_9_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %273 = allo.stream_construct() {name = "pipe_9_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %274 = allo.stream_construct() {name = "pipe_9_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %275 = allo.stream_construct() {name = "pipe_9_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %276 = allo.stream_construct() {name = "pipe_9_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %277 = allo.stream_construct() {name = "pipe_9_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %278 = allo.stream_construct() {name = "pipe_9_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %279 = allo.stream_construct() {name = "pipe_9_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %280 = allo.stream_construct() {name = "pipe_9_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %281 = allo.stream_construct() {name = "pipe_9_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %282 = allo.stream_construct() {name = "pipe_9_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %283 = allo.stream_construct() {name = "pipe_9_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %284 = allo.stream_construct() {name = "pipe_9_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %285 = allo.stream_construct() {name = "pipe_9_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %286 = allo.stream_construct() {name = "pipe_9_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %287 = allo.stream_construct() {name = "pipe_9_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %288 = allo.stream_construct() {name = "pipe_9_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %289 = allo.stream_construct() {name = "pipe_9_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %290 = allo.stream_construct() {name = "pipe_9_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %291 = allo.stream_construct() {name = "pipe_9_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %292 = allo.stream_construct() {name = "pipe_9_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %293 = allo.stream_construct() {name = "pipe_9_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %294 = allo.stream_construct() {name = "pipe_9_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %295 = allo.stream_construct() {name = "pipe_9_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %296 = allo.stream_construct() {name = "pipe_9_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %297 = allo.stream_construct() {name = "pipe_9_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %298 = allo.stream_construct() {name = "pipe_9_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %299 = allo.stream_construct() {name = "pipe_9_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %300 = allo.stream_construct() {name = "pipe_10_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %301 = allo.stream_construct() {name = "pipe_10_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %302 = allo.stream_construct() {name = "pipe_10_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %303 = allo.stream_construct() {name = "pipe_10_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %304 = allo.stream_construct() {name = "pipe_10_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %305 = allo.stream_construct() {name = "pipe_10_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %306 = allo.stream_construct() {name = "pipe_10_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %307 = allo.stream_construct() {name = "pipe_10_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %308 = allo.stream_construct() {name = "pipe_10_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %309 = allo.stream_construct() {name = "pipe_10_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %310 = allo.stream_construct() {name = "pipe_10_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %311 = allo.stream_construct() {name = "pipe_10_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %312 = allo.stream_construct() {name = "pipe_10_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %313 = allo.stream_construct() {name = "pipe_10_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %314 = allo.stream_construct() {name = "pipe_10_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %315 = allo.stream_construct() {name = "pipe_10_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %316 = allo.stream_construct() {name = "pipe_10_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %317 = allo.stream_construct() {name = "pipe_10_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %318 = allo.stream_construct() {name = "pipe_10_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %319 = allo.stream_construct() {name = "pipe_10_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %320 = allo.stream_construct() {name = "pipe_10_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %321 = allo.stream_construct() {name = "pipe_10_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %322 = allo.stream_construct() {name = "pipe_10_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %323 = allo.stream_construct() {name = "pipe_10_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %324 = allo.stream_construct() {name = "pipe_10_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %325 = allo.stream_construct() {name = "pipe_10_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %326 = allo.stream_construct() {name = "pipe_10_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %327 = allo.stream_construct() {name = "pipe_10_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %328 = allo.stream_construct() {name = "pipe_10_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %329 = allo.stream_construct() {name = "pipe_10_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %330 = allo.stream_construct() {name = "pipe_11_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %331 = allo.stream_construct() {name = "pipe_11_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %332 = allo.stream_construct() {name = "pipe_11_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %333 = allo.stream_construct() {name = "pipe_11_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %334 = allo.stream_construct() {name = "pipe_11_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %335 = allo.stream_construct() {name = "pipe_11_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %336 = allo.stream_construct() {name = "pipe_11_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %337 = allo.stream_construct() {name = "pipe_11_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %338 = allo.stream_construct() {name = "pipe_11_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %339 = allo.stream_construct() {name = "pipe_11_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %340 = allo.stream_construct() {name = "pipe_11_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %341 = allo.stream_construct() {name = "pipe_11_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %342 = allo.stream_construct() {name = "pipe_11_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %343 = allo.stream_construct() {name = "pipe_11_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %344 = allo.stream_construct() {name = "pipe_11_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %345 = allo.stream_construct() {name = "pipe_11_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %346 = allo.stream_construct() {name = "pipe_11_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %347 = allo.stream_construct() {name = "pipe_11_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %348 = allo.stream_construct() {name = "pipe_11_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %349 = allo.stream_construct() {name = "pipe_11_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %350 = allo.stream_construct() {name = "pipe_11_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %351 = allo.stream_construct() {name = "pipe_11_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %352 = allo.stream_construct() {name = "pipe_11_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %353 = allo.stream_construct() {name = "pipe_11_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %354 = allo.stream_construct() {name = "pipe_11_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %355 = allo.stream_construct() {name = "pipe_11_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %356 = allo.stream_construct() {name = "pipe_11_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %357 = allo.stream_construct() {name = "pipe_11_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %358 = allo.stream_construct() {name = "pipe_11_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %359 = allo.stream_construct() {name = "pipe_11_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %360 = allo.stream_construct() {name = "pipe_12_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %361 = allo.stream_construct() {name = "pipe_12_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %362 = allo.stream_construct() {name = "pipe_12_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %363 = allo.stream_construct() {name = "pipe_12_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %364 = allo.stream_construct() {name = "pipe_12_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %365 = allo.stream_construct() {name = "pipe_12_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %366 = allo.stream_construct() {name = "pipe_12_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %367 = allo.stream_construct() {name = "pipe_12_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %368 = allo.stream_construct() {name = "pipe_12_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %369 = allo.stream_construct() {name = "pipe_12_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %370 = allo.stream_construct() {name = "pipe_12_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %371 = allo.stream_construct() {name = "pipe_12_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %372 = allo.stream_construct() {name = "pipe_12_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %373 = allo.stream_construct() {name = "pipe_12_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %374 = allo.stream_construct() {name = "pipe_12_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %375 = allo.stream_construct() {name = "pipe_12_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %376 = allo.stream_construct() {name = "pipe_12_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %377 = allo.stream_construct() {name = "pipe_12_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %378 = allo.stream_construct() {name = "pipe_12_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %379 = allo.stream_construct() {name = "pipe_12_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %380 = allo.stream_construct() {name = "pipe_12_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %381 = allo.stream_construct() {name = "pipe_12_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %382 = allo.stream_construct() {name = "pipe_12_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %383 = allo.stream_construct() {name = "pipe_12_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %384 = allo.stream_construct() {name = "pipe_12_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %385 = allo.stream_construct() {name = "pipe_12_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %386 = allo.stream_construct() {name = "pipe_12_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %387 = allo.stream_construct() {name = "pipe_12_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %388 = allo.stream_construct() {name = "pipe_12_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %389 = allo.stream_construct() {name = "pipe_12_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %390 = allo.stream_construct() {name = "pipe_13_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %391 = allo.stream_construct() {name = "pipe_13_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %392 = allo.stream_construct() {name = "pipe_13_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %393 = allo.stream_construct() {name = "pipe_13_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %394 = allo.stream_construct() {name = "pipe_13_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %395 = allo.stream_construct() {name = "pipe_13_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %396 = allo.stream_construct() {name = "pipe_13_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %397 = allo.stream_construct() {name = "pipe_13_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %398 = allo.stream_construct() {name = "pipe_13_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %399 = allo.stream_construct() {name = "pipe_13_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %400 = allo.stream_construct() {name = "pipe_13_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %401 = allo.stream_construct() {name = "pipe_13_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %402 = allo.stream_construct() {name = "pipe_13_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %403 = allo.stream_construct() {name = "pipe_13_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %404 = allo.stream_construct() {name = "pipe_13_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %405 = allo.stream_construct() {name = "pipe_13_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %406 = allo.stream_construct() {name = "pipe_13_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %407 = allo.stream_construct() {name = "pipe_13_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %408 = allo.stream_construct() {name = "pipe_13_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %409 = allo.stream_construct() {name = "pipe_13_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %410 = allo.stream_construct() {name = "pipe_13_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %411 = allo.stream_construct() {name = "pipe_13_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %412 = allo.stream_construct() {name = "pipe_13_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %413 = allo.stream_construct() {name = "pipe_13_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %414 = allo.stream_construct() {name = "pipe_13_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %415 = allo.stream_construct() {name = "pipe_13_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %416 = allo.stream_construct() {name = "pipe_13_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %417 = allo.stream_construct() {name = "pipe_13_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %418 = allo.stream_construct() {name = "pipe_13_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %419 = allo.stream_construct() {name = "pipe_13_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    return
  }
}
