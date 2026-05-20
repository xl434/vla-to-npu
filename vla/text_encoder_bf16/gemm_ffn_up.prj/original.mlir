module {
  func.func private @fill_zeros_bf16_64_64_vector(memref<64x64xbf16>)
  func.func private @matmul_scalar_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func private @add_bf16_vector(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func @"gemm_0_0_0-gemm_1_0_0-gemm_2_0_0-gemm_3_0_0-gemm_4_0_0-gemm_5_0_0-gemm_6_0_0-gemm_7_0_0-gemm_8_0_0-gemm_9_0_0-gemm_10_0_0-gemm_11_0_0-gemm_12_0_0-gemm_13_0_0-gemm_14_0_0x10"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
    %cst_19 = arith.constant 0.000000e+00 : bf16
    %alloc_20 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_18, %alloc_20 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_21 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_21) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg24, %arg25, %alloc_21) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_22 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_21, %alloc_20, %alloc_22) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_23 = arith.constant 0.000000e+00 : bf16
    %alloc_24 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_22, %alloc_24 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_25 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_25) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg29, %arg30, %alloc_25) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_26 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_25, %alloc_24, %alloc_26) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_27 = arith.constant 0.000000e+00 : bf16
    %alloc_28 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_26, %alloc_28 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_29 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_29) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg34, %arg35, %alloc_29) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_30 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_29, %alloc_28, %alloc_30) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_31 = arith.constant 0.000000e+00 : bf16
    %alloc_32 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_30, %alloc_32 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_33 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_33) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg39, %arg40, %alloc_33) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_34 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_33, %alloc_32, %alloc_34) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_35 = arith.constant 0.000000e+00 : bf16
    %alloc_36 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_34, %alloc_36 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_37 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_37) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg44, %arg45, %alloc_37) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_38 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_37, %alloc_36, %alloc_38) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_39 = arith.constant 0.000000e+00 : bf16
    %alloc_40 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_38, %alloc_40 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_41 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_41) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg49, %arg50, %alloc_41) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_42 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_41, %alloc_40, %alloc_42) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_43 = arith.constant 0.000000e+00 : bf16
    %alloc_44 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_42, %alloc_44 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_45 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_45) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg54, %arg55, %alloc_45) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_46 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_45, %alloc_44, %alloc_46) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_47 = arith.constant 0.000000e+00 : bf16
    %alloc_48 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_46, %alloc_48 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_49 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_49) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg59, %arg60, %alloc_49) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_50 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_49, %alloc_48, %alloc_50) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_51 = arith.constant 0.000000e+00 : bf16
    %alloc_52 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_50, %alloc_52 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_53 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_53) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg64, %arg65, %alloc_53) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_54 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_53, %alloc_52, %alloc_54) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_55 = arith.constant 0.000000e+00 : bf16
    %alloc_56 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_54, %alloc_56 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_57 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_57) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg69, %arg70, %alloc_57) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_58 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_57, %alloc_56, %alloc_58) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_58, %arg71 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @"gemm_0_0_1-gemm_1_0_1-gemm_2_0_1-gemm_3_0_1-gemm_4_0_1-gemm_5_0_1-gemm_6_0_1-gemm_7_0_1-gemm_8_0_1-gemm_9_0_1-gemm_10_0_1-gemm_11_0_1-gemm_12_0_1-gemm_13_0_1-gemm_14_0_1x10"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
    %cst_19 = arith.constant 0.000000e+00 : bf16
    %alloc_20 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_18, %alloc_20 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_21 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_21) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg24, %arg25, %alloc_21) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_22 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_21, %alloc_20, %alloc_22) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_23 = arith.constant 0.000000e+00 : bf16
    %alloc_24 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_22, %alloc_24 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_25 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_25) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg29, %arg30, %alloc_25) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_26 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_25, %alloc_24, %alloc_26) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_27 = arith.constant 0.000000e+00 : bf16
    %alloc_28 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_26, %alloc_28 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_29 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_29) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg34, %arg35, %alloc_29) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_30 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_29, %alloc_28, %alloc_30) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_31 = arith.constant 0.000000e+00 : bf16
    %alloc_32 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_30, %alloc_32 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_33 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_33) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg39, %arg40, %alloc_33) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_34 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_33, %alloc_32, %alloc_34) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_35 = arith.constant 0.000000e+00 : bf16
    %alloc_36 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_34, %alloc_36 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_37 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_37) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg44, %arg45, %alloc_37) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_38 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_37, %alloc_36, %alloc_38) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_39 = arith.constant 0.000000e+00 : bf16
    %alloc_40 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_38, %alloc_40 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_41 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_41) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg49, %arg50, %alloc_41) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_42 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_41, %alloc_40, %alloc_42) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_43 = arith.constant 0.000000e+00 : bf16
    %alloc_44 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_42, %alloc_44 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_45 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_45) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg54, %arg55, %alloc_45) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_46 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_45, %alloc_44, %alloc_46) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_47 = arith.constant 0.000000e+00 : bf16
    %alloc_48 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_46, %alloc_48 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_49 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_49) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg59, %arg60, %alloc_49) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_50 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_49, %alloc_48, %alloc_50) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_51 = arith.constant 0.000000e+00 : bf16
    %alloc_52 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_50, %alloc_52 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_53 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_53) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg64, %arg65, %alloc_53) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_54 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_53, %alloc_52, %alloc_54) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_55 = arith.constant 0.000000e+00 : bf16
    %alloc_56 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_54, %alloc_56 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_57 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_57) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg69, %arg70, %alloc_57) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_58 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_57, %alloc_56, %alloc_58) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_58, %arg71 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @"gemm_0_0_2-gemm_1_0_2-gemm_2_0_2-gemm_3_0_2-gemm_4_0_2-gemm_5_0_2-gemm_6_0_2-gemm_7_0_2-gemm_8_0_2-gemm_9_0_2-gemm_10_0_2-gemm_11_0_2-gemm_12_0_2-gemm_13_0_2-gemm_14_0_2x10"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
    %cst_19 = arith.constant 0.000000e+00 : bf16
    %alloc_20 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_18, %alloc_20 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_21 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_21) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg24, %arg25, %alloc_21) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_22 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_21, %alloc_20, %alloc_22) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_23 = arith.constant 0.000000e+00 : bf16
    %alloc_24 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_22, %alloc_24 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_25 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_25) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg29, %arg30, %alloc_25) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_26 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_25, %alloc_24, %alloc_26) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_27 = arith.constant 0.000000e+00 : bf16
    %alloc_28 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_26, %alloc_28 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_29 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_29) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg34, %arg35, %alloc_29) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_30 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_29, %alloc_28, %alloc_30) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_31 = arith.constant 0.000000e+00 : bf16
    %alloc_32 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_30, %alloc_32 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_33 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_33) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg39, %arg40, %alloc_33) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_34 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_33, %alloc_32, %alloc_34) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_35 = arith.constant 0.000000e+00 : bf16
    %alloc_36 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_34, %alloc_36 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_37 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_37) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg44, %arg45, %alloc_37) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_38 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_37, %alloc_36, %alloc_38) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_39 = arith.constant 0.000000e+00 : bf16
    %alloc_40 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_38, %alloc_40 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_41 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_41) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg49, %arg50, %alloc_41) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_42 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_41, %alloc_40, %alloc_42) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_43 = arith.constant 0.000000e+00 : bf16
    %alloc_44 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_42, %alloc_44 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_45 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_45) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg54, %arg55, %alloc_45) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_46 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_45, %alloc_44, %alloc_46) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_47 = arith.constant 0.000000e+00 : bf16
    %alloc_48 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_46, %alloc_48 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_49 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_49) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg59, %arg60, %alloc_49) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_50 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_49, %alloc_48, %alloc_50) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_51 = arith.constant 0.000000e+00 : bf16
    %alloc_52 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_50, %alloc_52 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_53 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_53) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg64, %arg65, %alloc_53) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_54 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_53, %alloc_52, %alloc_54) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_55 = arith.constant 0.000000e+00 : bf16
    %alloc_56 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_54, %alloc_56 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_57 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_57) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg69, %arg70, %alloc_57) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_58 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_57, %alloc_56, %alloc_58) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_58, %arg71 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @"gemm_0_0_3-gemm_1_0_3-gemm_2_0_3-gemm_3_0_3-gemm_4_0_3-gemm_5_0_3-gemm_6_0_3-gemm_7_0_3-gemm_8_0_3-gemm_9_0_3-gemm_10_0_3-gemm_11_0_3-gemm_12_0_3-gemm_13_0_3-gemm_14_0_3x10"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
    %cst_19 = arith.constant 0.000000e+00 : bf16
    %alloc_20 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_18, %alloc_20 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_21 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_21) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg24, %arg25, %alloc_21) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_22 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_21, %alloc_20, %alloc_22) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_23 = arith.constant 0.000000e+00 : bf16
    %alloc_24 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_22, %alloc_24 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_25 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_25) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg29, %arg30, %alloc_25) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_26 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_25, %alloc_24, %alloc_26) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_27 = arith.constant 0.000000e+00 : bf16
    %alloc_28 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_26, %alloc_28 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_29 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_29) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg34, %arg35, %alloc_29) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_30 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_29, %alloc_28, %alloc_30) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_31 = arith.constant 0.000000e+00 : bf16
    %alloc_32 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_30, %alloc_32 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_33 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_33) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg39, %arg40, %alloc_33) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_34 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_33, %alloc_32, %alloc_34) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_35 = arith.constant 0.000000e+00 : bf16
    %alloc_36 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_34, %alloc_36 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_37 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_37) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg44, %arg45, %alloc_37) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_38 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_37, %alloc_36, %alloc_38) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_39 = arith.constant 0.000000e+00 : bf16
    %alloc_40 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_38, %alloc_40 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_41 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_41) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg49, %arg50, %alloc_41) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_42 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_41, %alloc_40, %alloc_42) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_43 = arith.constant 0.000000e+00 : bf16
    %alloc_44 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_42, %alloc_44 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_45 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_45) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg54, %arg55, %alloc_45) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_46 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_45, %alloc_44, %alloc_46) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_47 = arith.constant 0.000000e+00 : bf16
    %alloc_48 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_46, %alloc_48 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_49 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_49) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg59, %arg60, %alloc_49) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_50 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_49, %alloc_48, %alloc_50) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_51 = arith.constant 0.000000e+00 : bf16
    %alloc_52 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_50, %alloc_52 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_53 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_53) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg64, %arg65, %alloc_53) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_54 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_53, %alloc_52, %alloc_54) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_55 = arith.constant 0.000000e+00 : bf16
    %alloc_56 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_54, %alloc_56 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_57 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_57) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg69, %arg70, %alloc_57) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_58 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_57, %alloc_56, %alloc_58) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_58, %arg71 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @"gemm_0_1_0-gemm_1_1_0-gemm_2_1_0-gemm_3_1_0-gemm_4_1_0-gemm_5_1_0-gemm_6_1_0-gemm_7_1_0-gemm_8_1_0-gemm_9_1_0-gemm_10_1_0-gemm_11_1_0-gemm_12_1_0-gemm_13_1_0-gemm_14_1_0x10"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
    %cst_19 = arith.constant 0.000000e+00 : bf16
    %alloc_20 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_18, %alloc_20 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_21 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_21) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg24, %arg25, %alloc_21) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_22 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_21, %alloc_20, %alloc_22) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_23 = arith.constant 0.000000e+00 : bf16
    %alloc_24 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_22, %alloc_24 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_25 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_25) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg29, %arg30, %alloc_25) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_26 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_25, %alloc_24, %alloc_26) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_27 = arith.constant 0.000000e+00 : bf16
    %alloc_28 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_26, %alloc_28 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_29 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_29) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg34, %arg35, %alloc_29) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_30 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_29, %alloc_28, %alloc_30) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_31 = arith.constant 0.000000e+00 : bf16
    %alloc_32 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_30, %alloc_32 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_33 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_33) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg39, %arg40, %alloc_33) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_34 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_33, %alloc_32, %alloc_34) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_35 = arith.constant 0.000000e+00 : bf16
    %alloc_36 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_34, %alloc_36 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_37 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_37) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg44, %arg45, %alloc_37) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_38 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_37, %alloc_36, %alloc_38) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_39 = arith.constant 0.000000e+00 : bf16
    %alloc_40 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_38, %alloc_40 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_41 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_41) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg49, %arg50, %alloc_41) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_42 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_41, %alloc_40, %alloc_42) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_43 = arith.constant 0.000000e+00 : bf16
    %alloc_44 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_42, %alloc_44 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_45 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_45) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg54, %arg55, %alloc_45) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_46 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_45, %alloc_44, %alloc_46) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_47 = arith.constant 0.000000e+00 : bf16
    %alloc_48 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_46, %alloc_48 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_49 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_49) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg59, %arg60, %alloc_49) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_50 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_49, %alloc_48, %alloc_50) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_51 = arith.constant 0.000000e+00 : bf16
    %alloc_52 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_50, %alloc_52 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_53 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_53) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg64, %arg65, %alloc_53) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_54 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_53, %alloc_52, %alloc_54) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_55 = arith.constant 0.000000e+00 : bf16
    %alloc_56 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_54, %alloc_56 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_57 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_57) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg69, %arg70, %alloc_57) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_58 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_57, %alloc_56, %alloc_58) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_58, %arg71 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @"gemm_0_1_1-gemm_1_1_1-gemm_2_1_1-gemm_3_1_1-gemm_4_1_1-gemm_5_1_1-gemm_6_1_1-gemm_7_1_1-gemm_8_1_1-gemm_9_1_1-gemm_10_1_1-gemm_11_1_1-gemm_12_1_1-gemm_13_1_1-gemm_14_1_1x10"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
    %cst_19 = arith.constant 0.000000e+00 : bf16
    %alloc_20 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_18, %alloc_20 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_21 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_21) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg24, %arg25, %alloc_21) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_22 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_21, %alloc_20, %alloc_22) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_23 = arith.constant 0.000000e+00 : bf16
    %alloc_24 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_22, %alloc_24 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_25 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_25) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg29, %arg30, %alloc_25) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_26 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_25, %alloc_24, %alloc_26) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_27 = arith.constant 0.000000e+00 : bf16
    %alloc_28 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_26, %alloc_28 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_29 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_29) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg34, %arg35, %alloc_29) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_30 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_29, %alloc_28, %alloc_30) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_31 = arith.constant 0.000000e+00 : bf16
    %alloc_32 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_30, %alloc_32 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_33 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_33) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg39, %arg40, %alloc_33) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_34 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_33, %alloc_32, %alloc_34) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_35 = arith.constant 0.000000e+00 : bf16
    %alloc_36 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_34, %alloc_36 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_37 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_37) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg44, %arg45, %alloc_37) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_38 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_37, %alloc_36, %alloc_38) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_39 = arith.constant 0.000000e+00 : bf16
    %alloc_40 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_38, %alloc_40 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_41 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_41) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg49, %arg50, %alloc_41) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_42 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_41, %alloc_40, %alloc_42) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_43 = arith.constant 0.000000e+00 : bf16
    %alloc_44 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_42, %alloc_44 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_45 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_45) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg54, %arg55, %alloc_45) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_46 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_45, %alloc_44, %alloc_46) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_47 = arith.constant 0.000000e+00 : bf16
    %alloc_48 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_46, %alloc_48 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_49 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_49) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg59, %arg60, %alloc_49) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_50 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_49, %alloc_48, %alloc_50) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_51 = arith.constant 0.000000e+00 : bf16
    %alloc_52 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_50, %alloc_52 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_53 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_53) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg64, %arg65, %alloc_53) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_54 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_53, %alloc_52, %alloc_54) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_55 = arith.constant 0.000000e+00 : bf16
    %alloc_56 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_54, %alloc_56 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_57 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_57) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg69, %arg70, %alloc_57) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_58 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_57, %alloc_56, %alloc_58) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_58, %arg71 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @"gemm_0_1_2-gemm_1_1_2-gemm_2_1_2-gemm_3_1_2-gemm_4_1_2-gemm_5_1_2-gemm_6_1_2-gemm_7_1_2-gemm_8_1_2-gemm_9_1_2-gemm_10_1_2-gemm_11_1_2-gemm_12_1_2-gemm_13_1_2-gemm_14_1_2x10"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
    %cst_19 = arith.constant 0.000000e+00 : bf16
    %alloc_20 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_18, %alloc_20 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_21 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_21) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg24, %arg25, %alloc_21) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_22 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_21, %alloc_20, %alloc_22) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_23 = arith.constant 0.000000e+00 : bf16
    %alloc_24 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_22, %alloc_24 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_25 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_25) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg29, %arg30, %alloc_25) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_26 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_25, %alloc_24, %alloc_26) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_27 = arith.constant 0.000000e+00 : bf16
    %alloc_28 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_26, %alloc_28 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_29 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_29) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg34, %arg35, %alloc_29) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_30 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_29, %alloc_28, %alloc_30) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_31 = arith.constant 0.000000e+00 : bf16
    %alloc_32 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_30, %alloc_32 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_33 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_33) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg39, %arg40, %alloc_33) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_34 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_33, %alloc_32, %alloc_34) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_35 = arith.constant 0.000000e+00 : bf16
    %alloc_36 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_34, %alloc_36 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_37 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_37) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg44, %arg45, %alloc_37) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_38 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_37, %alloc_36, %alloc_38) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_39 = arith.constant 0.000000e+00 : bf16
    %alloc_40 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_38, %alloc_40 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_41 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_41) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg49, %arg50, %alloc_41) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_42 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_41, %alloc_40, %alloc_42) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_43 = arith.constant 0.000000e+00 : bf16
    %alloc_44 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_42, %alloc_44 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_45 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_45) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg54, %arg55, %alloc_45) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_46 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_45, %alloc_44, %alloc_46) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_47 = arith.constant 0.000000e+00 : bf16
    %alloc_48 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_46, %alloc_48 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_49 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_49) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg59, %arg60, %alloc_49) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_50 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_49, %alloc_48, %alloc_50) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_51 = arith.constant 0.000000e+00 : bf16
    %alloc_52 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_50, %alloc_52 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_53 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_53) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg64, %arg65, %alloc_53) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_54 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_53, %alloc_52, %alloc_54) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_55 = arith.constant 0.000000e+00 : bf16
    %alloc_56 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_54, %alloc_56 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_57 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_57) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg69, %arg70, %alloc_57) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_58 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_57, %alloc_56, %alloc_58) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_58, %arg71 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @"gemm_0_1_3-gemm_1_1_3-gemm_2_1_3-gemm_3_1_3-gemm_4_1_3-gemm_5_1_3-gemm_6_1_3-gemm_7_1_3-gemm_8_1_3-gemm_9_1_3-gemm_10_1_3-gemm_11_1_3-gemm_12_1_3-gemm_13_1_3-gemm_14_1_3x10"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
    %cst_19 = arith.constant 0.000000e+00 : bf16
    %alloc_20 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_18, %alloc_20 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_21 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_21) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg24, %arg25, %alloc_21) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_22 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_21, %alloc_20, %alloc_22) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_23 = arith.constant 0.000000e+00 : bf16
    %alloc_24 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_22, %alloc_24 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_25 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_25) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg29, %arg30, %alloc_25) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_26 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_25, %alloc_24, %alloc_26) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_27 = arith.constant 0.000000e+00 : bf16
    %alloc_28 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_26, %alloc_28 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_29 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_29) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg34, %arg35, %alloc_29) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_30 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_29, %alloc_28, %alloc_30) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_31 = arith.constant 0.000000e+00 : bf16
    %alloc_32 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_30, %alloc_32 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_33 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_33) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg39, %arg40, %alloc_33) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_34 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_33, %alloc_32, %alloc_34) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_35 = arith.constant 0.000000e+00 : bf16
    %alloc_36 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_34, %alloc_36 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_37 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_37) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg44, %arg45, %alloc_37) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_38 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_37, %alloc_36, %alloc_38) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_39 = arith.constant 0.000000e+00 : bf16
    %alloc_40 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_38, %alloc_40 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_41 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_41) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg49, %arg50, %alloc_41) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_42 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_41, %alloc_40, %alloc_42) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_43 = arith.constant 0.000000e+00 : bf16
    %alloc_44 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_42, %alloc_44 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_45 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_45) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg54, %arg55, %alloc_45) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_46 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_45, %alloc_44, %alloc_46) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_47 = arith.constant 0.000000e+00 : bf16
    %alloc_48 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_46, %alloc_48 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_49 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_49) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg59, %arg60, %alloc_49) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_50 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_49, %alloc_48, %alloc_50) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_51 = arith.constant 0.000000e+00 : bf16
    %alloc_52 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_50, %alloc_52 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_53 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_53) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg64, %arg65, %alloc_53) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_54 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_53, %alloc_52, %alloc_54) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %cst_55 = arith.constant 0.000000e+00 : bf16
    %alloc_56 = memref.alloc() {name = "C_in"} : memref<64x64xbf16>
    memref.copy %alloc_54, %alloc_56 {to = "C_in"} : memref<64x64xbf16> to memref<64x64xbf16>
    %alloc_57 = memref.alloc() : memref<64x64xbf16>
    call @fill_zeros_bf16_64_64_vector(%alloc_57) {lib = "fill_zeros_bf16_64_64_vector"} : (memref<64x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg69, %arg70, %alloc_57) {lib = "matmul_scalar_bf16_bf16"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    %alloc_58 = memref.alloc() {name = "C_out"} : memref<64x64xbf16>
    call @add_bf16_vector(%alloc_57, %alloc_56, %alloc_58) {lib = "add_bf16_vector"} : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    memref.copy %alloc_58, %arg71 {to = "local_C"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @top(%arg0: memref<128x960xbf16>, %arg1: memref<960x2560xbf16>, %arg2: memref<128x2560xbf16>) attributes {dataflow, itypes = "___"} {
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
    %15 = allo.stream_construct() {name = "pipe_0_0_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %16 = allo.stream_construct() {name = "pipe_0_0_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %17 = allo.stream_construct() {name = "pipe_0_0_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %18 = allo.stream_construct() {name = "pipe_0_0_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %19 = allo.stream_construct() {name = "pipe_0_0_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %20 = allo.stream_construct() {name = "pipe_0_0_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %21 = allo.stream_construct() {name = "pipe_0_0_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %22 = allo.stream_construct() {name = "pipe_0_0_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %23 = allo.stream_construct() {name = "pipe_0_0_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %24 = allo.stream_construct() {name = "pipe_0_0_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %25 = allo.stream_construct() {name = "pipe_0_0_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %26 = allo.stream_construct() {name = "pipe_0_0_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %27 = allo.stream_construct() {name = "pipe_0_0_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %28 = allo.stream_construct() {name = "pipe_0_0_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %29 = allo.stream_construct() {name = "pipe_0_0_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %30 = allo.stream_construct() {name = "pipe_0_0_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %31 = allo.stream_construct() {name = "pipe_0_0_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %32 = allo.stream_construct() {name = "pipe_0_0_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %33 = allo.stream_construct() {name = "pipe_0_0_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %34 = allo.stream_construct() {name = "pipe_0_0_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %35 = allo.stream_construct() {name = "pipe_0_0_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %36 = allo.stream_construct() {name = "pipe_0_0_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %37 = allo.stream_construct() {name = "pipe_0_0_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %38 = allo.stream_construct() {name = "pipe_0_0_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %39 = allo.stream_construct() {name = "pipe_0_0_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %40 = allo.stream_construct() {name = "pipe_0_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %41 = allo.stream_construct() {name = "pipe_0_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %42 = allo.stream_construct() {name = "pipe_0_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %43 = allo.stream_construct() {name = "pipe_0_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %44 = allo.stream_construct() {name = "pipe_0_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %45 = allo.stream_construct() {name = "pipe_0_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %46 = allo.stream_construct() {name = "pipe_0_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %47 = allo.stream_construct() {name = "pipe_0_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %48 = allo.stream_construct() {name = "pipe_0_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %49 = allo.stream_construct() {name = "pipe_0_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %50 = allo.stream_construct() {name = "pipe_0_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %51 = allo.stream_construct() {name = "pipe_0_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %52 = allo.stream_construct() {name = "pipe_0_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %53 = allo.stream_construct() {name = "pipe_0_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %54 = allo.stream_construct() {name = "pipe_0_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %55 = allo.stream_construct() {name = "pipe_0_1_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %56 = allo.stream_construct() {name = "pipe_0_1_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %57 = allo.stream_construct() {name = "pipe_0_1_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %58 = allo.stream_construct() {name = "pipe_0_1_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %59 = allo.stream_construct() {name = "pipe_0_1_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %60 = allo.stream_construct() {name = "pipe_0_1_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %61 = allo.stream_construct() {name = "pipe_0_1_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %62 = allo.stream_construct() {name = "pipe_0_1_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %63 = allo.stream_construct() {name = "pipe_0_1_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %64 = allo.stream_construct() {name = "pipe_0_1_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %65 = allo.stream_construct() {name = "pipe_0_1_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %66 = allo.stream_construct() {name = "pipe_0_1_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %67 = allo.stream_construct() {name = "pipe_0_1_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %68 = allo.stream_construct() {name = "pipe_0_1_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %69 = allo.stream_construct() {name = "pipe_0_1_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %70 = allo.stream_construct() {name = "pipe_0_1_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %71 = allo.stream_construct() {name = "pipe_0_1_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %72 = allo.stream_construct() {name = "pipe_0_1_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %73 = allo.stream_construct() {name = "pipe_0_1_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %74 = allo.stream_construct() {name = "pipe_0_1_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %75 = allo.stream_construct() {name = "pipe_0_1_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %76 = allo.stream_construct() {name = "pipe_0_1_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %77 = allo.stream_construct() {name = "pipe_0_1_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %78 = allo.stream_construct() {name = "pipe_0_1_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %79 = allo.stream_construct() {name = "pipe_0_1_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %80 = allo.stream_construct() {name = "pipe_1_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %81 = allo.stream_construct() {name = "pipe_1_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %82 = allo.stream_construct() {name = "pipe_1_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %83 = allo.stream_construct() {name = "pipe_1_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %84 = allo.stream_construct() {name = "pipe_1_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %85 = allo.stream_construct() {name = "pipe_1_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %86 = allo.stream_construct() {name = "pipe_1_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %87 = allo.stream_construct() {name = "pipe_1_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %88 = allo.stream_construct() {name = "pipe_1_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %89 = allo.stream_construct() {name = "pipe_1_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %90 = allo.stream_construct() {name = "pipe_1_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %91 = allo.stream_construct() {name = "pipe_1_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %92 = allo.stream_construct() {name = "pipe_1_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %93 = allo.stream_construct() {name = "pipe_1_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %94 = allo.stream_construct() {name = "pipe_1_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %95 = allo.stream_construct() {name = "pipe_1_0_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %96 = allo.stream_construct() {name = "pipe_1_0_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %97 = allo.stream_construct() {name = "pipe_1_0_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %98 = allo.stream_construct() {name = "pipe_1_0_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %99 = allo.stream_construct() {name = "pipe_1_0_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %100 = allo.stream_construct() {name = "pipe_1_0_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %101 = allo.stream_construct() {name = "pipe_1_0_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %102 = allo.stream_construct() {name = "pipe_1_0_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %103 = allo.stream_construct() {name = "pipe_1_0_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %104 = allo.stream_construct() {name = "pipe_1_0_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %105 = allo.stream_construct() {name = "pipe_1_0_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %106 = allo.stream_construct() {name = "pipe_1_0_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %107 = allo.stream_construct() {name = "pipe_1_0_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %108 = allo.stream_construct() {name = "pipe_1_0_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %109 = allo.stream_construct() {name = "pipe_1_0_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %110 = allo.stream_construct() {name = "pipe_1_0_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %111 = allo.stream_construct() {name = "pipe_1_0_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %112 = allo.stream_construct() {name = "pipe_1_0_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %113 = allo.stream_construct() {name = "pipe_1_0_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %114 = allo.stream_construct() {name = "pipe_1_0_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %115 = allo.stream_construct() {name = "pipe_1_0_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %116 = allo.stream_construct() {name = "pipe_1_0_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %117 = allo.stream_construct() {name = "pipe_1_0_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %118 = allo.stream_construct() {name = "pipe_1_0_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %119 = allo.stream_construct() {name = "pipe_1_0_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %120 = allo.stream_construct() {name = "pipe_1_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %121 = allo.stream_construct() {name = "pipe_1_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %122 = allo.stream_construct() {name = "pipe_1_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %123 = allo.stream_construct() {name = "pipe_1_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %124 = allo.stream_construct() {name = "pipe_1_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %125 = allo.stream_construct() {name = "pipe_1_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %126 = allo.stream_construct() {name = "pipe_1_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %127 = allo.stream_construct() {name = "pipe_1_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %128 = allo.stream_construct() {name = "pipe_1_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %129 = allo.stream_construct() {name = "pipe_1_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %130 = allo.stream_construct() {name = "pipe_1_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %131 = allo.stream_construct() {name = "pipe_1_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %132 = allo.stream_construct() {name = "pipe_1_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %133 = allo.stream_construct() {name = "pipe_1_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %134 = allo.stream_construct() {name = "pipe_1_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %135 = allo.stream_construct() {name = "pipe_1_1_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %136 = allo.stream_construct() {name = "pipe_1_1_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %137 = allo.stream_construct() {name = "pipe_1_1_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %138 = allo.stream_construct() {name = "pipe_1_1_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %139 = allo.stream_construct() {name = "pipe_1_1_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %140 = allo.stream_construct() {name = "pipe_1_1_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %141 = allo.stream_construct() {name = "pipe_1_1_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %142 = allo.stream_construct() {name = "pipe_1_1_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %143 = allo.stream_construct() {name = "pipe_1_1_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %144 = allo.stream_construct() {name = "pipe_1_1_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %145 = allo.stream_construct() {name = "pipe_1_1_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %146 = allo.stream_construct() {name = "pipe_1_1_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %147 = allo.stream_construct() {name = "pipe_1_1_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %148 = allo.stream_construct() {name = "pipe_1_1_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %149 = allo.stream_construct() {name = "pipe_1_1_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %150 = allo.stream_construct() {name = "pipe_1_1_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %151 = allo.stream_construct() {name = "pipe_1_1_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %152 = allo.stream_construct() {name = "pipe_1_1_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %153 = allo.stream_construct() {name = "pipe_1_1_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %154 = allo.stream_construct() {name = "pipe_1_1_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %155 = allo.stream_construct() {name = "pipe_1_1_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %156 = allo.stream_construct() {name = "pipe_1_1_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %157 = allo.stream_construct() {name = "pipe_1_1_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %158 = allo.stream_construct() {name = "pipe_1_1_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %159 = allo.stream_construct() {name = "pipe_1_1_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %160 = allo.stream_construct() {name = "pipe_2_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %161 = allo.stream_construct() {name = "pipe_2_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %162 = allo.stream_construct() {name = "pipe_2_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %163 = allo.stream_construct() {name = "pipe_2_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %164 = allo.stream_construct() {name = "pipe_2_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %165 = allo.stream_construct() {name = "pipe_2_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %166 = allo.stream_construct() {name = "pipe_2_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %167 = allo.stream_construct() {name = "pipe_2_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %168 = allo.stream_construct() {name = "pipe_2_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %169 = allo.stream_construct() {name = "pipe_2_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %170 = allo.stream_construct() {name = "pipe_2_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %171 = allo.stream_construct() {name = "pipe_2_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %172 = allo.stream_construct() {name = "pipe_2_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %173 = allo.stream_construct() {name = "pipe_2_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %174 = allo.stream_construct() {name = "pipe_2_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %175 = allo.stream_construct() {name = "pipe_2_0_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %176 = allo.stream_construct() {name = "pipe_2_0_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %177 = allo.stream_construct() {name = "pipe_2_0_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %178 = allo.stream_construct() {name = "pipe_2_0_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %179 = allo.stream_construct() {name = "pipe_2_0_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %180 = allo.stream_construct() {name = "pipe_2_0_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %181 = allo.stream_construct() {name = "pipe_2_0_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %182 = allo.stream_construct() {name = "pipe_2_0_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %183 = allo.stream_construct() {name = "pipe_2_0_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %184 = allo.stream_construct() {name = "pipe_2_0_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %185 = allo.stream_construct() {name = "pipe_2_0_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %186 = allo.stream_construct() {name = "pipe_2_0_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %187 = allo.stream_construct() {name = "pipe_2_0_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %188 = allo.stream_construct() {name = "pipe_2_0_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %189 = allo.stream_construct() {name = "pipe_2_0_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %190 = allo.stream_construct() {name = "pipe_2_0_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %191 = allo.stream_construct() {name = "pipe_2_0_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %192 = allo.stream_construct() {name = "pipe_2_0_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %193 = allo.stream_construct() {name = "pipe_2_0_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %194 = allo.stream_construct() {name = "pipe_2_0_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %195 = allo.stream_construct() {name = "pipe_2_0_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %196 = allo.stream_construct() {name = "pipe_2_0_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %197 = allo.stream_construct() {name = "pipe_2_0_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %198 = allo.stream_construct() {name = "pipe_2_0_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %199 = allo.stream_construct() {name = "pipe_2_0_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %200 = allo.stream_construct() {name = "pipe_2_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %201 = allo.stream_construct() {name = "pipe_2_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %202 = allo.stream_construct() {name = "pipe_2_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %203 = allo.stream_construct() {name = "pipe_2_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %204 = allo.stream_construct() {name = "pipe_2_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %205 = allo.stream_construct() {name = "pipe_2_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %206 = allo.stream_construct() {name = "pipe_2_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %207 = allo.stream_construct() {name = "pipe_2_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %208 = allo.stream_construct() {name = "pipe_2_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %209 = allo.stream_construct() {name = "pipe_2_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %210 = allo.stream_construct() {name = "pipe_2_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %211 = allo.stream_construct() {name = "pipe_2_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %212 = allo.stream_construct() {name = "pipe_2_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %213 = allo.stream_construct() {name = "pipe_2_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %214 = allo.stream_construct() {name = "pipe_2_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %215 = allo.stream_construct() {name = "pipe_2_1_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %216 = allo.stream_construct() {name = "pipe_2_1_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %217 = allo.stream_construct() {name = "pipe_2_1_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %218 = allo.stream_construct() {name = "pipe_2_1_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %219 = allo.stream_construct() {name = "pipe_2_1_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %220 = allo.stream_construct() {name = "pipe_2_1_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %221 = allo.stream_construct() {name = "pipe_2_1_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %222 = allo.stream_construct() {name = "pipe_2_1_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %223 = allo.stream_construct() {name = "pipe_2_1_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %224 = allo.stream_construct() {name = "pipe_2_1_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %225 = allo.stream_construct() {name = "pipe_2_1_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %226 = allo.stream_construct() {name = "pipe_2_1_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %227 = allo.stream_construct() {name = "pipe_2_1_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %228 = allo.stream_construct() {name = "pipe_2_1_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %229 = allo.stream_construct() {name = "pipe_2_1_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %230 = allo.stream_construct() {name = "pipe_2_1_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %231 = allo.stream_construct() {name = "pipe_2_1_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %232 = allo.stream_construct() {name = "pipe_2_1_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %233 = allo.stream_construct() {name = "pipe_2_1_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %234 = allo.stream_construct() {name = "pipe_2_1_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %235 = allo.stream_construct() {name = "pipe_2_1_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %236 = allo.stream_construct() {name = "pipe_2_1_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %237 = allo.stream_construct() {name = "pipe_2_1_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %238 = allo.stream_construct() {name = "pipe_2_1_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %239 = allo.stream_construct() {name = "pipe_2_1_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %240 = allo.stream_construct() {name = "pipe_3_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %241 = allo.stream_construct() {name = "pipe_3_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %242 = allo.stream_construct() {name = "pipe_3_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %243 = allo.stream_construct() {name = "pipe_3_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %244 = allo.stream_construct() {name = "pipe_3_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %245 = allo.stream_construct() {name = "pipe_3_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %246 = allo.stream_construct() {name = "pipe_3_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %247 = allo.stream_construct() {name = "pipe_3_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %248 = allo.stream_construct() {name = "pipe_3_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %249 = allo.stream_construct() {name = "pipe_3_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %250 = allo.stream_construct() {name = "pipe_3_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %251 = allo.stream_construct() {name = "pipe_3_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %252 = allo.stream_construct() {name = "pipe_3_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %253 = allo.stream_construct() {name = "pipe_3_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %254 = allo.stream_construct() {name = "pipe_3_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %255 = allo.stream_construct() {name = "pipe_3_0_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %256 = allo.stream_construct() {name = "pipe_3_0_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %257 = allo.stream_construct() {name = "pipe_3_0_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %258 = allo.stream_construct() {name = "pipe_3_0_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %259 = allo.stream_construct() {name = "pipe_3_0_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %260 = allo.stream_construct() {name = "pipe_3_0_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %261 = allo.stream_construct() {name = "pipe_3_0_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %262 = allo.stream_construct() {name = "pipe_3_0_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %263 = allo.stream_construct() {name = "pipe_3_0_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %264 = allo.stream_construct() {name = "pipe_3_0_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %265 = allo.stream_construct() {name = "pipe_3_0_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %266 = allo.stream_construct() {name = "pipe_3_0_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %267 = allo.stream_construct() {name = "pipe_3_0_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %268 = allo.stream_construct() {name = "pipe_3_0_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %269 = allo.stream_construct() {name = "pipe_3_0_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %270 = allo.stream_construct() {name = "pipe_3_0_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %271 = allo.stream_construct() {name = "pipe_3_0_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %272 = allo.stream_construct() {name = "pipe_3_0_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %273 = allo.stream_construct() {name = "pipe_3_0_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %274 = allo.stream_construct() {name = "pipe_3_0_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %275 = allo.stream_construct() {name = "pipe_3_0_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %276 = allo.stream_construct() {name = "pipe_3_0_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %277 = allo.stream_construct() {name = "pipe_3_0_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %278 = allo.stream_construct() {name = "pipe_3_0_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %279 = allo.stream_construct() {name = "pipe_3_0_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %280 = allo.stream_construct() {name = "pipe_3_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %281 = allo.stream_construct() {name = "pipe_3_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %282 = allo.stream_construct() {name = "pipe_3_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %283 = allo.stream_construct() {name = "pipe_3_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %284 = allo.stream_construct() {name = "pipe_3_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %285 = allo.stream_construct() {name = "pipe_3_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %286 = allo.stream_construct() {name = "pipe_3_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %287 = allo.stream_construct() {name = "pipe_3_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %288 = allo.stream_construct() {name = "pipe_3_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %289 = allo.stream_construct() {name = "pipe_3_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %290 = allo.stream_construct() {name = "pipe_3_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %291 = allo.stream_construct() {name = "pipe_3_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %292 = allo.stream_construct() {name = "pipe_3_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %293 = allo.stream_construct() {name = "pipe_3_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %294 = allo.stream_construct() {name = "pipe_3_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %295 = allo.stream_construct() {name = "pipe_3_1_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %296 = allo.stream_construct() {name = "pipe_3_1_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %297 = allo.stream_construct() {name = "pipe_3_1_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %298 = allo.stream_construct() {name = "pipe_3_1_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %299 = allo.stream_construct() {name = "pipe_3_1_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %300 = allo.stream_construct() {name = "pipe_3_1_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %301 = allo.stream_construct() {name = "pipe_3_1_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %302 = allo.stream_construct() {name = "pipe_3_1_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %303 = allo.stream_construct() {name = "pipe_3_1_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %304 = allo.stream_construct() {name = "pipe_3_1_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %305 = allo.stream_construct() {name = "pipe_3_1_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %306 = allo.stream_construct() {name = "pipe_3_1_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %307 = allo.stream_construct() {name = "pipe_3_1_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %308 = allo.stream_construct() {name = "pipe_3_1_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %309 = allo.stream_construct() {name = "pipe_3_1_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %310 = allo.stream_construct() {name = "pipe_3_1_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %311 = allo.stream_construct() {name = "pipe_3_1_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %312 = allo.stream_construct() {name = "pipe_3_1_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %313 = allo.stream_construct() {name = "pipe_3_1_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %314 = allo.stream_construct() {name = "pipe_3_1_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %315 = allo.stream_construct() {name = "pipe_3_1_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %316 = allo.stream_construct() {name = "pipe_3_1_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %317 = allo.stream_construct() {name = "pipe_3_1_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %318 = allo.stream_construct() {name = "pipe_3_1_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %319 = allo.stream_construct() {name = "pipe_3_1_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %320 = allo.stream_construct() {name = "pipe_4_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %321 = allo.stream_construct() {name = "pipe_4_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %322 = allo.stream_construct() {name = "pipe_4_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %323 = allo.stream_construct() {name = "pipe_4_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %324 = allo.stream_construct() {name = "pipe_4_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %325 = allo.stream_construct() {name = "pipe_4_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %326 = allo.stream_construct() {name = "pipe_4_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %327 = allo.stream_construct() {name = "pipe_4_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %328 = allo.stream_construct() {name = "pipe_4_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %329 = allo.stream_construct() {name = "pipe_4_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %330 = allo.stream_construct() {name = "pipe_4_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %331 = allo.stream_construct() {name = "pipe_4_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %332 = allo.stream_construct() {name = "pipe_4_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %333 = allo.stream_construct() {name = "pipe_4_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %334 = allo.stream_construct() {name = "pipe_4_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %335 = allo.stream_construct() {name = "pipe_4_0_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %336 = allo.stream_construct() {name = "pipe_4_0_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %337 = allo.stream_construct() {name = "pipe_4_0_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %338 = allo.stream_construct() {name = "pipe_4_0_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %339 = allo.stream_construct() {name = "pipe_4_0_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %340 = allo.stream_construct() {name = "pipe_4_0_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %341 = allo.stream_construct() {name = "pipe_4_0_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %342 = allo.stream_construct() {name = "pipe_4_0_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %343 = allo.stream_construct() {name = "pipe_4_0_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %344 = allo.stream_construct() {name = "pipe_4_0_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %345 = allo.stream_construct() {name = "pipe_4_0_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %346 = allo.stream_construct() {name = "pipe_4_0_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %347 = allo.stream_construct() {name = "pipe_4_0_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %348 = allo.stream_construct() {name = "pipe_4_0_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %349 = allo.stream_construct() {name = "pipe_4_0_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %350 = allo.stream_construct() {name = "pipe_4_0_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %351 = allo.stream_construct() {name = "pipe_4_0_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %352 = allo.stream_construct() {name = "pipe_4_0_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %353 = allo.stream_construct() {name = "pipe_4_0_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %354 = allo.stream_construct() {name = "pipe_4_0_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %355 = allo.stream_construct() {name = "pipe_4_0_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %356 = allo.stream_construct() {name = "pipe_4_0_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %357 = allo.stream_construct() {name = "pipe_4_0_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %358 = allo.stream_construct() {name = "pipe_4_0_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %359 = allo.stream_construct() {name = "pipe_4_0_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %360 = allo.stream_construct() {name = "pipe_4_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %361 = allo.stream_construct() {name = "pipe_4_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %362 = allo.stream_construct() {name = "pipe_4_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %363 = allo.stream_construct() {name = "pipe_4_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %364 = allo.stream_construct() {name = "pipe_4_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %365 = allo.stream_construct() {name = "pipe_4_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %366 = allo.stream_construct() {name = "pipe_4_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %367 = allo.stream_construct() {name = "pipe_4_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %368 = allo.stream_construct() {name = "pipe_4_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %369 = allo.stream_construct() {name = "pipe_4_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %370 = allo.stream_construct() {name = "pipe_4_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %371 = allo.stream_construct() {name = "pipe_4_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %372 = allo.stream_construct() {name = "pipe_4_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %373 = allo.stream_construct() {name = "pipe_4_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %374 = allo.stream_construct() {name = "pipe_4_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %375 = allo.stream_construct() {name = "pipe_4_1_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %376 = allo.stream_construct() {name = "pipe_4_1_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %377 = allo.stream_construct() {name = "pipe_4_1_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %378 = allo.stream_construct() {name = "pipe_4_1_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %379 = allo.stream_construct() {name = "pipe_4_1_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %380 = allo.stream_construct() {name = "pipe_4_1_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %381 = allo.stream_construct() {name = "pipe_4_1_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %382 = allo.stream_construct() {name = "pipe_4_1_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %383 = allo.stream_construct() {name = "pipe_4_1_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %384 = allo.stream_construct() {name = "pipe_4_1_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %385 = allo.stream_construct() {name = "pipe_4_1_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %386 = allo.stream_construct() {name = "pipe_4_1_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %387 = allo.stream_construct() {name = "pipe_4_1_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %388 = allo.stream_construct() {name = "pipe_4_1_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %389 = allo.stream_construct() {name = "pipe_4_1_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %390 = allo.stream_construct() {name = "pipe_4_1_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %391 = allo.stream_construct() {name = "pipe_4_1_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %392 = allo.stream_construct() {name = "pipe_4_1_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %393 = allo.stream_construct() {name = "pipe_4_1_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %394 = allo.stream_construct() {name = "pipe_4_1_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %395 = allo.stream_construct() {name = "pipe_4_1_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %396 = allo.stream_construct() {name = "pipe_4_1_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %397 = allo.stream_construct() {name = "pipe_4_1_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %398 = allo.stream_construct() {name = "pipe_4_1_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %399 = allo.stream_construct() {name = "pipe_4_1_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %400 = allo.stream_construct() {name = "pipe_5_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %401 = allo.stream_construct() {name = "pipe_5_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %402 = allo.stream_construct() {name = "pipe_5_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %403 = allo.stream_construct() {name = "pipe_5_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %404 = allo.stream_construct() {name = "pipe_5_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %405 = allo.stream_construct() {name = "pipe_5_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %406 = allo.stream_construct() {name = "pipe_5_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %407 = allo.stream_construct() {name = "pipe_5_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %408 = allo.stream_construct() {name = "pipe_5_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %409 = allo.stream_construct() {name = "pipe_5_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %410 = allo.stream_construct() {name = "pipe_5_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %411 = allo.stream_construct() {name = "pipe_5_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %412 = allo.stream_construct() {name = "pipe_5_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %413 = allo.stream_construct() {name = "pipe_5_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %414 = allo.stream_construct() {name = "pipe_5_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %415 = allo.stream_construct() {name = "pipe_5_0_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %416 = allo.stream_construct() {name = "pipe_5_0_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %417 = allo.stream_construct() {name = "pipe_5_0_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %418 = allo.stream_construct() {name = "pipe_5_0_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %419 = allo.stream_construct() {name = "pipe_5_0_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %420 = allo.stream_construct() {name = "pipe_5_0_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %421 = allo.stream_construct() {name = "pipe_5_0_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %422 = allo.stream_construct() {name = "pipe_5_0_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %423 = allo.stream_construct() {name = "pipe_5_0_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %424 = allo.stream_construct() {name = "pipe_5_0_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %425 = allo.stream_construct() {name = "pipe_5_0_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %426 = allo.stream_construct() {name = "pipe_5_0_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %427 = allo.stream_construct() {name = "pipe_5_0_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %428 = allo.stream_construct() {name = "pipe_5_0_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %429 = allo.stream_construct() {name = "pipe_5_0_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %430 = allo.stream_construct() {name = "pipe_5_0_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %431 = allo.stream_construct() {name = "pipe_5_0_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %432 = allo.stream_construct() {name = "pipe_5_0_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %433 = allo.stream_construct() {name = "pipe_5_0_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %434 = allo.stream_construct() {name = "pipe_5_0_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %435 = allo.stream_construct() {name = "pipe_5_0_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %436 = allo.stream_construct() {name = "pipe_5_0_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %437 = allo.stream_construct() {name = "pipe_5_0_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %438 = allo.stream_construct() {name = "pipe_5_0_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %439 = allo.stream_construct() {name = "pipe_5_0_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %440 = allo.stream_construct() {name = "pipe_5_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %441 = allo.stream_construct() {name = "pipe_5_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %442 = allo.stream_construct() {name = "pipe_5_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %443 = allo.stream_construct() {name = "pipe_5_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %444 = allo.stream_construct() {name = "pipe_5_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %445 = allo.stream_construct() {name = "pipe_5_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %446 = allo.stream_construct() {name = "pipe_5_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %447 = allo.stream_construct() {name = "pipe_5_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %448 = allo.stream_construct() {name = "pipe_5_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %449 = allo.stream_construct() {name = "pipe_5_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %450 = allo.stream_construct() {name = "pipe_5_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %451 = allo.stream_construct() {name = "pipe_5_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %452 = allo.stream_construct() {name = "pipe_5_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %453 = allo.stream_construct() {name = "pipe_5_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %454 = allo.stream_construct() {name = "pipe_5_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %455 = allo.stream_construct() {name = "pipe_5_1_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %456 = allo.stream_construct() {name = "pipe_5_1_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %457 = allo.stream_construct() {name = "pipe_5_1_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %458 = allo.stream_construct() {name = "pipe_5_1_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %459 = allo.stream_construct() {name = "pipe_5_1_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %460 = allo.stream_construct() {name = "pipe_5_1_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %461 = allo.stream_construct() {name = "pipe_5_1_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %462 = allo.stream_construct() {name = "pipe_5_1_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %463 = allo.stream_construct() {name = "pipe_5_1_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %464 = allo.stream_construct() {name = "pipe_5_1_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %465 = allo.stream_construct() {name = "pipe_5_1_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %466 = allo.stream_construct() {name = "pipe_5_1_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %467 = allo.stream_construct() {name = "pipe_5_1_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %468 = allo.stream_construct() {name = "pipe_5_1_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %469 = allo.stream_construct() {name = "pipe_5_1_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %470 = allo.stream_construct() {name = "pipe_5_1_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %471 = allo.stream_construct() {name = "pipe_5_1_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %472 = allo.stream_construct() {name = "pipe_5_1_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %473 = allo.stream_construct() {name = "pipe_5_1_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %474 = allo.stream_construct() {name = "pipe_5_1_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %475 = allo.stream_construct() {name = "pipe_5_1_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %476 = allo.stream_construct() {name = "pipe_5_1_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %477 = allo.stream_construct() {name = "pipe_5_1_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %478 = allo.stream_construct() {name = "pipe_5_1_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %479 = allo.stream_construct() {name = "pipe_5_1_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %480 = allo.stream_construct() {name = "pipe_6_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %481 = allo.stream_construct() {name = "pipe_6_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %482 = allo.stream_construct() {name = "pipe_6_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %483 = allo.stream_construct() {name = "pipe_6_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %484 = allo.stream_construct() {name = "pipe_6_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %485 = allo.stream_construct() {name = "pipe_6_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %486 = allo.stream_construct() {name = "pipe_6_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %487 = allo.stream_construct() {name = "pipe_6_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %488 = allo.stream_construct() {name = "pipe_6_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %489 = allo.stream_construct() {name = "pipe_6_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %490 = allo.stream_construct() {name = "pipe_6_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %491 = allo.stream_construct() {name = "pipe_6_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %492 = allo.stream_construct() {name = "pipe_6_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %493 = allo.stream_construct() {name = "pipe_6_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %494 = allo.stream_construct() {name = "pipe_6_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %495 = allo.stream_construct() {name = "pipe_6_0_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %496 = allo.stream_construct() {name = "pipe_6_0_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %497 = allo.stream_construct() {name = "pipe_6_0_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %498 = allo.stream_construct() {name = "pipe_6_0_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %499 = allo.stream_construct() {name = "pipe_6_0_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %500 = allo.stream_construct() {name = "pipe_6_0_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %501 = allo.stream_construct() {name = "pipe_6_0_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %502 = allo.stream_construct() {name = "pipe_6_0_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %503 = allo.stream_construct() {name = "pipe_6_0_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %504 = allo.stream_construct() {name = "pipe_6_0_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %505 = allo.stream_construct() {name = "pipe_6_0_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %506 = allo.stream_construct() {name = "pipe_6_0_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %507 = allo.stream_construct() {name = "pipe_6_0_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %508 = allo.stream_construct() {name = "pipe_6_0_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %509 = allo.stream_construct() {name = "pipe_6_0_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %510 = allo.stream_construct() {name = "pipe_6_0_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %511 = allo.stream_construct() {name = "pipe_6_0_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %512 = allo.stream_construct() {name = "pipe_6_0_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %513 = allo.stream_construct() {name = "pipe_6_0_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %514 = allo.stream_construct() {name = "pipe_6_0_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %515 = allo.stream_construct() {name = "pipe_6_0_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %516 = allo.stream_construct() {name = "pipe_6_0_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %517 = allo.stream_construct() {name = "pipe_6_0_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %518 = allo.stream_construct() {name = "pipe_6_0_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %519 = allo.stream_construct() {name = "pipe_6_0_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %520 = allo.stream_construct() {name = "pipe_6_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %521 = allo.stream_construct() {name = "pipe_6_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %522 = allo.stream_construct() {name = "pipe_6_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %523 = allo.stream_construct() {name = "pipe_6_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %524 = allo.stream_construct() {name = "pipe_6_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %525 = allo.stream_construct() {name = "pipe_6_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %526 = allo.stream_construct() {name = "pipe_6_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %527 = allo.stream_construct() {name = "pipe_6_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %528 = allo.stream_construct() {name = "pipe_6_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %529 = allo.stream_construct() {name = "pipe_6_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %530 = allo.stream_construct() {name = "pipe_6_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %531 = allo.stream_construct() {name = "pipe_6_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %532 = allo.stream_construct() {name = "pipe_6_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %533 = allo.stream_construct() {name = "pipe_6_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %534 = allo.stream_construct() {name = "pipe_6_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %535 = allo.stream_construct() {name = "pipe_6_1_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %536 = allo.stream_construct() {name = "pipe_6_1_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %537 = allo.stream_construct() {name = "pipe_6_1_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %538 = allo.stream_construct() {name = "pipe_6_1_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %539 = allo.stream_construct() {name = "pipe_6_1_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %540 = allo.stream_construct() {name = "pipe_6_1_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %541 = allo.stream_construct() {name = "pipe_6_1_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %542 = allo.stream_construct() {name = "pipe_6_1_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %543 = allo.stream_construct() {name = "pipe_6_1_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %544 = allo.stream_construct() {name = "pipe_6_1_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %545 = allo.stream_construct() {name = "pipe_6_1_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %546 = allo.stream_construct() {name = "pipe_6_1_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %547 = allo.stream_construct() {name = "pipe_6_1_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %548 = allo.stream_construct() {name = "pipe_6_1_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %549 = allo.stream_construct() {name = "pipe_6_1_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %550 = allo.stream_construct() {name = "pipe_6_1_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %551 = allo.stream_construct() {name = "pipe_6_1_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %552 = allo.stream_construct() {name = "pipe_6_1_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %553 = allo.stream_construct() {name = "pipe_6_1_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %554 = allo.stream_construct() {name = "pipe_6_1_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %555 = allo.stream_construct() {name = "pipe_6_1_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %556 = allo.stream_construct() {name = "pipe_6_1_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %557 = allo.stream_construct() {name = "pipe_6_1_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %558 = allo.stream_construct() {name = "pipe_6_1_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %559 = allo.stream_construct() {name = "pipe_6_1_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %560 = allo.stream_construct() {name = "pipe_7_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %561 = allo.stream_construct() {name = "pipe_7_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %562 = allo.stream_construct() {name = "pipe_7_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %563 = allo.stream_construct() {name = "pipe_7_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %564 = allo.stream_construct() {name = "pipe_7_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %565 = allo.stream_construct() {name = "pipe_7_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %566 = allo.stream_construct() {name = "pipe_7_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %567 = allo.stream_construct() {name = "pipe_7_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %568 = allo.stream_construct() {name = "pipe_7_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %569 = allo.stream_construct() {name = "pipe_7_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %570 = allo.stream_construct() {name = "pipe_7_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %571 = allo.stream_construct() {name = "pipe_7_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %572 = allo.stream_construct() {name = "pipe_7_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %573 = allo.stream_construct() {name = "pipe_7_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %574 = allo.stream_construct() {name = "pipe_7_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %575 = allo.stream_construct() {name = "pipe_7_0_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %576 = allo.stream_construct() {name = "pipe_7_0_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %577 = allo.stream_construct() {name = "pipe_7_0_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %578 = allo.stream_construct() {name = "pipe_7_0_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %579 = allo.stream_construct() {name = "pipe_7_0_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %580 = allo.stream_construct() {name = "pipe_7_0_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %581 = allo.stream_construct() {name = "pipe_7_0_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %582 = allo.stream_construct() {name = "pipe_7_0_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %583 = allo.stream_construct() {name = "pipe_7_0_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %584 = allo.stream_construct() {name = "pipe_7_0_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %585 = allo.stream_construct() {name = "pipe_7_0_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %586 = allo.stream_construct() {name = "pipe_7_0_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %587 = allo.stream_construct() {name = "pipe_7_0_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %588 = allo.stream_construct() {name = "pipe_7_0_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %589 = allo.stream_construct() {name = "pipe_7_0_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %590 = allo.stream_construct() {name = "pipe_7_0_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %591 = allo.stream_construct() {name = "pipe_7_0_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %592 = allo.stream_construct() {name = "pipe_7_0_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %593 = allo.stream_construct() {name = "pipe_7_0_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %594 = allo.stream_construct() {name = "pipe_7_0_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %595 = allo.stream_construct() {name = "pipe_7_0_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %596 = allo.stream_construct() {name = "pipe_7_0_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %597 = allo.stream_construct() {name = "pipe_7_0_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %598 = allo.stream_construct() {name = "pipe_7_0_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %599 = allo.stream_construct() {name = "pipe_7_0_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %600 = allo.stream_construct() {name = "pipe_7_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %601 = allo.stream_construct() {name = "pipe_7_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %602 = allo.stream_construct() {name = "pipe_7_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %603 = allo.stream_construct() {name = "pipe_7_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %604 = allo.stream_construct() {name = "pipe_7_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %605 = allo.stream_construct() {name = "pipe_7_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %606 = allo.stream_construct() {name = "pipe_7_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %607 = allo.stream_construct() {name = "pipe_7_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %608 = allo.stream_construct() {name = "pipe_7_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %609 = allo.stream_construct() {name = "pipe_7_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %610 = allo.stream_construct() {name = "pipe_7_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %611 = allo.stream_construct() {name = "pipe_7_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %612 = allo.stream_construct() {name = "pipe_7_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %613 = allo.stream_construct() {name = "pipe_7_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %614 = allo.stream_construct() {name = "pipe_7_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %615 = allo.stream_construct() {name = "pipe_7_1_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %616 = allo.stream_construct() {name = "pipe_7_1_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %617 = allo.stream_construct() {name = "pipe_7_1_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %618 = allo.stream_construct() {name = "pipe_7_1_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %619 = allo.stream_construct() {name = "pipe_7_1_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %620 = allo.stream_construct() {name = "pipe_7_1_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %621 = allo.stream_construct() {name = "pipe_7_1_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %622 = allo.stream_construct() {name = "pipe_7_1_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %623 = allo.stream_construct() {name = "pipe_7_1_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %624 = allo.stream_construct() {name = "pipe_7_1_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %625 = allo.stream_construct() {name = "pipe_7_1_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %626 = allo.stream_construct() {name = "pipe_7_1_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %627 = allo.stream_construct() {name = "pipe_7_1_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %628 = allo.stream_construct() {name = "pipe_7_1_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %629 = allo.stream_construct() {name = "pipe_7_1_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %630 = allo.stream_construct() {name = "pipe_7_1_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %631 = allo.stream_construct() {name = "pipe_7_1_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %632 = allo.stream_construct() {name = "pipe_7_1_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %633 = allo.stream_construct() {name = "pipe_7_1_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %634 = allo.stream_construct() {name = "pipe_7_1_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %635 = allo.stream_construct() {name = "pipe_7_1_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %636 = allo.stream_construct() {name = "pipe_7_1_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %637 = allo.stream_construct() {name = "pipe_7_1_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %638 = allo.stream_construct() {name = "pipe_7_1_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %639 = allo.stream_construct() {name = "pipe_7_1_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %640 = allo.stream_construct() {name = "pipe_8_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %641 = allo.stream_construct() {name = "pipe_8_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %642 = allo.stream_construct() {name = "pipe_8_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %643 = allo.stream_construct() {name = "pipe_8_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %644 = allo.stream_construct() {name = "pipe_8_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %645 = allo.stream_construct() {name = "pipe_8_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %646 = allo.stream_construct() {name = "pipe_8_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %647 = allo.stream_construct() {name = "pipe_8_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %648 = allo.stream_construct() {name = "pipe_8_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %649 = allo.stream_construct() {name = "pipe_8_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %650 = allo.stream_construct() {name = "pipe_8_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %651 = allo.stream_construct() {name = "pipe_8_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %652 = allo.stream_construct() {name = "pipe_8_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %653 = allo.stream_construct() {name = "pipe_8_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %654 = allo.stream_construct() {name = "pipe_8_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %655 = allo.stream_construct() {name = "pipe_8_0_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %656 = allo.stream_construct() {name = "pipe_8_0_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %657 = allo.stream_construct() {name = "pipe_8_0_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %658 = allo.stream_construct() {name = "pipe_8_0_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %659 = allo.stream_construct() {name = "pipe_8_0_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %660 = allo.stream_construct() {name = "pipe_8_0_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %661 = allo.stream_construct() {name = "pipe_8_0_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %662 = allo.stream_construct() {name = "pipe_8_0_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %663 = allo.stream_construct() {name = "pipe_8_0_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %664 = allo.stream_construct() {name = "pipe_8_0_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %665 = allo.stream_construct() {name = "pipe_8_0_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %666 = allo.stream_construct() {name = "pipe_8_0_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %667 = allo.stream_construct() {name = "pipe_8_0_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %668 = allo.stream_construct() {name = "pipe_8_0_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %669 = allo.stream_construct() {name = "pipe_8_0_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %670 = allo.stream_construct() {name = "pipe_8_0_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %671 = allo.stream_construct() {name = "pipe_8_0_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %672 = allo.stream_construct() {name = "pipe_8_0_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %673 = allo.stream_construct() {name = "pipe_8_0_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %674 = allo.stream_construct() {name = "pipe_8_0_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %675 = allo.stream_construct() {name = "pipe_8_0_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %676 = allo.stream_construct() {name = "pipe_8_0_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %677 = allo.stream_construct() {name = "pipe_8_0_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %678 = allo.stream_construct() {name = "pipe_8_0_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %679 = allo.stream_construct() {name = "pipe_8_0_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %680 = allo.stream_construct() {name = "pipe_8_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %681 = allo.stream_construct() {name = "pipe_8_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %682 = allo.stream_construct() {name = "pipe_8_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %683 = allo.stream_construct() {name = "pipe_8_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %684 = allo.stream_construct() {name = "pipe_8_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %685 = allo.stream_construct() {name = "pipe_8_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %686 = allo.stream_construct() {name = "pipe_8_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %687 = allo.stream_construct() {name = "pipe_8_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %688 = allo.stream_construct() {name = "pipe_8_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %689 = allo.stream_construct() {name = "pipe_8_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %690 = allo.stream_construct() {name = "pipe_8_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %691 = allo.stream_construct() {name = "pipe_8_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %692 = allo.stream_construct() {name = "pipe_8_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %693 = allo.stream_construct() {name = "pipe_8_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %694 = allo.stream_construct() {name = "pipe_8_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %695 = allo.stream_construct() {name = "pipe_8_1_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %696 = allo.stream_construct() {name = "pipe_8_1_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %697 = allo.stream_construct() {name = "pipe_8_1_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %698 = allo.stream_construct() {name = "pipe_8_1_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %699 = allo.stream_construct() {name = "pipe_8_1_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %700 = allo.stream_construct() {name = "pipe_8_1_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %701 = allo.stream_construct() {name = "pipe_8_1_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %702 = allo.stream_construct() {name = "pipe_8_1_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %703 = allo.stream_construct() {name = "pipe_8_1_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %704 = allo.stream_construct() {name = "pipe_8_1_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %705 = allo.stream_construct() {name = "pipe_8_1_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %706 = allo.stream_construct() {name = "pipe_8_1_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %707 = allo.stream_construct() {name = "pipe_8_1_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %708 = allo.stream_construct() {name = "pipe_8_1_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %709 = allo.stream_construct() {name = "pipe_8_1_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %710 = allo.stream_construct() {name = "pipe_8_1_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %711 = allo.stream_construct() {name = "pipe_8_1_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %712 = allo.stream_construct() {name = "pipe_8_1_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %713 = allo.stream_construct() {name = "pipe_8_1_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %714 = allo.stream_construct() {name = "pipe_8_1_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %715 = allo.stream_construct() {name = "pipe_8_1_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %716 = allo.stream_construct() {name = "pipe_8_1_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %717 = allo.stream_construct() {name = "pipe_8_1_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %718 = allo.stream_construct() {name = "pipe_8_1_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %719 = allo.stream_construct() {name = "pipe_8_1_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %720 = allo.stream_construct() {name = "pipe_9_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %721 = allo.stream_construct() {name = "pipe_9_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %722 = allo.stream_construct() {name = "pipe_9_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %723 = allo.stream_construct() {name = "pipe_9_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %724 = allo.stream_construct() {name = "pipe_9_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %725 = allo.stream_construct() {name = "pipe_9_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %726 = allo.stream_construct() {name = "pipe_9_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %727 = allo.stream_construct() {name = "pipe_9_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %728 = allo.stream_construct() {name = "pipe_9_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %729 = allo.stream_construct() {name = "pipe_9_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %730 = allo.stream_construct() {name = "pipe_9_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %731 = allo.stream_construct() {name = "pipe_9_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %732 = allo.stream_construct() {name = "pipe_9_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %733 = allo.stream_construct() {name = "pipe_9_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %734 = allo.stream_construct() {name = "pipe_9_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %735 = allo.stream_construct() {name = "pipe_9_0_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %736 = allo.stream_construct() {name = "pipe_9_0_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %737 = allo.stream_construct() {name = "pipe_9_0_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %738 = allo.stream_construct() {name = "pipe_9_0_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %739 = allo.stream_construct() {name = "pipe_9_0_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %740 = allo.stream_construct() {name = "pipe_9_0_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %741 = allo.stream_construct() {name = "pipe_9_0_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %742 = allo.stream_construct() {name = "pipe_9_0_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %743 = allo.stream_construct() {name = "pipe_9_0_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %744 = allo.stream_construct() {name = "pipe_9_0_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %745 = allo.stream_construct() {name = "pipe_9_0_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %746 = allo.stream_construct() {name = "pipe_9_0_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %747 = allo.stream_construct() {name = "pipe_9_0_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %748 = allo.stream_construct() {name = "pipe_9_0_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %749 = allo.stream_construct() {name = "pipe_9_0_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %750 = allo.stream_construct() {name = "pipe_9_0_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %751 = allo.stream_construct() {name = "pipe_9_0_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %752 = allo.stream_construct() {name = "pipe_9_0_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %753 = allo.stream_construct() {name = "pipe_9_0_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %754 = allo.stream_construct() {name = "pipe_9_0_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %755 = allo.stream_construct() {name = "pipe_9_0_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %756 = allo.stream_construct() {name = "pipe_9_0_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %757 = allo.stream_construct() {name = "pipe_9_0_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %758 = allo.stream_construct() {name = "pipe_9_0_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %759 = allo.stream_construct() {name = "pipe_9_0_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %760 = allo.stream_construct() {name = "pipe_9_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %761 = allo.stream_construct() {name = "pipe_9_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %762 = allo.stream_construct() {name = "pipe_9_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %763 = allo.stream_construct() {name = "pipe_9_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %764 = allo.stream_construct() {name = "pipe_9_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %765 = allo.stream_construct() {name = "pipe_9_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %766 = allo.stream_construct() {name = "pipe_9_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %767 = allo.stream_construct() {name = "pipe_9_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %768 = allo.stream_construct() {name = "pipe_9_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %769 = allo.stream_construct() {name = "pipe_9_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %770 = allo.stream_construct() {name = "pipe_9_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %771 = allo.stream_construct() {name = "pipe_9_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %772 = allo.stream_construct() {name = "pipe_9_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %773 = allo.stream_construct() {name = "pipe_9_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %774 = allo.stream_construct() {name = "pipe_9_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %775 = allo.stream_construct() {name = "pipe_9_1_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %776 = allo.stream_construct() {name = "pipe_9_1_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %777 = allo.stream_construct() {name = "pipe_9_1_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %778 = allo.stream_construct() {name = "pipe_9_1_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %779 = allo.stream_construct() {name = "pipe_9_1_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %780 = allo.stream_construct() {name = "pipe_9_1_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %781 = allo.stream_construct() {name = "pipe_9_1_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %782 = allo.stream_construct() {name = "pipe_9_1_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %783 = allo.stream_construct() {name = "pipe_9_1_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %784 = allo.stream_construct() {name = "pipe_9_1_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %785 = allo.stream_construct() {name = "pipe_9_1_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %786 = allo.stream_construct() {name = "pipe_9_1_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %787 = allo.stream_construct() {name = "pipe_9_1_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %788 = allo.stream_construct() {name = "pipe_9_1_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %789 = allo.stream_construct() {name = "pipe_9_1_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %790 = allo.stream_construct() {name = "pipe_9_1_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %791 = allo.stream_construct() {name = "pipe_9_1_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %792 = allo.stream_construct() {name = "pipe_9_1_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %793 = allo.stream_construct() {name = "pipe_9_1_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %794 = allo.stream_construct() {name = "pipe_9_1_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %795 = allo.stream_construct() {name = "pipe_9_1_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %796 = allo.stream_construct() {name = "pipe_9_1_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %797 = allo.stream_construct() {name = "pipe_9_1_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %798 = allo.stream_construct() {name = "pipe_9_1_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %799 = allo.stream_construct() {name = "pipe_9_1_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %800 = allo.stream_construct() {name = "pipe_10_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %801 = allo.stream_construct() {name = "pipe_10_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %802 = allo.stream_construct() {name = "pipe_10_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %803 = allo.stream_construct() {name = "pipe_10_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %804 = allo.stream_construct() {name = "pipe_10_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %805 = allo.stream_construct() {name = "pipe_10_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %806 = allo.stream_construct() {name = "pipe_10_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %807 = allo.stream_construct() {name = "pipe_10_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %808 = allo.stream_construct() {name = "pipe_10_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %809 = allo.stream_construct() {name = "pipe_10_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %810 = allo.stream_construct() {name = "pipe_10_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %811 = allo.stream_construct() {name = "pipe_10_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %812 = allo.stream_construct() {name = "pipe_10_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %813 = allo.stream_construct() {name = "pipe_10_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %814 = allo.stream_construct() {name = "pipe_10_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %815 = allo.stream_construct() {name = "pipe_10_0_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %816 = allo.stream_construct() {name = "pipe_10_0_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %817 = allo.stream_construct() {name = "pipe_10_0_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %818 = allo.stream_construct() {name = "pipe_10_0_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %819 = allo.stream_construct() {name = "pipe_10_0_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %820 = allo.stream_construct() {name = "pipe_10_0_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %821 = allo.stream_construct() {name = "pipe_10_0_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %822 = allo.stream_construct() {name = "pipe_10_0_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %823 = allo.stream_construct() {name = "pipe_10_0_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %824 = allo.stream_construct() {name = "pipe_10_0_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %825 = allo.stream_construct() {name = "pipe_10_0_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %826 = allo.stream_construct() {name = "pipe_10_0_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %827 = allo.stream_construct() {name = "pipe_10_0_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %828 = allo.stream_construct() {name = "pipe_10_0_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %829 = allo.stream_construct() {name = "pipe_10_0_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %830 = allo.stream_construct() {name = "pipe_10_0_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %831 = allo.stream_construct() {name = "pipe_10_0_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %832 = allo.stream_construct() {name = "pipe_10_0_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %833 = allo.stream_construct() {name = "pipe_10_0_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %834 = allo.stream_construct() {name = "pipe_10_0_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %835 = allo.stream_construct() {name = "pipe_10_0_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %836 = allo.stream_construct() {name = "pipe_10_0_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %837 = allo.stream_construct() {name = "pipe_10_0_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %838 = allo.stream_construct() {name = "pipe_10_0_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %839 = allo.stream_construct() {name = "pipe_10_0_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %840 = allo.stream_construct() {name = "pipe_10_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %841 = allo.stream_construct() {name = "pipe_10_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %842 = allo.stream_construct() {name = "pipe_10_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %843 = allo.stream_construct() {name = "pipe_10_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %844 = allo.stream_construct() {name = "pipe_10_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %845 = allo.stream_construct() {name = "pipe_10_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %846 = allo.stream_construct() {name = "pipe_10_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %847 = allo.stream_construct() {name = "pipe_10_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %848 = allo.stream_construct() {name = "pipe_10_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %849 = allo.stream_construct() {name = "pipe_10_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %850 = allo.stream_construct() {name = "pipe_10_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %851 = allo.stream_construct() {name = "pipe_10_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %852 = allo.stream_construct() {name = "pipe_10_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %853 = allo.stream_construct() {name = "pipe_10_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %854 = allo.stream_construct() {name = "pipe_10_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %855 = allo.stream_construct() {name = "pipe_10_1_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %856 = allo.stream_construct() {name = "pipe_10_1_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %857 = allo.stream_construct() {name = "pipe_10_1_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %858 = allo.stream_construct() {name = "pipe_10_1_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %859 = allo.stream_construct() {name = "pipe_10_1_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %860 = allo.stream_construct() {name = "pipe_10_1_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %861 = allo.stream_construct() {name = "pipe_10_1_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %862 = allo.stream_construct() {name = "pipe_10_1_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %863 = allo.stream_construct() {name = "pipe_10_1_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %864 = allo.stream_construct() {name = "pipe_10_1_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %865 = allo.stream_construct() {name = "pipe_10_1_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %866 = allo.stream_construct() {name = "pipe_10_1_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %867 = allo.stream_construct() {name = "pipe_10_1_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %868 = allo.stream_construct() {name = "pipe_10_1_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %869 = allo.stream_construct() {name = "pipe_10_1_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %870 = allo.stream_construct() {name = "pipe_10_1_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %871 = allo.stream_construct() {name = "pipe_10_1_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %872 = allo.stream_construct() {name = "pipe_10_1_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %873 = allo.stream_construct() {name = "pipe_10_1_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %874 = allo.stream_construct() {name = "pipe_10_1_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %875 = allo.stream_construct() {name = "pipe_10_1_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %876 = allo.stream_construct() {name = "pipe_10_1_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %877 = allo.stream_construct() {name = "pipe_10_1_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %878 = allo.stream_construct() {name = "pipe_10_1_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %879 = allo.stream_construct() {name = "pipe_10_1_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %880 = allo.stream_construct() {name = "pipe_11_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %881 = allo.stream_construct() {name = "pipe_11_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %882 = allo.stream_construct() {name = "pipe_11_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %883 = allo.stream_construct() {name = "pipe_11_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %884 = allo.stream_construct() {name = "pipe_11_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %885 = allo.stream_construct() {name = "pipe_11_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %886 = allo.stream_construct() {name = "pipe_11_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %887 = allo.stream_construct() {name = "pipe_11_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %888 = allo.stream_construct() {name = "pipe_11_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %889 = allo.stream_construct() {name = "pipe_11_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %890 = allo.stream_construct() {name = "pipe_11_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %891 = allo.stream_construct() {name = "pipe_11_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %892 = allo.stream_construct() {name = "pipe_11_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %893 = allo.stream_construct() {name = "pipe_11_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %894 = allo.stream_construct() {name = "pipe_11_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %895 = allo.stream_construct() {name = "pipe_11_0_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %896 = allo.stream_construct() {name = "pipe_11_0_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %897 = allo.stream_construct() {name = "pipe_11_0_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %898 = allo.stream_construct() {name = "pipe_11_0_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %899 = allo.stream_construct() {name = "pipe_11_0_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %900 = allo.stream_construct() {name = "pipe_11_0_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %901 = allo.stream_construct() {name = "pipe_11_0_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %902 = allo.stream_construct() {name = "pipe_11_0_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %903 = allo.stream_construct() {name = "pipe_11_0_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %904 = allo.stream_construct() {name = "pipe_11_0_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %905 = allo.stream_construct() {name = "pipe_11_0_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %906 = allo.stream_construct() {name = "pipe_11_0_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %907 = allo.stream_construct() {name = "pipe_11_0_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %908 = allo.stream_construct() {name = "pipe_11_0_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %909 = allo.stream_construct() {name = "pipe_11_0_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %910 = allo.stream_construct() {name = "pipe_11_0_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %911 = allo.stream_construct() {name = "pipe_11_0_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %912 = allo.stream_construct() {name = "pipe_11_0_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %913 = allo.stream_construct() {name = "pipe_11_0_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %914 = allo.stream_construct() {name = "pipe_11_0_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %915 = allo.stream_construct() {name = "pipe_11_0_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %916 = allo.stream_construct() {name = "pipe_11_0_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %917 = allo.stream_construct() {name = "pipe_11_0_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %918 = allo.stream_construct() {name = "pipe_11_0_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %919 = allo.stream_construct() {name = "pipe_11_0_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %920 = allo.stream_construct() {name = "pipe_11_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %921 = allo.stream_construct() {name = "pipe_11_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %922 = allo.stream_construct() {name = "pipe_11_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %923 = allo.stream_construct() {name = "pipe_11_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %924 = allo.stream_construct() {name = "pipe_11_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %925 = allo.stream_construct() {name = "pipe_11_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %926 = allo.stream_construct() {name = "pipe_11_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %927 = allo.stream_construct() {name = "pipe_11_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %928 = allo.stream_construct() {name = "pipe_11_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %929 = allo.stream_construct() {name = "pipe_11_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %930 = allo.stream_construct() {name = "pipe_11_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %931 = allo.stream_construct() {name = "pipe_11_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %932 = allo.stream_construct() {name = "pipe_11_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %933 = allo.stream_construct() {name = "pipe_11_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %934 = allo.stream_construct() {name = "pipe_11_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %935 = allo.stream_construct() {name = "pipe_11_1_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %936 = allo.stream_construct() {name = "pipe_11_1_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %937 = allo.stream_construct() {name = "pipe_11_1_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %938 = allo.stream_construct() {name = "pipe_11_1_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %939 = allo.stream_construct() {name = "pipe_11_1_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %940 = allo.stream_construct() {name = "pipe_11_1_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %941 = allo.stream_construct() {name = "pipe_11_1_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %942 = allo.stream_construct() {name = "pipe_11_1_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %943 = allo.stream_construct() {name = "pipe_11_1_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %944 = allo.stream_construct() {name = "pipe_11_1_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %945 = allo.stream_construct() {name = "pipe_11_1_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %946 = allo.stream_construct() {name = "pipe_11_1_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %947 = allo.stream_construct() {name = "pipe_11_1_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %948 = allo.stream_construct() {name = "pipe_11_1_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %949 = allo.stream_construct() {name = "pipe_11_1_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %950 = allo.stream_construct() {name = "pipe_11_1_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %951 = allo.stream_construct() {name = "pipe_11_1_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %952 = allo.stream_construct() {name = "pipe_11_1_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %953 = allo.stream_construct() {name = "pipe_11_1_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %954 = allo.stream_construct() {name = "pipe_11_1_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %955 = allo.stream_construct() {name = "pipe_11_1_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %956 = allo.stream_construct() {name = "pipe_11_1_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %957 = allo.stream_construct() {name = "pipe_11_1_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %958 = allo.stream_construct() {name = "pipe_11_1_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %959 = allo.stream_construct() {name = "pipe_11_1_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %960 = allo.stream_construct() {name = "pipe_12_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %961 = allo.stream_construct() {name = "pipe_12_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %962 = allo.stream_construct() {name = "pipe_12_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %963 = allo.stream_construct() {name = "pipe_12_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %964 = allo.stream_construct() {name = "pipe_12_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %965 = allo.stream_construct() {name = "pipe_12_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %966 = allo.stream_construct() {name = "pipe_12_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %967 = allo.stream_construct() {name = "pipe_12_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %968 = allo.stream_construct() {name = "pipe_12_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %969 = allo.stream_construct() {name = "pipe_12_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %970 = allo.stream_construct() {name = "pipe_12_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %971 = allo.stream_construct() {name = "pipe_12_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %972 = allo.stream_construct() {name = "pipe_12_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %973 = allo.stream_construct() {name = "pipe_12_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %974 = allo.stream_construct() {name = "pipe_12_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %975 = allo.stream_construct() {name = "pipe_12_0_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %976 = allo.stream_construct() {name = "pipe_12_0_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %977 = allo.stream_construct() {name = "pipe_12_0_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %978 = allo.stream_construct() {name = "pipe_12_0_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %979 = allo.stream_construct() {name = "pipe_12_0_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %980 = allo.stream_construct() {name = "pipe_12_0_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %981 = allo.stream_construct() {name = "pipe_12_0_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %982 = allo.stream_construct() {name = "pipe_12_0_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %983 = allo.stream_construct() {name = "pipe_12_0_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %984 = allo.stream_construct() {name = "pipe_12_0_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %985 = allo.stream_construct() {name = "pipe_12_0_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %986 = allo.stream_construct() {name = "pipe_12_0_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %987 = allo.stream_construct() {name = "pipe_12_0_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %988 = allo.stream_construct() {name = "pipe_12_0_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %989 = allo.stream_construct() {name = "pipe_12_0_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %990 = allo.stream_construct() {name = "pipe_12_0_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %991 = allo.stream_construct() {name = "pipe_12_0_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %992 = allo.stream_construct() {name = "pipe_12_0_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %993 = allo.stream_construct() {name = "pipe_12_0_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %994 = allo.stream_construct() {name = "pipe_12_0_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %995 = allo.stream_construct() {name = "pipe_12_0_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %996 = allo.stream_construct() {name = "pipe_12_0_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %997 = allo.stream_construct() {name = "pipe_12_0_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %998 = allo.stream_construct() {name = "pipe_12_0_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %999 = allo.stream_construct() {name = "pipe_12_0_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %1000 = allo.stream_construct() {name = "pipe_12_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1001 = allo.stream_construct() {name = "pipe_12_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1002 = allo.stream_construct() {name = "pipe_12_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1003 = allo.stream_construct() {name = "pipe_12_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1004 = allo.stream_construct() {name = "pipe_12_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1005 = allo.stream_construct() {name = "pipe_12_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1006 = allo.stream_construct() {name = "pipe_12_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1007 = allo.stream_construct() {name = "pipe_12_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1008 = allo.stream_construct() {name = "pipe_12_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %1009 = allo.stream_construct() {name = "pipe_12_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %1010 = allo.stream_construct() {name = "pipe_12_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %1011 = allo.stream_construct() {name = "pipe_12_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %1012 = allo.stream_construct() {name = "pipe_12_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %1013 = allo.stream_construct() {name = "pipe_12_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %1014 = allo.stream_construct() {name = "pipe_12_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %1015 = allo.stream_construct() {name = "pipe_12_1_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %1016 = allo.stream_construct() {name = "pipe_12_1_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %1017 = allo.stream_construct() {name = "pipe_12_1_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %1018 = allo.stream_construct() {name = "pipe_12_1_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %1019 = allo.stream_construct() {name = "pipe_12_1_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %1020 = allo.stream_construct() {name = "pipe_12_1_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %1021 = allo.stream_construct() {name = "pipe_12_1_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %1022 = allo.stream_construct() {name = "pipe_12_1_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %1023 = allo.stream_construct() {name = "pipe_12_1_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %1024 = allo.stream_construct() {name = "pipe_12_1_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %1025 = allo.stream_construct() {name = "pipe_12_1_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %1026 = allo.stream_construct() {name = "pipe_12_1_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %1027 = allo.stream_construct() {name = "pipe_12_1_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %1028 = allo.stream_construct() {name = "pipe_12_1_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %1029 = allo.stream_construct() {name = "pipe_12_1_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %1030 = allo.stream_construct() {name = "pipe_12_1_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %1031 = allo.stream_construct() {name = "pipe_12_1_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %1032 = allo.stream_construct() {name = "pipe_12_1_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %1033 = allo.stream_construct() {name = "pipe_12_1_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %1034 = allo.stream_construct() {name = "pipe_12_1_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %1035 = allo.stream_construct() {name = "pipe_12_1_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %1036 = allo.stream_construct() {name = "pipe_12_1_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %1037 = allo.stream_construct() {name = "pipe_12_1_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %1038 = allo.stream_construct() {name = "pipe_12_1_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %1039 = allo.stream_construct() {name = "pipe_12_1_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %1040 = allo.stream_construct() {name = "pipe_13_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1041 = allo.stream_construct() {name = "pipe_13_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1042 = allo.stream_construct() {name = "pipe_13_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1043 = allo.stream_construct() {name = "pipe_13_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1044 = allo.stream_construct() {name = "pipe_13_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1045 = allo.stream_construct() {name = "pipe_13_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1046 = allo.stream_construct() {name = "pipe_13_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1047 = allo.stream_construct() {name = "pipe_13_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1048 = allo.stream_construct() {name = "pipe_13_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %1049 = allo.stream_construct() {name = "pipe_13_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %1050 = allo.stream_construct() {name = "pipe_13_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %1051 = allo.stream_construct() {name = "pipe_13_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %1052 = allo.stream_construct() {name = "pipe_13_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %1053 = allo.stream_construct() {name = "pipe_13_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %1054 = allo.stream_construct() {name = "pipe_13_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %1055 = allo.stream_construct() {name = "pipe_13_0_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %1056 = allo.stream_construct() {name = "pipe_13_0_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %1057 = allo.stream_construct() {name = "pipe_13_0_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %1058 = allo.stream_construct() {name = "pipe_13_0_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %1059 = allo.stream_construct() {name = "pipe_13_0_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %1060 = allo.stream_construct() {name = "pipe_13_0_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %1061 = allo.stream_construct() {name = "pipe_13_0_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %1062 = allo.stream_construct() {name = "pipe_13_0_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %1063 = allo.stream_construct() {name = "pipe_13_0_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %1064 = allo.stream_construct() {name = "pipe_13_0_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %1065 = allo.stream_construct() {name = "pipe_13_0_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %1066 = allo.stream_construct() {name = "pipe_13_0_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %1067 = allo.stream_construct() {name = "pipe_13_0_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %1068 = allo.stream_construct() {name = "pipe_13_0_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %1069 = allo.stream_construct() {name = "pipe_13_0_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %1070 = allo.stream_construct() {name = "pipe_13_0_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %1071 = allo.stream_construct() {name = "pipe_13_0_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %1072 = allo.stream_construct() {name = "pipe_13_0_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %1073 = allo.stream_construct() {name = "pipe_13_0_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %1074 = allo.stream_construct() {name = "pipe_13_0_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %1075 = allo.stream_construct() {name = "pipe_13_0_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %1076 = allo.stream_construct() {name = "pipe_13_0_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %1077 = allo.stream_construct() {name = "pipe_13_0_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %1078 = allo.stream_construct() {name = "pipe_13_0_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %1079 = allo.stream_construct() {name = "pipe_13_0_39"} : !allo.stream<memref<64x64xbf16>, 2>
    %1080 = allo.stream_construct() {name = "pipe_13_1_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %1081 = allo.stream_construct() {name = "pipe_13_1_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %1082 = allo.stream_construct() {name = "pipe_13_1_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %1083 = allo.stream_construct() {name = "pipe_13_1_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %1084 = allo.stream_construct() {name = "pipe_13_1_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %1085 = allo.stream_construct() {name = "pipe_13_1_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %1086 = allo.stream_construct() {name = "pipe_13_1_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %1087 = allo.stream_construct() {name = "pipe_13_1_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %1088 = allo.stream_construct() {name = "pipe_13_1_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %1089 = allo.stream_construct() {name = "pipe_13_1_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %1090 = allo.stream_construct() {name = "pipe_13_1_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %1091 = allo.stream_construct() {name = "pipe_13_1_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %1092 = allo.stream_construct() {name = "pipe_13_1_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %1093 = allo.stream_construct() {name = "pipe_13_1_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %1094 = allo.stream_construct() {name = "pipe_13_1_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %1095 = allo.stream_construct() {name = "pipe_13_1_15"} : !allo.stream<memref<64x64xbf16>, 2>
    %1096 = allo.stream_construct() {name = "pipe_13_1_16"} : !allo.stream<memref<64x64xbf16>, 2>
    %1097 = allo.stream_construct() {name = "pipe_13_1_17"} : !allo.stream<memref<64x64xbf16>, 2>
    %1098 = allo.stream_construct() {name = "pipe_13_1_18"} : !allo.stream<memref<64x64xbf16>, 2>
    %1099 = allo.stream_construct() {name = "pipe_13_1_19"} : !allo.stream<memref<64x64xbf16>, 2>
    %1100 = allo.stream_construct() {name = "pipe_13_1_20"} : !allo.stream<memref<64x64xbf16>, 2>
    %1101 = allo.stream_construct() {name = "pipe_13_1_21"} : !allo.stream<memref<64x64xbf16>, 2>
    %1102 = allo.stream_construct() {name = "pipe_13_1_22"} : !allo.stream<memref<64x64xbf16>, 2>
    %1103 = allo.stream_construct() {name = "pipe_13_1_23"} : !allo.stream<memref<64x64xbf16>, 2>
    %1104 = allo.stream_construct() {name = "pipe_13_1_24"} : !allo.stream<memref<64x64xbf16>, 2>
    %1105 = allo.stream_construct() {name = "pipe_13_1_25"} : !allo.stream<memref<64x64xbf16>, 2>
    %1106 = allo.stream_construct() {name = "pipe_13_1_26"} : !allo.stream<memref<64x64xbf16>, 2>
    %1107 = allo.stream_construct() {name = "pipe_13_1_27"} : !allo.stream<memref<64x64xbf16>, 2>
    %1108 = allo.stream_construct() {name = "pipe_13_1_28"} : !allo.stream<memref<64x64xbf16>, 2>
    %1109 = allo.stream_construct() {name = "pipe_13_1_29"} : !allo.stream<memref<64x64xbf16>, 2>
    %1110 = allo.stream_construct() {name = "pipe_13_1_30"} : !allo.stream<memref<64x64xbf16>, 2>
    %1111 = allo.stream_construct() {name = "pipe_13_1_31"} : !allo.stream<memref<64x64xbf16>, 2>
    %1112 = allo.stream_construct() {name = "pipe_13_1_32"} : !allo.stream<memref<64x64xbf16>, 2>
    %1113 = allo.stream_construct() {name = "pipe_13_1_33"} : !allo.stream<memref<64x64xbf16>, 2>
    %1114 = allo.stream_construct() {name = "pipe_13_1_34"} : !allo.stream<memref<64x64xbf16>, 2>
    %1115 = allo.stream_construct() {name = "pipe_13_1_35"} : !allo.stream<memref<64x64xbf16>, 2>
    %1116 = allo.stream_construct() {name = "pipe_13_1_36"} : !allo.stream<memref<64x64xbf16>, 2>
    %1117 = allo.stream_construct() {name = "pipe_13_1_37"} : !allo.stream<memref<64x64xbf16>, 2>
    %1118 = allo.stream_construct() {name = "pipe_13_1_38"} : !allo.stream<memref<64x64xbf16>, 2>
    %1119 = allo.stream_construct() {name = "pipe_13_1_39"} : !allo.stream<memref<64x64xbf16>, 2>
    return
  }
}
