module {
  func.func private @fill_zeros_bf16_64_64_vector(memref<64x64xbf16>)
  func.func private @matmul_scalar_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func private @add_bf16_vector(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func @"gemm_0_0_12-gemm_1_0_12-gemm_2_0_12-gemm_3_0_12-gemm_4_0_12-gemm_5_0_12-gemm_6_0_12-gemm_7_0_12-gemm_8_0_12-gemm_9_0_12-gemm_10_0_12-gemm_11_0_12-gemm_12_0_12-gemm_13_0_12-gemm_14_0_12"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
  func.func @"gemm_0_0_13-gemm_1_0_13-gemm_2_0_13-gemm_3_0_13-gemm_4_0_13-gemm_5_0_13-gemm_6_0_13-gemm_7_0_13-gemm_8_0_13-gemm_9_0_13-gemm_10_0_13-gemm_11_0_13-gemm_12_0_13-gemm_13_0_13-gemm_14_0_13"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
  func.func @"gemm_0_0_14-gemm_1_0_14-gemm_2_0_14-gemm_3_0_14-gemm_4_0_14-gemm_5_0_14-gemm_6_0_14-gemm_7_0_14-gemm_8_0_14-gemm_9_0_14-gemm_10_0_14-gemm_11_0_14-gemm_12_0_14-gemm_13_0_14-gemm_14_0_14"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
  func.func @"gemm_0_0_0-gemm_1_0_0-gemm_2_0_0-gemm_3_0_0-gemm_4_0_0-gemm_5_0_0-gemm_6_0_0-gemm_7_0_0-gemm_8_0_0-gemm_9_0_0-gemm_10_0_0-gemm_11_0_0-gemm_12_0_0-gemm_13_0_0-gemm_14_0_0x3"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
  func.func @"gemm_0_0_1-gemm_1_0_1-gemm_2_0_1-gemm_3_0_1-gemm_4_0_1-gemm_5_0_1-gemm_6_0_1-gemm_7_0_1-gemm_8_0_1-gemm_9_0_1-gemm_10_0_1-gemm_11_0_1-gemm_12_0_1-gemm_13_0_1-gemm_14_0_1x3"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
  func.func @"gemm_0_0_2-gemm_1_0_2-gemm_2_0_2-gemm_3_0_2-gemm_4_0_2-gemm_5_0_2-gemm_6_0_2-gemm_7_0_2-gemm_8_0_2-gemm_9_0_2-gemm_10_0_2-gemm_11_0_2-gemm_12_0_2-gemm_13_0_2-gemm_14_0_2x3"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
  func.func @"gemm_0_0_3-gemm_1_0_3-gemm_2_0_3-gemm_3_0_3-gemm_4_0_3-gemm_5_0_3-gemm_6_0_3-gemm_7_0_3-gemm_8_0_3-gemm_9_0_3-gemm_10_0_3-gemm_11_0_3-gemm_12_0_3-gemm_13_0_3-gemm_14_0_3x3"(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>, %arg3: !allo.stream<memref<64x64xbf16>, 2>, %arg4: memref<64x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<64x64xbf16>, %arg7: !allo.stream<memref<64x64xbf16>, 2>, %arg8: !allo.stream<memref<64x64xbf16>, 2>, %arg9: memref<64x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<64x64xbf16>, %arg12: !allo.stream<memref<64x64xbf16>, 2>, %arg13: !allo.stream<memref<64x64xbf16>, 2>, %arg14: memref<64x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<64x64xbf16>, %arg17: !allo.stream<memref<64x64xbf16>, 2>, %arg18: !allo.stream<memref<64x64xbf16>, 2>, %arg19: memref<64x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<64x64xbf16>, %arg22: !allo.stream<memref<64x64xbf16>, 2>, %arg23: !allo.stream<memref<64x64xbf16>, 2>, %arg24: memref<64x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<64x64xbf16>, %arg27: !allo.stream<memref<64x64xbf16>, 2>, %arg28: !allo.stream<memref<64x64xbf16>, 2>, %arg29: memref<64x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<64x64xbf16>, %arg32: !allo.stream<memref<64x64xbf16>, 2>, %arg33: !allo.stream<memref<64x64xbf16>, 2>, %arg34: memref<64x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<64x64xbf16>, %arg37: !allo.stream<memref<64x64xbf16>, 2>, %arg38: !allo.stream<memref<64x64xbf16>, 2>, %arg39: memref<64x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<64x64xbf16>, %arg42: !allo.stream<memref<64x64xbf16>, 2>, %arg43: !allo.stream<memref<64x64xbf16>, 2>, %arg44: memref<64x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<64x64xbf16>, %arg47: !allo.stream<memref<64x64xbf16>, 2>, %arg48: !allo.stream<memref<64x64xbf16>, 2>, %arg49: memref<64x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<64x64xbf16>, %arg52: !allo.stream<memref<64x64xbf16>, 2>, %arg53: !allo.stream<memref<64x64xbf16>, 2>, %arg54: memref<64x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<64x64xbf16>, %arg57: !allo.stream<memref<64x64xbf16>, 2>, %arg58: !allo.stream<memref<64x64xbf16>, 2>, %arg59: memref<64x64xbf16>, %arg60: memref<64x64xbf16>, %arg61: memref<64x64xbf16>, %arg62: !allo.stream<memref<64x64xbf16>, 2>, %arg63: !allo.stream<memref<64x64xbf16>, 2>, %arg64: memref<64x64xbf16>, %arg65: memref<64x64xbf16>, %arg66: memref<64x64xbf16>, %arg67: !allo.stream<memref<64x64xbf16>, 2>, %arg68: !allo.stream<memref<64x64xbf16>, 2>, %arg69: memref<64x64xbf16>, %arg70: memref<64x64xbf16>, %arg71: memref<64x64xbf16>, %arg72: !allo.stream<memref<64x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
  func.func @top(%arg0: memref<64x960xbf16>, %arg1: memref<960x960xbf16>, %arg2: memref<64x960xbf16>) attributes {dataflow, itypes = "___"} {
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
    %15 = allo.stream_construct() {name = "pipe_1_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %16 = allo.stream_construct() {name = "pipe_1_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %17 = allo.stream_construct() {name = "pipe_1_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %18 = allo.stream_construct() {name = "pipe_1_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %19 = allo.stream_construct() {name = "pipe_1_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %20 = allo.stream_construct() {name = "pipe_1_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %21 = allo.stream_construct() {name = "pipe_1_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %22 = allo.stream_construct() {name = "pipe_1_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %23 = allo.stream_construct() {name = "pipe_1_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %24 = allo.stream_construct() {name = "pipe_1_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %25 = allo.stream_construct() {name = "pipe_1_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %26 = allo.stream_construct() {name = "pipe_1_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %27 = allo.stream_construct() {name = "pipe_1_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %28 = allo.stream_construct() {name = "pipe_1_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %29 = allo.stream_construct() {name = "pipe_1_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %30 = allo.stream_construct() {name = "pipe_2_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %31 = allo.stream_construct() {name = "pipe_2_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %32 = allo.stream_construct() {name = "pipe_2_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %33 = allo.stream_construct() {name = "pipe_2_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %34 = allo.stream_construct() {name = "pipe_2_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %35 = allo.stream_construct() {name = "pipe_2_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %36 = allo.stream_construct() {name = "pipe_2_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %37 = allo.stream_construct() {name = "pipe_2_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %38 = allo.stream_construct() {name = "pipe_2_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %39 = allo.stream_construct() {name = "pipe_2_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %40 = allo.stream_construct() {name = "pipe_2_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %41 = allo.stream_construct() {name = "pipe_2_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %42 = allo.stream_construct() {name = "pipe_2_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %43 = allo.stream_construct() {name = "pipe_2_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %44 = allo.stream_construct() {name = "pipe_2_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %45 = allo.stream_construct() {name = "pipe_3_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %46 = allo.stream_construct() {name = "pipe_3_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %47 = allo.stream_construct() {name = "pipe_3_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %48 = allo.stream_construct() {name = "pipe_3_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %49 = allo.stream_construct() {name = "pipe_3_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %50 = allo.stream_construct() {name = "pipe_3_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %51 = allo.stream_construct() {name = "pipe_3_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %52 = allo.stream_construct() {name = "pipe_3_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %53 = allo.stream_construct() {name = "pipe_3_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %54 = allo.stream_construct() {name = "pipe_3_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %55 = allo.stream_construct() {name = "pipe_3_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %56 = allo.stream_construct() {name = "pipe_3_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %57 = allo.stream_construct() {name = "pipe_3_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %58 = allo.stream_construct() {name = "pipe_3_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %59 = allo.stream_construct() {name = "pipe_3_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %60 = allo.stream_construct() {name = "pipe_4_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %61 = allo.stream_construct() {name = "pipe_4_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %62 = allo.stream_construct() {name = "pipe_4_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %63 = allo.stream_construct() {name = "pipe_4_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %64 = allo.stream_construct() {name = "pipe_4_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %65 = allo.stream_construct() {name = "pipe_4_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %66 = allo.stream_construct() {name = "pipe_4_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %67 = allo.stream_construct() {name = "pipe_4_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %68 = allo.stream_construct() {name = "pipe_4_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %69 = allo.stream_construct() {name = "pipe_4_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %70 = allo.stream_construct() {name = "pipe_4_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %71 = allo.stream_construct() {name = "pipe_4_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %72 = allo.stream_construct() {name = "pipe_4_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %73 = allo.stream_construct() {name = "pipe_4_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %74 = allo.stream_construct() {name = "pipe_4_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %75 = allo.stream_construct() {name = "pipe_5_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %76 = allo.stream_construct() {name = "pipe_5_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %77 = allo.stream_construct() {name = "pipe_5_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %78 = allo.stream_construct() {name = "pipe_5_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %79 = allo.stream_construct() {name = "pipe_5_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %80 = allo.stream_construct() {name = "pipe_5_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %81 = allo.stream_construct() {name = "pipe_5_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %82 = allo.stream_construct() {name = "pipe_5_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %83 = allo.stream_construct() {name = "pipe_5_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %84 = allo.stream_construct() {name = "pipe_5_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %85 = allo.stream_construct() {name = "pipe_5_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %86 = allo.stream_construct() {name = "pipe_5_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %87 = allo.stream_construct() {name = "pipe_5_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %88 = allo.stream_construct() {name = "pipe_5_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %89 = allo.stream_construct() {name = "pipe_5_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %90 = allo.stream_construct() {name = "pipe_6_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %91 = allo.stream_construct() {name = "pipe_6_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %92 = allo.stream_construct() {name = "pipe_6_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %93 = allo.stream_construct() {name = "pipe_6_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %94 = allo.stream_construct() {name = "pipe_6_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %95 = allo.stream_construct() {name = "pipe_6_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %96 = allo.stream_construct() {name = "pipe_6_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %97 = allo.stream_construct() {name = "pipe_6_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %98 = allo.stream_construct() {name = "pipe_6_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %99 = allo.stream_construct() {name = "pipe_6_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %100 = allo.stream_construct() {name = "pipe_6_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %101 = allo.stream_construct() {name = "pipe_6_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %102 = allo.stream_construct() {name = "pipe_6_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %103 = allo.stream_construct() {name = "pipe_6_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %104 = allo.stream_construct() {name = "pipe_6_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %105 = allo.stream_construct() {name = "pipe_7_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %106 = allo.stream_construct() {name = "pipe_7_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %107 = allo.stream_construct() {name = "pipe_7_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %108 = allo.stream_construct() {name = "pipe_7_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %109 = allo.stream_construct() {name = "pipe_7_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %110 = allo.stream_construct() {name = "pipe_7_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %111 = allo.stream_construct() {name = "pipe_7_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %112 = allo.stream_construct() {name = "pipe_7_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %113 = allo.stream_construct() {name = "pipe_7_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %114 = allo.stream_construct() {name = "pipe_7_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %115 = allo.stream_construct() {name = "pipe_7_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %116 = allo.stream_construct() {name = "pipe_7_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %117 = allo.stream_construct() {name = "pipe_7_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %118 = allo.stream_construct() {name = "pipe_7_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %119 = allo.stream_construct() {name = "pipe_7_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %120 = allo.stream_construct() {name = "pipe_8_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %121 = allo.stream_construct() {name = "pipe_8_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %122 = allo.stream_construct() {name = "pipe_8_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %123 = allo.stream_construct() {name = "pipe_8_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %124 = allo.stream_construct() {name = "pipe_8_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %125 = allo.stream_construct() {name = "pipe_8_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %126 = allo.stream_construct() {name = "pipe_8_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %127 = allo.stream_construct() {name = "pipe_8_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %128 = allo.stream_construct() {name = "pipe_8_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %129 = allo.stream_construct() {name = "pipe_8_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %130 = allo.stream_construct() {name = "pipe_8_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %131 = allo.stream_construct() {name = "pipe_8_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %132 = allo.stream_construct() {name = "pipe_8_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %133 = allo.stream_construct() {name = "pipe_8_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %134 = allo.stream_construct() {name = "pipe_8_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %135 = allo.stream_construct() {name = "pipe_9_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %136 = allo.stream_construct() {name = "pipe_9_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %137 = allo.stream_construct() {name = "pipe_9_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %138 = allo.stream_construct() {name = "pipe_9_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %139 = allo.stream_construct() {name = "pipe_9_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %140 = allo.stream_construct() {name = "pipe_9_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %141 = allo.stream_construct() {name = "pipe_9_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %142 = allo.stream_construct() {name = "pipe_9_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %143 = allo.stream_construct() {name = "pipe_9_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %144 = allo.stream_construct() {name = "pipe_9_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %145 = allo.stream_construct() {name = "pipe_9_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %146 = allo.stream_construct() {name = "pipe_9_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %147 = allo.stream_construct() {name = "pipe_9_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %148 = allo.stream_construct() {name = "pipe_9_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %149 = allo.stream_construct() {name = "pipe_9_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %150 = allo.stream_construct() {name = "pipe_10_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %151 = allo.stream_construct() {name = "pipe_10_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %152 = allo.stream_construct() {name = "pipe_10_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %153 = allo.stream_construct() {name = "pipe_10_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %154 = allo.stream_construct() {name = "pipe_10_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %155 = allo.stream_construct() {name = "pipe_10_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %156 = allo.stream_construct() {name = "pipe_10_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %157 = allo.stream_construct() {name = "pipe_10_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %158 = allo.stream_construct() {name = "pipe_10_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %159 = allo.stream_construct() {name = "pipe_10_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %160 = allo.stream_construct() {name = "pipe_10_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %161 = allo.stream_construct() {name = "pipe_10_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %162 = allo.stream_construct() {name = "pipe_10_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %163 = allo.stream_construct() {name = "pipe_10_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %164 = allo.stream_construct() {name = "pipe_10_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %165 = allo.stream_construct() {name = "pipe_11_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %166 = allo.stream_construct() {name = "pipe_11_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %167 = allo.stream_construct() {name = "pipe_11_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %168 = allo.stream_construct() {name = "pipe_11_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %169 = allo.stream_construct() {name = "pipe_11_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %170 = allo.stream_construct() {name = "pipe_11_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %171 = allo.stream_construct() {name = "pipe_11_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %172 = allo.stream_construct() {name = "pipe_11_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %173 = allo.stream_construct() {name = "pipe_11_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %174 = allo.stream_construct() {name = "pipe_11_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %175 = allo.stream_construct() {name = "pipe_11_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %176 = allo.stream_construct() {name = "pipe_11_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %177 = allo.stream_construct() {name = "pipe_11_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %178 = allo.stream_construct() {name = "pipe_11_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %179 = allo.stream_construct() {name = "pipe_11_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %180 = allo.stream_construct() {name = "pipe_12_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %181 = allo.stream_construct() {name = "pipe_12_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %182 = allo.stream_construct() {name = "pipe_12_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %183 = allo.stream_construct() {name = "pipe_12_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %184 = allo.stream_construct() {name = "pipe_12_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %185 = allo.stream_construct() {name = "pipe_12_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %186 = allo.stream_construct() {name = "pipe_12_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %187 = allo.stream_construct() {name = "pipe_12_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %188 = allo.stream_construct() {name = "pipe_12_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %189 = allo.stream_construct() {name = "pipe_12_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %190 = allo.stream_construct() {name = "pipe_12_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %191 = allo.stream_construct() {name = "pipe_12_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %192 = allo.stream_construct() {name = "pipe_12_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %193 = allo.stream_construct() {name = "pipe_12_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %194 = allo.stream_construct() {name = "pipe_12_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    %195 = allo.stream_construct() {name = "pipe_13_0_0"} : !allo.stream<memref<64x64xbf16>, 2>
    %196 = allo.stream_construct() {name = "pipe_13_0_1"} : !allo.stream<memref<64x64xbf16>, 2>
    %197 = allo.stream_construct() {name = "pipe_13_0_2"} : !allo.stream<memref<64x64xbf16>, 2>
    %198 = allo.stream_construct() {name = "pipe_13_0_3"} : !allo.stream<memref<64x64xbf16>, 2>
    %199 = allo.stream_construct() {name = "pipe_13_0_4"} : !allo.stream<memref<64x64xbf16>, 2>
    %200 = allo.stream_construct() {name = "pipe_13_0_5"} : !allo.stream<memref<64x64xbf16>, 2>
    %201 = allo.stream_construct() {name = "pipe_13_0_6"} : !allo.stream<memref<64x64xbf16>, 2>
    %202 = allo.stream_construct() {name = "pipe_13_0_7"} : !allo.stream<memref<64x64xbf16>, 2>
    %203 = allo.stream_construct() {name = "pipe_13_0_8"} : !allo.stream<memref<64x64xbf16>, 2>
    %204 = allo.stream_construct() {name = "pipe_13_0_9"} : !allo.stream<memref<64x64xbf16>, 2>
    %205 = allo.stream_construct() {name = "pipe_13_0_10"} : !allo.stream<memref<64x64xbf16>, 2>
    %206 = allo.stream_construct() {name = "pipe_13_0_11"} : !allo.stream<memref<64x64xbf16>, 2>
    %207 = allo.stream_construct() {name = "pipe_13_0_12"} : !allo.stream<memref<64x64xbf16>, 2>
    %208 = allo.stream_construct() {name = "pipe_13_0_13"} : !allo.stream<memref<64x64xbf16>, 2>
    %209 = allo.stream_construct() {name = "pipe_13_0_14"} : !allo.stream<memref<64x64xbf16>, 2>
    return
  }
}
