module {
  func.func private @fill_zeros_bf16_32_64_vector(memref<32x64xbf16>)
  func.func private @matmul_scalar_bf16_bf16(memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>)
  func.func private @add_bf16_vector(memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>)
  func.func @"gemm_0_0_0-gemm_1_0_0-gemm_2_0_0-gemm_3_0_0-gemm_4_0_0-gemm_5_0_0-gemm_6_0_0-gemm_7_0_0-gemm_8_0_0-gemm_9_0_0-gemm_10_0_0-gemm_11_0_0x8"(%arg0: memref<32x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<32x64xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: memref<32x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<32x64xbf16>, %arg7: !allo.stream<memref<32x64xbf16>, 2>, %arg8: !allo.stream<memref<32x64xbf16>, 2>, %arg9: memref<32x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<32x64xbf16>, %arg12: !allo.stream<memref<32x64xbf16>, 2>, %arg13: !allo.stream<memref<32x64xbf16>, 2>, %arg14: memref<32x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<32x64xbf16>, %arg17: !allo.stream<memref<32x64xbf16>, 2>, %arg18: !allo.stream<memref<32x64xbf16>, 2>, %arg19: memref<32x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<32x64xbf16>, %arg22: !allo.stream<memref<32x64xbf16>, 2>, %arg23: !allo.stream<memref<32x64xbf16>, 2>, %arg24: memref<32x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<32x64xbf16>, %arg27: !allo.stream<memref<32x64xbf16>, 2>, %arg28: !allo.stream<memref<32x64xbf16>, 2>, %arg29: memref<32x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<32x64xbf16>, %arg32: !allo.stream<memref<32x64xbf16>, 2>, %arg33: !allo.stream<memref<32x64xbf16>, 2>, %arg34: memref<32x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<32x64xbf16>, %arg37: !allo.stream<memref<32x64xbf16>, 2>, %arg38: !allo.stream<memref<32x64xbf16>, 2>, %arg39: memref<32x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<32x64xbf16>, %arg42: !allo.stream<memref<32x64xbf16>, 2>, %arg43: !allo.stream<memref<32x64xbf16>, 2>, %arg44: memref<32x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<32x64xbf16>, %arg47: !allo.stream<memref<32x64xbf16>, 2>, %arg48: !allo.stream<memref<32x64xbf16>, 2>, %arg49: memref<32x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<32x64xbf16>, %arg52: !allo.stream<memref<32x64xbf16>, 2>, %arg53: !allo.stream<memref<32x64xbf16>, 2>, %arg54: memref<32x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<32x64xbf16>, %arg57: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    %alloc_0 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_0) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_1 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_1) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_5 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_5) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_7 = arith.constant 0.000000e+00 : bf16
    %alloc_8 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_6, %alloc_8 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_9 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_9) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg9, %arg10, %alloc_9) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_10 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_9, %alloc_8, %alloc_10) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_11 = arith.constant 0.000000e+00 : bf16
    %alloc_12 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_10, %alloc_12 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_13 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_13) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg14, %arg15, %alloc_13) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_14 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_13, %alloc_12, %alloc_14) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_15 = arith.constant 0.000000e+00 : bf16
    %alloc_16 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_14, %alloc_16 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_17 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_17) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg19, %arg20, %alloc_17) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_18 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_17, %alloc_16, %alloc_18) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_19 = arith.constant 0.000000e+00 : bf16
    %alloc_20 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_18, %alloc_20 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_21 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_21) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg24, %arg25, %alloc_21) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_22 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_21, %alloc_20, %alloc_22) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_23 = arith.constant 0.000000e+00 : bf16
    %alloc_24 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_22, %alloc_24 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_25 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_25) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg29, %arg30, %alloc_25) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_26 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_25, %alloc_24, %alloc_26) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_27 = arith.constant 0.000000e+00 : bf16
    %alloc_28 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_26, %alloc_28 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_29 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_29) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg34, %arg35, %alloc_29) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_30 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_29, %alloc_28, %alloc_30) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_31 = arith.constant 0.000000e+00 : bf16
    %alloc_32 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_30, %alloc_32 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_33 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_33) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg39, %arg40, %alloc_33) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_34 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_33, %alloc_32, %alloc_34) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_35 = arith.constant 0.000000e+00 : bf16
    %alloc_36 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_34, %alloc_36 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_37 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_37) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg44, %arg45, %alloc_37) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_38 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_37, %alloc_36, %alloc_38) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_39 = arith.constant 0.000000e+00 : bf16
    %alloc_40 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_38, %alloc_40 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_41 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_41) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg49, %arg50, %alloc_41) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_42 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_41, %alloc_40, %alloc_42) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_43 = arith.constant 0.000000e+00 : bf16
    %alloc_44 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_42, %alloc_44 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_45 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_45) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg54, %arg55, %alloc_45) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_46 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_45, %alloc_44, %alloc_46) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    memref.copy %alloc_46, %arg56 {to = "local_C"} : memref<32x64xbf16> to memref<32x64xbf16>
    return
  }
  func.func @"gemm_0_0_1-gemm_1_0_1-gemm_2_0_1-gemm_3_0_1-gemm_4_0_1-gemm_5_0_1-gemm_6_0_1-gemm_7_0_1-gemm_8_0_1-gemm_9_0_1-gemm_10_0_1-gemm_11_0_1x8"(%arg0: memref<32x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<32x64xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: memref<32x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<32x64xbf16>, %arg7: !allo.stream<memref<32x64xbf16>, 2>, %arg8: !allo.stream<memref<32x64xbf16>, 2>, %arg9: memref<32x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<32x64xbf16>, %arg12: !allo.stream<memref<32x64xbf16>, 2>, %arg13: !allo.stream<memref<32x64xbf16>, 2>, %arg14: memref<32x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<32x64xbf16>, %arg17: !allo.stream<memref<32x64xbf16>, 2>, %arg18: !allo.stream<memref<32x64xbf16>, 2>, %arg19: memref<32x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<32x64xbf16>, %arg22: !allo.stream<memref<32x64xbf16>, 2>, %arg23: !allo.stream<memref<32x64xbf16>, 2>, %arg24: memref<32x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<32x64xbf16>, %arg27: !allo.stream<memref<32x64xbf16>, 2>, %arg28: !allo.stream<memref<32x64xbf16>, 2>, %arg29: memref<32x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<32x64xbf16>, %arg32: !allo.stream<memref<32x64xbf16>, 2>, %arg33: !allo.stream<memref<32x64xbf16>, 2>, %arg34: memref<32x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<32x64xbf16>, %arg37: !allo.stream<memref<32x64xbf16>, 2>, %arg38: !allo.stream<memref<32x64xbf16>, 2>, %arg39: memref<32x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<32x64xbf16>, %arg42: !allo.stream<memref<32x64xbf16>, 2>, %arg43: !allo.stream<memref<32x64xbf16>, 2>, %arg44: memref<32x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<32x64xbf16>, %arg47: !allo.stream<memref<32x64xbf16>, 2>, %arg48: !allo.stream<memref<32x64xbf16>, 2>, %arg49: memref<32x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<32x64xbf16>, %arg52: !allo.stream<memref<32x64xbf16>, 2>, %arg53: !allo.stream<memref<32x64xbf16>, 2>, %arg54: memref<32x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<32x64xbf16>, %arg57: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    %alloc_0 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_0) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_1 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_1) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_5 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_5) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_7 = arith.constant 0.000000e+00 : bf16
    %alloc_8 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_6, %alloc_8 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_9 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_9) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg9, %arg10, %alloc_9) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_10 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_9, %alloc_8, %alloc_10) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_11 = arith.constant 0.000000e+00 : bf16
    %alloc_12 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_10, %alloc_12 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_13 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_13) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg14, %arg15, %alloc_13) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_14 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_13, %alloc_12, %alloc_14) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_15 = arith.constant 0.000000e+00 : bf16
    %alloc_16 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_14, %alloc_16 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_17 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_17) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg19, %arg20, %alloc_17) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_18 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_17, %alloc_16, %alloc_18) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_19 = arith.constant 0.000000e+00 : bf16
    %alloc_20 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_18, %alloc_20 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_21 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_21) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg24, %arg25, %alloc_21) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_22 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_21, %alloc_20, %alloc_22) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_23 = arith.constant 0.000000e+00 : bf16
    %alloc_24 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_22, %alloc_24 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_25 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_25) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg29, %arg30, %alloc_25) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_26 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_25, %alloc_24, %alloc_26) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_27 = arith.constant 0.000000e+00 : bf16
    %alloc_28 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_26, %alloc_28 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_29 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_29) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg34, %arg35, %alloc_29) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_30 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_29, %alloc_28, %alloc_30) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_31 = arith.constant 0.000000e+00 : bf16
    %alloc_32 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_30, %alloc_32 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_33 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_33) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg39, %arg40, %alloc_33) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_34 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_33, %alloc_32, %alloc_34) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_35 = arith.constant 0.000000e+00 : bf16
    %alloc_36 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_34, %alloc_36 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_37 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_37) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg44, %arg45, %alloc_37) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_38 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_37, %alloc_36, %alloc_38) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_39 = arith.constant 0.000000e+00 : bf16
    %alloc_40 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_38, %alloc_40 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_41 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_41) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg49, %arg50, %alloc_41) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_42 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_41, %alloc_40, %alloc_42) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_43 = arith.constant 0.000000e+00 : bf16
    %alloc_44 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_42, %alloc_44 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_45 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_45) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg54, %arg55, %alloc_45) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_46 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_45, %alloc_44, %alloc_46) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    memref.copy %alloc_46, %arg56 {to = "local_C"} : memref<32x64xbf16> to memref<32x64xbf16>
    return
  }
  func.func @"gemm_0_0_2-gemm_1_0_2-gemm_2_0_2-gemm_3_0_2-gemm_4_0_2-gemm_5_0_2-gemm_6_0_2-gemm_7_0_2-gemm_8_0_2-gemm_9_0_2-gemm_10_0_2-gemm_11_0_2x8"(%arg0: memref<32x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<32x64xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: memref<32x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<32x64xbf16>, %arg7: !allo.stream<memref<32x64xbf16>, 2>, %arg8: !allo.stream<memref<32x64xbf16>, 2>, %arg9: memref<32x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<32x64xbf16>, %arg12: !allo.stream<memref<32x64xbf16>, 2>, %arg13: !allo.stream<memref<32x64xbf16>, 2>, %arg14: memref<32x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<32x64xbf16>, %arg17: !allo.stream<memref<32x64xbf16>, 2>, %arg18: !allo.stream<memref<32x64xbf16>, 2>, %arg19: memref<32x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<32x64xbf16>, %arg22: !allo.stream<memref<32x64xbf16>, 2>, %arg23: !allo.stream<memref<32x64xbf16>, 2>, %arg24: memref<32x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<32x64xbf16>, %arg27: !allo.stream<memref<32x64xbf16>, 2>, %arg28: !allo.stream<memref<32x64xbf16>, 2>, %arg29: memref<32x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<32x64xbf16>, %arg32: !allo.stream<memref<32x64xbf16>, 2>, %arg33: !allo.stream<memref<32x64xbf16>, 2>, %arg34: memref<32x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<32x64xbf16>, %arg37: !allo.stream<memref<32x64xbf16>, 2>, %arg38: !allo.stream<memref<32x64xbf16>, 2>, %arg39: memref<32x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<32x64xbf16>, %arg42: !allo.stream<memref<32x64xbf16>, 2>, %arg43: !allo.stream<memref<32x64xbf16>, 2>, %arg44: memref<32x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<32x64xbf16>, %arg47: !allo.stream<memref<32x64xbf16>, 2>, %arg48: !allo.stream<memref<32x64xbf16>, 2>, %arg49: memref<32x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<32x64xbf16>, %arg52: !allo.stream<memref<32x64xbf16>, 2>, %arg53: !allo.stream<memref<32x64xbf16>, 2>, %arg54: memref<32x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<32x64xbf16>, %arg57: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    %alloc_0 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_0) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_1 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_1) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_5 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_5) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_7 = arith.constant 0.000000e+00 : bf16
    %alloc_8 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_6, %alloc_8 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_9 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_9) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg9, %arg10, %alloc_9) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_10 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_9, %alloc_8, %alloc_10) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_11 = arith.constant 0.000000e+00 : bf16
    %alloc_12 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_10, %alloc_12 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_13 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_13) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg14, %arg15, %alloc_13) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_14 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_13, %alloc_12, %alloc_14) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_15 = arith.constant 0.000000e+00 : bf16
    %alloc_16 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_14, %alloc_16 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_17 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_17) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg19, %arg20, %alloc_17) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_18 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_17, %alloc_16, %alloc_18) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_19 = arith.constant 0.000000e+00 : bf16
    %alloc_20 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_18, %alloc_20 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_21 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_21) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg24, %arg25, %alloc_21) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_22 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_21, %alloc_20, %alloc_22) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_23 = arith.constant 0.000000e+00 : bf16
    %alloc_24 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_22, %alloc_24 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_25 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_25) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg29, %arg30, %alloc_25) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_26 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_25, %alloc_24, %alloc_26) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_27 = arith.constant 0.000000e+00 : bf16
    %alloc_28 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_26, %alloc_28 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_29 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_29) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg34, %arg35, %alloc_29) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_30 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_29, %alloc_28, %alloc_30) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_31 = arith.constant 0.000000e+00 : bf16
    %alloc_32 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_30, %alloc_32 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_33 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_33) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg39, %arg40, %alloc_33) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_34 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_33, %alloc_32, %alloc_34) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_35 = arith.constant 0.000000e+00 : bf16
    %alloc_36 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_34, %alloc_36 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_37 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_37) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg44, %arg45, %alloc_37) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_38 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_37, %alloc_36, %alloc_38) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_39 = arith.constant 0.000000e+00 : bf16
    %alloc_40 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_38, %alloc_40 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_41 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_41) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg49, %arg50, %alloc_41) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_42 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_41, %alloc_40, %alloc_42) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_43 = arith.constant 0.000000e+00 : bf16
    %alloc_44 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_42, %alloc_44 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_45 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_45) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg54, %arg55, %alloc_45) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_46 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_45, %alloc_44, %alloc_46) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    memref.copy %alloc_46, %arg56 {to = "local_C"} : memref<32x64xbf16> to memref<32x64xbf16>
    return
  }
  func.func @"gemm_0_0_3-gemm_1_0_3-gemm_2_0_3-gemm_3_0_3-gemm_4_0_3-gemm_5_0_3-gemm_6_0_3-gemm_7_0_3-gemm_8_0_3-gemm_9_0_3-gemm_10_0_3-gemm_11_0_3x8"(%arg0: memref<32x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<32x64xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: memref<32x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<32x64xbf16>, %arg7: !allo.stream<memref<32x64xbf16>, 2>, %arg8: !allo.stream<memref<32x64xbf16>, 2>, %arg9: memref<32x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<32x64xbf16>, %arg12: !allo.stream<memref<32x64xbf16>, 2>, %arg13: !allo.stream<memref<32x64xbf16>, 2>, %arg14: memref<32x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<32x64xbf16>, %arg17: !allo.stream<memref<32x64xbf16>, 2>, %arg18: !allo.stream<memref<32x64xbf16>, 2>, %arg19: memref<32x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<32x64xbf16>, %arg22: !allo.stream<memref<32x64xbf16>, 2>, %arg23: !allo.stream<memref<32x64xbf16>, 2>, %arg24: memref<32x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<32x64xbf16>, %arg27: !allo.stream<memref<32x64xbf16>, 2>, %arg28: !allo.stream<memref<32x64xbf16>, 2>, %arg29: memref<32x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<32x64xbf16>, %arg32: !allo.stream<memref<32x64xbf16>, 2>, %arg33: !allo.stream<memref<32x64xbf16>, 2>, %arg34: memref<32x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<32x64xbf16>, %arg37: !allo.stream<memref<32x64xbf16>, 2>, %arg38: !allo.stream<memref<32x64xbf16>, 2>, %arg39: memref<32x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<32x64xbf16>, %arg42: !allo.stream<memref<32x64xbf16>, 2>, %arg43: !allo.stream<memref<32x64xbf16>, 2>, %arg44: memref<32x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<32x64xbf16>, %arg47: !allo.stream<memref<32x64xbf16>, 2>, %arg48: !allo.stream<memref<32x64xbf16>, 2>, %arg49: memref<32x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<32x64xbf16>, %arg52: !allo.stream<memref<32x64xbf16>, 2>, %arg53: !allo.stream<memref<32x64xbf16>, 2>, %arg54: memref<32x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<32x64xbf16>, %arg57: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    %alloc_0 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_0) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_1 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_1) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg0, %arg1, %alloc_1) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_1, %alloc, %alloc_2) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_3 = arith.constant 0.000000e+00 : bf16
    %alloc_4 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_2, %alloc_4 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_5 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_5) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg4, %arg5, %alloc_5) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_6 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_5, %alloc_4, %alloc_6) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_7 = arith.constant 0.000000e+00 : bf16
    %alloc_8 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_6, %alloc_8 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_9 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_9) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg9, %arg10, %alloc_9) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_10 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_9, %alloc_8, %alloc_10) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_11 = arith.constant 0.000000e+00 : bf16
    %alloc_12 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_10, %alloc_12 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_13 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_13) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg14, %arg15, %alloc_13) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_14 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_13, %alloc_12, %alloc_14) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_15 = arith.constant 0.000000e+00 : bf16
    %alloc_16 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_14, %alloc_16 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_17 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_17) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg19, %arg20, %alloc_17) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_18 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_17, %alloc_16, %alloc_18) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_19 = arith.constant 0.000000e+00 : bf16
    %alloc_20 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_18, %alloc_20 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_21 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_21) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg24, %arg25, %alloc_21) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_22 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_21, %alloc_20, %alloc_22) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_23 = arith.constant 0.000000e+00 : bf16
    %alloc_24 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_22, %alloc_24 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_25 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_25) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg29, %arg30, %alloc_25) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_26 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_25, %alloc_24, %alloc_26) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_27 = arith.constant 0.000000e+00 : bf16
    %alloc_28 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_26, %alloc_28 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_29 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_29) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg34, %arg35, %alloc_29) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_30 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_29, %alloc_28, %alloc_30) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_31 = arith.constant 0.000000e+00 : bf16
    %alloc_32 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_30, %alloc_32 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_33 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_33) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg39, %arg40, %alloc_33) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_34 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_33, %alloc_32, %alloc_34) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_35 = arith.constant 0.000000e+00 : bf16
    %alloc_36 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_34, %alloc_36 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_37 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_37) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg44, %arg45, %alloc_37) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_38 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_37, %alloc_36, %alloc_38) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_39 = arith.constant 0.000000e+00 : bf16
    %alloc_40 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_38, %alloc_40 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_41 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_41) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg49, %arg50, %alloc_41) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_42 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_41, %alloc_40, %alloc_42) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    %cst_43 = arith.constant 0.000000e+00 : bf16
    %alloc_44 = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    memref.copy %alloc_42, %alloc_44 {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_45 = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc_45) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_scalar_bf16_bf16(%arg54, %arg55, %alloc_45) {lib = "matmul_scalar_bf16_bf16"} : (memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>) -> ()
    %alloc_46 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    call @add_bf16_vector(%alloc_45, %alloc_44, %alloc_46) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    memref.copy %alloc_46, %arg56 {to = "local_C"} : memref<32x64xbf16> to memref<32x64xbf16>
    return
  }
  func.func @top(%arg0: memref<32x768xbf16>, %arg1: memref<768x2048xbf16>, %arg2: memref<32x2048xbf16>) attributes {dataflow, itypes = "___"} {
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
    %15 = allo.stream_construct() {name = "pipe_0_0_15"} : !allo.stream<memref<32x64xbf16>, 2>
    %16 = allo.stream_construct() {name = "pipe_0_0_16"} : !allo.stream<memref<32x64xbf16>, 2>
    %17 = allo.stream_construct() {name = "pipe_0_0_17"} : !allo.stream<memref<32x64xbf16>, 2>
    %18 = allo.stream_construct() {name = "pipe_0_0_18"} : !allo.stream<memref<32x64xbf16>, 2>
    %19 = allo.stream_construct() {name = "pipe_0_0_19"} : !allo.stream<memref<32x64xbf16>, 2>
    %20 = allo.stream_construct() {name = "pipe_0_0_20"} : !allo.stream<memref<32x64xbf16>, 2>
    %21 = allo.stream_construct() {name = "pipe_0_0_21"} : !allo.stream<memref<32x64xbf16>, 2>
    %22 = allo.stream_construct() {name = "pipe_0_0_22"} : !allo.stream<memref<32x64xbf16>, 2>
    %23 = allo.stream_construct() {name = "pipe_0_0_23"} : !allo.stream<memref<32x64xbf16>, 2>
    %24 = allo.stream_construct() {name = "pipe_0_0_24"} : !allo.stream<memref<32x64xbf16>, 2>
    %25 = allo.stream_construct() {name = "pipe_0_0_25"} : !allo.stream<memref<32x64xbf16>, 2>
    %26 = allo.stream_construct() {name = "pipe_0_0_26"} : !allo.stream<memref<32x64xbf16>, 2>
    %27 = allo.stream_construct() {name = "pipe_0_0_27"} : !allo.stream<memref<32x64xbf16>, 2>
    %28 = allo.stream_construct() {name = "pipe_0_0_28"} : !allo.stream<memref<32x64xbf16>, 2>
    %29 = allo.stream_construct() {name = "pipe_0_0_29"} : !allo.stream<memref<32x64xbf16>, 2>
    %30 = allo.stream_construct() {name = "pipe_0_0_30"} : !allo.stream<memref<32x64xbf16>, 2>
    %31 = allo.stream_construct() {name = "pipe_0_0_31"} : !allo.stream<memref<32x64xbf16>, 2>
    %32 = allo.stream_construct() {name = "pipe_1_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %33 = allo.stream_construct() {name = "pipe_1_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %34 = allo.stream_construct() {name = "pipe_1_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %35 = allo.stream_construct() {name = "pipe_1_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %36 = allo.stream_construct() {name = "pipe_1_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %37 = allo.stream_construct() {name = "pipe_1_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %38 = allo.stream_construct() {name = "pipe_1_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %39 = allo.stream_construct() {name = "pipe_1_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %40 = allo.stream_construct() {name = "pipe_1_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %41 = allo.stream_construct() {name = "pipe_1_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %42 = allo.stream_construct() {name = "pipe_1_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %43 = allo.stream_construct() {name = "pipe_1_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %44 = allo.stream_construct() {name = "pipe_1_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %45 = allo.stream_construct() {name = "pipe_1_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %46 = allo.stream_construct() {name = "pipe_1_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    %47 = allo.stream_construct() {name = "pipe_1_0_15"} : !allo.stream<memref<32x64xbf16>, 2>
    %48 = allo.stream_construct() {name = "pipe_1_0_16"} : !allo.stream<memref<32x64xbf16>, 2>
    %49 = allo.stream_construct() {name = "pipe_1_0_17"} : !allo.stream<memref<32x64xbf16>, 2>
    %50 = allo.stream_construct() {name = "pipe_1_0_18"} : !allo.stream<memref<32x64xbf16>, 2>
    %51 = allo.stream_construct() {name = "pipe_1_0_19"} : !allo.stream<memref<32x64xbf16>, 2>
    %52 = allo.stream_construct() {name = "pipe_1_0_20"} : !allo.stream<memref<32x64xbf16>, 2>
    %53 = allo.stream_construct() {name = "pipe_1_0_21"} : !allo.stream<memref<32x64xbf16>, 2>
    %54 = allo.stream_construct() {name = "pipe_1_0_22"} : !allo.stream<memref<32x64xbf16>, 2>
    %55 = allo.stream_construct() {name = "pipe_1_0_23"} : !allo.stream<memref<32x64xbf16>, 2>
    %56 = allo.stream_construct() {name = "pipe_1_0_24"} : !allo.stream<memref<32x64xbf16>, 2>
    %57 = allo.stream_construct() {name = "pipe_1_0_25"} : !allo.stream<memref<32x64xbf16>, 2>
    %58 = allo.stream_construct() {name = "pipe_1_0_26"} : !allo.stream<memref<32x64xbf16>, 2>
    %59 = allo.stream_construct() {name = "pipe_1_0_27"} : !allo.stream<memref<32x64xbf16>, 2>
    %60 = allo.stream_construct() {name = "pipe_1_0_28"} : !allo.stream<memref<32x64xbf16>, 2>
    %61 = allo.stream_construct() {name = "pipe_1_0_29"} : !allo.stream<memref<32x64xbf16>, 2>
    %62 = allo.stream_construct() {name = "pipe_1_0_30"} : !allo.stream<memref<32x64xbf16>, 2>
    %63 = allo.stream_construct() {name = "pipe_1_0_31"} : !allo.stream<memref<32x64xbf16>, 2>
    %64 = allo.stream_construct() {name = "pipe_2_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %65 = allo.stream_construct() {name = "pipe_2_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %66 = allo.stream_construct() {name = "pipe_2_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %67 = allo.stream_construct() {name = "pipe_2_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %68 = allo.stream_construct() {name = "pipe_2_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %69 = allo.stream_construct() {name = "pipe_2_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %70 = allo.stream_construct() {name = "pipe_2_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %71 = allo.stream_construct() {name = "pipe_2_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %72 = allo.stream_construct() {name = "pipe_2_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %73 = allo.stream_construct() {name = "pipe_2_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %74 = allo.stream_construct() {name = "pipe_2_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %75 = allo.stream_construct() {name = "pipe_2_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %76 = allo.stream_construct() {name = "pipe_2_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %77 = allo.stream_construct() {name = "pipe_2_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %78 = allo.stream_construct() {name = "pipe_2_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    %79 = allo.stream_construct() {name = "pipe_2_0_15"} : !allo.stream<memref<32x64xbf16>, 2>
    %80 = allo.stream_construct() {name = "pipe_2_0_16"} : !allo.stream<memref<32x64xbf16>, 2>
    %81 = allo.stream_construct() {name = "pipe_2_0_17"} : !allo.stream<memref<32x64xbf16>, 2>
    %82 = allo.stream_construct() {name = "pipe_2_0_18"} : !allo.stream<memref<32x64xbf16>, 2>
    %83 = allo.stream_construct() {name = "pipe_2_0_19"} : !allo.stream<memref<32x64xbf16>, 2>
    %84 = allo.stream_construct() {name = "pipe_2_0_20"} : !allo.stream<memref<32x64xbf16>, 2>
    %85 = allo.stream_construct() {name = "pipe_2_0_21"} : !allo.stream<memref<32x64xbf16>, 2>
    %86 = allo.stream_construct() {name = "pipe_2_0_22"} : !allo.stream<memref<32x64xbf16>, 2>
    %87 = allo.stream_construct() {name = "pipe_2_0_23"} : !allo.stream<memref<32x64xbf16>, 2>
    %88 = allo.stream_construct() {name = "pipe_2_0_24"} : !allo.stream<memref<32x64xbf16>, 2>
    %89 = allo.stream_construct() {name = "pipe_2_0_25"} : !allo.stream<memref<32x64xbf16>, 2>
    %90 = allo.stream_construct() {name = "pipe_2_0_26"} : !allo.stream<memref<32x64xbf16>, 2>
    %91 = allo.stream_construct() {name = "pipe_2_0_27"} : !allo.stream<memref<32x64xbf16>, 2>
    %92 = allo.stream_construct() {name = "pipe_2_0_28"} : !allo.stream<memref<32x64xbf16>, 2>
    %93 = allo.stream_construct() {name = "pipe_2_0_29"} : !allo.stream<memref<32x64xbf16>, 2>
    %94 = allo.stream_construct() {name = "pipe_2_0_30"} : !allo.stream<memref<32x64xbf16>, 2>
    %95 = allo.stream_construct() {name = "pipe_2_0_31"} : !allo.stream<memref<32x64xbf16>, 2>
    %96 = allo.stream_construct() {name = "pipe_3_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %97 = allo.stream_construct() {name = "pipe_3_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %98 = allo.stream_construct() {name = "pipe_3_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %99 = allo.stream_construct() {name = "pipe_3_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %100 = allo.stream_construct() {name = "pipe_3_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %101 = allo.stream_construct() {name = "pipe_3_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %102 = allo.stream_construct() {name = "pipe_3_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %103 = allo.stream_construct() {name = "pipe_3_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %104 = allo.stream_construct() {name = "pipe_3_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %105 = allo.stream_construct() {name = "pipe_3_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %106 = allo.stream_construct() {name = "pipe_3_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %107 = allo.stream_construct() {name = "pipe_3_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %108 = allo.stream_construct() {name = "pipe_3_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %109 = allo.stream_construct() {name = "pipe_3_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %110 = allo.stream_construct() {name = "pipe_3_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    %111 = allo.stream_construct() {name = "pipe_3_0_15"} : !allo.stream<memref<32x64xbf16>, 2>
    %112 = allo.stream_construct() {name = "pipe_3_0_16"} : !allo.stream<memref<32x64xbf16>, 2>
    %113 = allo.stream_construct() {name = "pipe_3_0_17"} : !allo.stream<memref<32x64xbf16>, 2>
    %114 = allo.stream_construct() {name = "pipe_3_0_18"} : !allo.stream<memref<32x64xbf16>, 2>
    %115 = allo.stream_construct() {name = "pipe_3_0_19"} : !allo.stream<memref<32x64xbf16>, 2>
    %116 = allo.stream_construct() {name = "pipe_3_0_20"} : !allo.stream<memref<32x64xbf16>, 2>
    %117 = allo.stream_construct() {name = "pipe_3_0_21"} : !allo.stream<memref<32x64xbf16>, 2>
    %118 = allo.stream_construct() {name = "pipe_3_0_22"} : !allo.stream<memref<32x64xbf16>, 2>
    %119 = allo.stream_construct() {name = "pipe_3_0_23"} : !allo.stream<memref<32x64xbf16>, 2>
    %120 = allo.stream_construct() {name = "pipe_3_0_24"} : !allo.stream<memref<32x64xbf16>, 2>
    %121 = allo.stream_construct() {name = "pipe_3_0_25"} : !allo.stream<memref<32x64xbf16>, 2>
    %122 = allo.stream_construct() {name = "pipe_3_0_26"} : !allo.stream<memref<32x64xbf16>, 2>
    %123 = allo.stream_construct() {name = "pipe_3_0_27"} : !allo.stream<memref<32x64xbf16>, 2>
    %124 = allo.stream_construct() {name = "pipe_3_0_28"} : !allo.stream<memref<32x64xbf16>, 2>
    %125 = allo.stream_construct() {name = "pipe_3_0_29"} : !allo.stream<memref<32x64xbf16>, 2>
    %126 = allo.stream_construct() {name = "pipe_3_0_30"} : !allo.stream<memref<32x64xbf16>, 2>
    %127 = allo.stream_construct() {name = "pipe_3_0_31"} : !allo.stream<memref<32x64xbf16>, 2>
    %128 = allo.stream_construct() {name = "pipe_4_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %129 = allo.stream_construct() {name = "pipe_4_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %130 = allo.stream_construct() {name = "pipe_4_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %131 = allo.stream_construct() {name = "pipe_4_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %132 = allo.stream_construct() {name = "pipe_4_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %133 = allo.stream_construct() {name = "pipe_4_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %134 = allo.stream_construct() {name = "pipe_4_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %135 = allo.stream_construct() {name = "pipe_4_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %136 = allo.stream_construct() {name = "pipe_4_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %137 = allo.stream_construct() {name = "pipe_4_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %138 = allo.stream_construct() {name = "pipe_4_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %139 = allo.stream_construct() {name = "pipe_4_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %140 = allo.stream_construct() {name = "pipe_4_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %141 = allo.stream_construct() {name = "pipe_4_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %142 = allo.stream_construct() {name = "pipe_4_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    %143 = allo.stream_construct() {name = "pipe_4_0_15"} : !allo.stream<memref<32x64xbf16>, 2>
    %144 = allo.stream_construct() {name = "pipe_4_0_16"} : !allo.stream<memref<32x64xbf16>, 2>
    %145 = allo.stream_construct() {name = "pipe_4_0_17"} : !allo.stream<memref<32x64xbf16>, 2>
    %146 = allo.stream_construct() {name = "pipe_4_0_18"} : !allo.stream<memref<32x64xbf16>, 2>
    %147 = allo.stream_construct() {name = "pipe_4_0_19"} : !allo.stream<memref<32x64xbf16>, 2>
    %148 = allo.stream_construct() {name = "pipe_4_0_20"} : !allo.stream<memref<32x64xbf16>, 2>
    %149 = allo.stream_construct() {name = "pipe_4_0_21"} : !allo.stream<memref<32x64xbf16>, 2>
    %150 = allo.stream_construct() {name = "pipe_4_0_22"} : !allo.stream<memref<32x64xbf16>, 2>
    %151 = allo.stream_construct() {name = "pipe_4_0_23"} : !allo.stream<memref<32x64xbf16>, 2>
    %152 = allo.stream_construct() {name = "pipe_4_0_24"} : !allo.stream<memref<32x64xbf16>, 2>
    %153 = allo.stream_construct() {name = "pipe_4_0_25"} : !allo.stream<memref<32x64xbf16>, 2>
    %154 = allo.stream_construct() {name = "pipe_4_0_26"} : !allo.stream<memref<32x64xbf16>, 2>
    %155 = allo.stream_construct() {name = "pipe_4_0_27"} : !allo.stream<memref<32x64xbf16>, 2>
    %156 = allo.stream_construct() {name = "pipe_4_0_28"} : !allo.stream<memref<32x64xbf16>, 2>
    %157 = allo.stream_construct() {name = "pipe_4_0_29"} : !allo.stream<memref<32x64xbf16>, 2>
    %158 = allo.stream_construct() {name = "pipe_4_0_30"} : !allo.stream<memref<32x64xbf16>, 2>
    %159 = allo.stream_construct() {name = "pipe_4_0_31"} : !allo.stream<memref<32x64xbf16>, 2>
    %160 = allo.stream_construct() {name = "pipe_5_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %161 = allo.stream_construct() {name = "pipe_5_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %162 = allo.stream_construct() {name = "pipe_5_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %163 = allo.stream_construct() {name = "pipe_5_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %164 = allo.stream_construct() {name = "pipe_5_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %165 = allo.stream_construct() {name = "pipe_5_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %166 = allo.stream_construct() {name = "pipe_5_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %167 = allo.stream_construct() {name = "pipe_5_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %168 = allo.stream_construct() {name = "pipe_5_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %169 = allo.stream_construct() {name = "pipe_5_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %170 = allo.stream_construct() {name = "pipe_5_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %171 = allo.stream_construct() {name = "pipe_5_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %172 = allo.stream_construct() {name = "pipe_5_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %173 = allo.stream_construct() {name = "pipe_5_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %174 = allo.stream_construct() {name = "pipe_5_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    %175 = allo.stream_construct() {name = "pipe_5_0_15"} : !allo.stream<memref<32x64xbf16>, 2>
    %176 = allo.stream_construct() {name = "pipe_5_0_16"} : !allo.stream<memref<32x64xbf16>, 2>
    %177 = allo.stream_construct() {name = "pipe_5_0_17"} : !allo.stream<memref<32x64xbf16>, 2>
    %178 = allo.stream_construct() {name = "pipe_5_0_18"} : !allo.stream<memref<32x64xbf16>, 2>
    %179 = allo.stream_construct() {name = "pipe_5_0_19"} : !allo.stream<memref<32x64xbf16>, 2>
    %180 = allo.stream_construct() {name = "pipe_5_0_20"} : !allo.stream<memref<32x64xbf16>, 2>
    %181 = allo.stream_construct() {name = "pipe_5_0_21"} : !allo.stream<memref<32x64xbf16>, 2>
    %182 = allo.stream_construct() {name = "pipe_5_0_22"} : !allo.stream<memref<32x64xbf16>, 2>
    %183 = allo.stream_construct() {name = "pipe_5_0_23"} : !allo.stream<memref<32x64xbf16>, 2>
    %184 = allo.stream_construct() {name = "pipe_5_0_24"} : !allo.stream<memref<32x64xbf16>, 2>
    %185 = allo.stream_construct() {name = "pipe_5_0_25"} : !allo.stream<memref<32x64xbf16>, 2>
    %186 = allo.stream_construct() {name = "pipe_5_0_26"} : !allo.stream<memref<32x64xbf16>, 2>
    %187 = allo.stream_construct() {name = "pipe_5_0_27"} : !allo.stream<memref<32x64xbf16>, 2>
    %188 = allo.stream_construct() {name = "pipe_5_0_28"} : !allo.stream<memref<32x64xbf16>, 2>
    %189 = allo.stream_construct() {name = "pipe_5_0_29"} : !allo.stream<memref<32x64xbf16>, 2>
    %190 = allo.stream_construct() {name = "pipe_5_0_30"} : !allo.stream<memref<32x64xbf16>, 2>
    %191 = allo.stream_construct() {name = "pipe_5_0_31"} : !allo.stream<memref<32x64xbf16>, 2>
    %192 = allo.stream_construct() {name = "pipe_6_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %193 = allo.stream_construct() {name = "pipe_6_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %194 = allo.stream_construct() {name = "pipe_6_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %195 = allo.stream_construct() {name = "pipe_6_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %196 = allo.stream_construct() {name = "pipe_6_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %197 = allo.stream_construct() {name = "pipe_6_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %198 = allo.stream_construct() {name = "pipe_6_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %199 = allo.stream_construct() {name = "pipe_6_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %200 = allo.stream_construct() {name = "pipe_6_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %201 = allo.stream_construct() {name = "pipe_6_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %202 = allo.stream_construct() {name = "pipe_6_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %203 = allo.stream_construct() {name = "pipe_6_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %204 = allo.stream_construct() {name = "pipe_6_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %205 = allo.stream_construct() {name = "pipe_6_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %206 = allo.stream_construct() {name = "pipe_6_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    %207 = allo.stream_construct() {name = "pipe_6_0_15"} : !allo.stream<memref<32x64xbf16>, 2>
    %208 = allo.stream_construct() {name = "pipe_6_0_16"} : !allo.stream<memref<32x64xbf16>, 2>
    %209 = allo.stream_construct() {name = "pipe_6_0_17"} : !allo.stream<memref<32x64xbf16>, 2>
    %210 = allo.stream_construct() {name = "pipe_6_0_18"} : !allo.stream<memref<32x64xbf16>, 2>
    %211 = allo.stream_construct() {name = "pipe_6_0_19"} : !allo.stream<memref<32x64xbf16>, 2>
    %212 = allo.stream_construct() {name = "pipe_6_0_20"} : !allo.stream<memref<32x64xbf16>, 2>
    %213 = allo.stream_construct() {name = "pipe_6_0_21"} : !allo.stream<memref<32x64xbf16>, 2>
    %214 = allo.stream_construct() {name = "pipe_6_0_22"} : !allo.stream<memref<32x64xbf16>, 2>
    %215 = allo.stream_construct() {name = "pipe_6_0_23"} : !allo.stream<memref<32x64xbf16>, 2>
    %216 = allo.stream_construct() {name = "pipe_6_0_24"} : !allo.stream<memref<32x64xbf16>, 2>
    %217 = allo.stream_construct() {name = "pipe_6_0_25"} : !allo.stream<memref<32x64xbf16>, 2>
    %218 = allo.stream_construct() {name = "pipe_6_0_26"} : !allo.stream<memref<32x64xbf16>, 2>
    %219 = allo.stream_construct() {name = "pipe_6_0_27"} : !allo.stream<memref<32x64xbf16>, 2>
    %220 = allo.stream_construct() {name = "pipe_6_0_28"} : !allo.stream<memref<32x64xbf16>, 2>
    %221 = allo.stream_construct() {name = "pipe_6_0_29"} : !allo.stream<memref<32x64xbf16>, 2>
    %222 = allo.stream_construct() {name = "pipe_6_0_30"} : !allo.stream<memref<32x64xbf16>, 2>
    %223 = allo.stream_construct() {name = "pipe_6_0_31"} : !allo.stream<memref<32x64xbf16>, 2>
    %224 = allo.stream_construct() {name = "pipe_7_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %225 = allo.stream_construct() {name = "pipe_7_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %226 = allo.stream_construct() {name = "pipe_7_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %227 = allo.stream_construct() {name = "pipe_7_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %228 = allo.stream_construct() {name = "pipe_7_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %229 = allo.stream_construct() {name = "pipe_7_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %230 = allo.stream_construct() {name = "pipe_7_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %231 = allo.stream_construct() {name = "pipe_7_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %232 = allo.stream_construct() {name = "pipe_7_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %233 = allo.stream_construct() {name = "pipe_7_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %234 = allo.stream_construct() {name = "pipe_7_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %235 = allo.stream_construct() {name = "pipe_7_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %236 = allo.stream_construct() {name = "pipe_7_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %237 = allo.stream_construct() {name = "pipe_7_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %238 = allo.stream_construct() {name = "pipe_7_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    %239 = allo.stream_construct() {name = "pipe_7_0_15"} : !allo.stream<memref<32x64xbf16>, 2>
    %240 = allo.stream_construct() {name = "pipe_7_0_16"} : !allo.stream<memref<32x64xbf16>, 2>
    %241 = allo.stream_construct() {name = "pipe_7_0_17"} : !allo.stream<memref<32x64xbf16>, 2>
    %242 = allo.stream_construct() {name = "pipe_7_0_18"} : !allo.stream<memref<32x64xbf16>, 2>
    %243 = allo.stream_construct() {name = "pipe_7_0_19"} : !allo.stream<memref<32x64xbf16>, 2>
    %244 = allo.stream_construct() {name = "pipe_7_0_20"} : !allo.stream<memref<32x64xbf16>, 2>
    %245 = allo.stream_construct() {name = "pipe_7_0_21"} : !allo.stream<memref<32x64xbf16>, 2>
    %246 = allo.stream_construct() {name = "pipe_7_0_22"} : !allo.stream<memref<32x64xbf16>, 2>
    %247 = allo.stream_construct() {name = "pipe_7_0_23"} : !allo.stream<memref<32x64xbf16>, 2>
    %248 = allo.stream_construct() {name = "pipe_7_0_24"} : !allo.stream<memref<32x64xbf16>, 2>
    %249 = allo.stream_construct() {name = "pipe_7_0_25"} : !allo.stream<memref<32x64xbf16>, 2>
    %250 = allo.stream_construct() {name = "pipe_7_0_26"} : !allo.stream<memref<32x64xbf16>, 2>
    %251 = allo.stream_construct() {name = "pipe_7_0_27"} : !allo.stream<memref<32x64xbf16>, 2>
    %252 = allo.stream_construct() {name = "pipe_7_0_28"} : !allo.stream<memref<32x64xbf16>, 2>
    %253 = allo.stream_construct() {name = "pipe_7_0_29"} : !allo.stream<memref<32x64xbf16>, 2>
    %254 = allo.stream_construct() {name = "pipe_7_0_30"} : !allo.stream<memref<32x64xbf16>, 2>
    %255 = allo.stream_construct() {name = "pipe_7_0_31"} : !allo.stream<memref<32x64xbf16>, 2>
    %256 = allo.stream_construct() {name = "pipe_8_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %257 = allo.stream_construct() {name = "pipe_8_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %258 = allo.stream_construct() {name = "pipe_8_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %259 = allo.stream_construct() {name = "pipe_8_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %260 = allo.stream_construct() {name = "pipe_8_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %261 = allo.stream_construct() {name = "pipe_8_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %262 = allo.stream_construct() {name = "pipe_8_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %263 = allo.stream_construct() {name = "pipe_8_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %264 = allo.stream_construct() {name = "pipe_8_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %265 = allo.stream_construct() {name = "pipe_8_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %266 = allo.stream_construct() {name = "pipe_8_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %267 = allo.stream_construct() {name = "pipe_8_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %268 = allo.stream_construct() {name = "pipe_8_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %269 = allo.stream_construct() {name = "pipe_8_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %270 = allo.stream_construct() {name = "pipe_8_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    %271 = allo.stream_construct() {name = "pipe_8_0_15"} : !allo.stream<memref<32x64xbf16>, 2>
    %272 = allo.stream_construct() {name = "pipe_8_0_16"} : !allo.stream<memref<32x64xbf16>, 2>
    %273 = allo.stream_construct() {name = "pipe_8_0_17"} : !allo.stream<memref<32x64xbf16>, 2>
    %274 = allo.stream_construct() {name = "pipe_8_0_18"} : !allo.stream<memref<32x64xbf16>, 2>
    %275 = allo.stream_construct() {name = "pipe_8_0_19"} : !allo.stream<memref<32x64xbf16>, 2>
    %276 = allo.stream_construct() {name = "pipe_8_0_20"} : !allo.stream<memref<32x64xbf16>, 2>
    %277 = allo.stream_construct() {name = "pipe_8_0_21"} : !allo.stream<memref<32x64xbf16>, 2>
    %278 = allo.stream_construct() {name = "pipe_8_0_22"} : !allo.stream<memref<32x64xbf16>, 2>
    %279 = allo.stream_construct() {name = "pipe_8_0_23"} : !allo.stream<memref<32x64xbf16>, 2>
    %280 = allo.stream_construct() {name = "pipe_8_0_24"} : !allo.stream<memref<32x64xbf16>, 2>
    %281 = allo.stream_construct() {name = "pipe_8_0_25"} : !allo.stream<memref<32x64xbf16>, 2>
    %282 = allo.stream_construct() {name = "pipe_8_0_26"} : !allo.stream<memref<32x64xbf16>, 2>
    %283 = allo.stream_construct() {name = "pipe_8_0_27"} : !allo.stream<memref<32x64xbf16>, 2>
    %284 = allo.stream_construct() {name = "pipe_8_0_28"} : !allo.stream<memref<32x64xbf16>, 2>
    %285 = allo.stream_construct() {name = "pipe_8_0_29"} : !allo.stream<memref<32x64xbf16>, 2>
    %286 = allo.stream_construct() {name = "pipe_8_0_30"} : !allo.stream<memref<32x64xbf16>, 2>
    %287 = allo.stream_construct() {name = "pipe_8_0_31"} : !allo.stream<memref<32x64xbf16>, 2>
    %288 = allo.stream_construct() {name = "pipe_9_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %289 = allo.stream_construct() {name = "pipe_9_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %290 = allo.stream_construct() {name = "pipe_9_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %291 = allo.stream_construct() {name = "pipe_9_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %292 = allo.stream_construct() {name = "pipe_9_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %293 = allo.stream_construct() {name = "pipe_9_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %294 = allo.stream_construct() {name = "pipe_9_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %295 = allo.stream_construct() {name = "pipe_9_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %296 = allo.stream_construct() {name = "pipe_9_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %297 = allo.stream_construct() {name = "pipe_9_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %298 = allo.stream_construct() {name = "pipe_9_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %299 = allo.stream_construct() {name = "pipe_9_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %300 = allo.stream_construct() {name = "pipe_9_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %301 = allo.stream_construct() {name = "pipe_9_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %302 = allo.stream_construct() {name = "pipe_9_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    %303 = allo.stream_construct() {name = "pipe_9_0_15"} : !allo.stream<memref<32x64xbf16>, 2>
    %304 = allo.stream_construct() {name = "pipe_9_0_16"} : !allo.stream<memref<32x64xbf16>, 2>
    %305 = allo.stream_construct() {name = "pipe_9_0_17"} : !allo.stream<memref<32x64xbf16>, 2>
    %306 = allo.stream_construct() {name = "pipe_9_0_18"} : !allo.stream<memref<32x64xbf16>, 2>
    %307 = allo.stream_construct() {name = "pipe_9_0_19"} : !allo.stream<memref<32x64xbf16>, 2>
    %308 = allo.stream_construct() {name = "pipe_9_0_20"} : !allo.stream<memref<32x64xbf16>, 2>
    %309 = allo.stream_construct() {name = "pipe_9_0_21"} : !allo.stream<memref<32x64xbf16>, 2>
    %310 = allo.stream_construct() {name = "pipe_9_0_22"} : !allo.stream<memref<32x64xbf16>, 2>
    %311 = allo.stream_construct() {name = "pipe_9_0_23"} : !allo.stream<memref<32x64xbf16>, 2>
    %312 = allo.stream_construct() {name = "pipe_9_0_24"} : !allo.stream<memref<32x64xbf16>, 2>
    %313 = allo.stream_construct() {name = "pipe_9_0_25"} : !allo.stream<memref<32x64xbf16>, 2>
    %314 = allo.stream_construct() {name = "pipe_9_0_26"} : !allo.stream<memref<32x64xbf16>, 2>
    %315 = allo.stream_construct() {name = "pipe_9_0_27"} : !allo.stream<memref<32x64xbf16>, 2>
    %316 = allo.stream_construct() {name = "pipe_9_0_28"} : !allo.stream<memref<32x64xbf16>, 2>
    %317 = allo.stream_construct() {name = "pipe_9_0_29"} : !allo.stream<memref<32x64xbf16>, 2>
    %318 = allo.stream_construct() {name = "pipe_9_0_30"} : !allo.stream<memref<32x64xbf16>, 2>
    %319 = allo.stream_construct() {name = "pipe_9_0_31"} : !allo.stream<memref<32x64xbf16>, 2>
    %320 = allo.stream_construct() {name = "pipe_10_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %321 = allo.stream_construct() {name = "pipe_10_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %322 = allo.stream_construct() {name = "pipe_10_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %323 = allo.stream_construct() {name = "pipe_10_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %324 = allo.stream_construct() {name = "pipe_10_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %325 = allo.stream_construct() {name = "pipe_10_0_5"} : !allo.stream<memref<32x64xbf16>, 2>
    %326 = allo.stream_construct() {name = "pipe_10_0_6"} : !allo.stream<memref<32x64xbf16>, 2>
    %327 = allo.stream_construct() {name = "pipe_10_0_7"} : !allo.stream<memref<32x64xbf16>, 2>
    %328 = allo.stream_construct() {name = "pipe_10_0_8"} : !allo.stream<memref<32x64xbf16>, 2>
    %329 = allo.stream_construct() {name = "pipe_10_0_9"} : !allo.stream<memref<32x64xbf16>, 2>
    %330 = allo.stream_construct() {name = "pipe_10_0_10"} : !allo.stream<memref<32x64xbf16>, 2>
    %331 = allo.stream_construct() {name = "pipe_10_0_11"} : !allo.stream<memref<32x64xbf16>, 2>
    %332 = allo.stream_construct() {name = "pipe_10_0_12"} : !allo.stream<memref<32x64xbf16>, 2>
    %333 = allo.stream_construct() {name = "pipe_10_0_13"} : !allo.stream<memref<32x64xbf16>, 2>
    %334 = allo.stream_construct() {name = "pipe_10_0_14"} : !allo.stream<memref<32x64xbf16>, 2>
    %335 = allo.stream_construct() {name = "pipe_10_0_15"} : !allo.stream<memref<32x64xbf16>, 2>
    %336 = allo.stream_construct() {name = "pipe_10_0_16"} : !allo.stream<memref<32x64xbf16>, 2>
    %337 = allo.stream_construct() {name = "pipe_10_0_17"} : !allo.stream<memref<32x64xbf16>, 2>
    %338 = allo.stream_construct() {name = "pipe_10_0_18"} : !allo.stream<memref<32x64xbf16>, 2>
    %339 = allo.stream_construct() {name = "pipe_10_0_19"} : !allo.stream<memref<32x64xbf16>, 2>
    %340 = allo.stream_construct() {name = "pipe_10_0_20"} : !allo.stream<memref<32x64xbf16>, 2>
    %341 = allo.stream_construct() {name = "pipe_10_0_21"} : !allo.stream<memref<32x64xbf16>, 2>
    %342 = allo.stream_construct() {name = "pipe_10_0_22"} : !allo.stream<memref<32x64xbf16>, 2>
    %343 = allo.stream_construct() {name = "pipe_10_0_23"} : !allo.stream<memref<32x64xbf16>, 2>
    %344 = allo.stream_construct() {name = "pipe_10_0_24"} : !allo.stream<memref<32x64xbf16>, 2>
    %345 = allo.stream_construct() {name = "pipe_10_0_25"} : !allo.stream<memref<32x64xbf16>, 2>
    %346 = allo.stream_construct() {name = "pipe_10_0_26"} : !allo.stream<memref<32x64xbf16>, 2>
    %347 = allo.stream_construct() {name = "pipe_10_0_27"} : !allo.stream<memref<32x64xbf16>, 2>
    %348 = allo.stream_construct() {name = "pipe_10_0_28"} : !allo.stream<memref<32x64xbf16>, 2>
    %349 = allo.stream_construct() {name = "pipe_10_0_29"} : !allo.stream<memref<32x64xbf16>, 2>
    %350 = allo.stream_construct() {name = "pipe_10_0_30"} : !allo.stream<memref<32x64xbf16>, 2>
    %351 = allo.stream_construct() {name = "pipe_10_0_31"} : !allo.stream<memref<32x64xbf16>, 2>
    return
  }
}
