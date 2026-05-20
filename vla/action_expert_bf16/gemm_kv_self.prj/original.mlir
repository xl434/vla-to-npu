module {
  func.func private @fill_zeros_bf16_32_64_vector(memref<32x64xbf16>)
  func.func private @matmul_scalar_bf16_bf16(memref<32x64xbf16>, memref<64x64xbf16>, memref<32x64xbf16>)
  func.func private @add_bf16_vector(memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>)
  func.func @"gemm_0_0_0-gemm_1_0_0-gemm_2_0_0-gemm_3_0_0-gemm_4_0_0-gemm_5_0_0-gemm_6_0_0-gemm_7_0_0-gemm_8_0_0-gemm_9_0_0-gemm_10_0_0-gemm_11_0_0"(%arg0: memref<32x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<32x64xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: memref<32x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<32x64xbf16>, %arg7: !allo.stream<memref<32x64xbf16>, 2>, %arg8: !allo.stream<memref<32x64xbf16>, 2>, %arg9: memref<32x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<32x64xbf16>, %arg12: !allo.stream<memref<32x64xbf16>, 2>, %arg13: !allo.stream<memref<32x64xbf16>, 2>, %arg14: memref<32x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<32x64xbf16>, %arg17: !allo.stream<memref<32x64xbf16>, 2>, %arg18: !allo.stream<memref<32x64xbf16>, 2>, %arg19: memref<32x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<32x64xbf16>, %arg22: !allo.stream<memref<32x64xbf16>, 2>, %arg23: !allo.stream<memref<32x64xbf16>, 2>, %arg24: memref<32x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<32x64xbf16>, %arg27: !allo.stream<memref<32x64xbf16>, 2>, %arg28: !allo.stream<memref<32x64xbf16>, 2>, %arg29: memref<32x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<32x64xbf16>, %arg32: !allo.stream<memref<32x64xbf16>, 2>, %arg33: !allo.stream<memref<32x64xbf16>, 2>, %arg34: memref<32x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<32x64xbf16>, %arg37: !allo.stream<memref<32x64xbf16>, 2>, %arg38: !allo.stream<memref<32x64xbf16>, 2>, %arg39: memref<32x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<32x64xbf16>, %arg42: !allo.stream<memref<32x64xbf16>, 2>, %arg43: !allo.stream<memref<32x64xbf16>, 2>, %arg44: memref<32x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<32x64xbf16>, %arg47: !allo.stream<memref<32x64xbf16>, 2>, %arg48: !allo.stream<memref<32x64xbf16>, 2>, %arg49: memref<32x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<32x64xbf16>, %arg52: !allo.stream<memref<32x64xbf16>, 2>, %arg53: !allo.stream<memref<32x64xbf16>, 2>, %arg54: memref<32x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<32x64xbf16>, %arg57: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
  func.func @"gemm_0_0_1-gemm_1_0_1-gemm_2_0_1-gemm_3_0_1-gemm_4_0_1-gemm_5_0_1-gemm_6_0_1-gemm_7_0_1-gemm_8_0_1-gemm_9_0_1-gemm_10_0_1-gemm_11_0_1"(%arg0: memref<32x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<32x64xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: memref<32x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<32x64xbf16>, %arg7: !allo.stream<memref<32x64xbf16>, 2>, %arg8: !allo.stream<memref<32x64xbf16>, 2>, %arg9: memref<32x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<32x64xbf16>, %arg12: !allo.stream<memref<32x64xbf16>, 2>, %arg13: !allo.stream<memref<32x64xbf16>, 2>, %arg14: memref<32x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<32x64xbf16>, %arg17: !allo.stream<memref<32x64xbf16>, 2>, %arg18: !allo.stream<memref<32x64xbf16>, 2>, %arg19: memref<32x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<32x64xbf16>, %arg22: !allo.stream<memref<32x64xbf16>, 2>, %arg23: !allo.stream<memref<32x64xbf16>, 2>, %arg24: memref<32x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<32x64xbf16>, %arg27: !allo.stream<memref<32x64xbf16>, 2>, %arg28: !allo.stream<memref<32x64xbf16>, 2>, %arg29: memref<32x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<32x64xbf16>, %arg32: !allo.stream<memref<32x64xbf16>, 2>, %arg33: !allo.stream<memref<32x64xbf16>, 2>, %arg34: memref<32x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<32x64xbf16>, %arg37: !allo.stream<memref<32x64xbf16>, 2>, %arg38: !allo.stream<memref<32x64xbf16>, 2>, %arg39: memref<32x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<32x64xbf16>, %arg42: !allo.stream<memref<32x64xbf16>, 2>, %arg43: !allo.stream<memref<32x64xbf16>, 2>, %arg44: memref<32x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<32x64xbf16>, %arg47: !allo.stream<memref<32x64xbf16>, 2>, %arg48: !allo.stream<memref<32x64xbf16>, 2>, %arg49: memref<32x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<32x64xbf16>, %arg52: !allo.stream<memref<32x64xbf16>, 2>, %arg53: !allo.stream<memref<32x64xbf16>, 2>, %arg54: memref<32x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<32x64xbf16>, %arg57: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
  func.func @"gemm_0_0_2-gemm_1_0_2-gemm_2_0_2-gemm_3_0_2-gemm_4_0_2-gemm_5_0_2-gemm_6_0_2-gemm_7_0_2-gemm_8_0_2-gemm_9_0_2-gemm_10_0_2-gemm_11_0_2"(%arg0: memref<32x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<32x64xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: memref<32x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<32x64xbf16>, %arg7: !allo.stream<memref<32x64xbf16>, 2>, %arg8: !allo.stream<memref<32x64xbf16>, 2>, %arg9: memref<32x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<32x64xbf16>, %arg12: !allo.stream<memref<32x64xbf16>, 2>, %arg13: !allo.stream<memref<32x64xbf16>, 2>, %arg14: memref<32x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<32x64xbf16>, %arg17: !allo.stream<memref<32x64xbf16>, 2>, %arg18: !allo.stream<memref<32x64xbf16>, 2>, %arg19: memref<32x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<32x64xbf16>, %arg22: !allo.stream<memref<32x64xbf16>, 2>, %arg23: !allo.stream<memref<32x64xbf16>, 2>, %arg24: memref<32x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<32x64xbf16>, %arg27: !allo.stream<memref<32x64xbf16>, 2>, %arg28: !allo.stream<memref<32x64xbf16>, 2>, %arg29: memref<32x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<32x64xbf16>, %arg32: !allo.stream<memref<32x64xbf16>, 2>, %arg33: !allo.stream<memref<32x64xbf16>, 2>, %arg34: memref<32x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<32x64xbf16>, %arg37: !allo.stream<memref<32x64xbf16>, 2>, %arg38: !allo.stream<memref<32x64xbf16>, 2>, %arg39: memref<32x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<32x64xbf16>, %arg42: !allo.stream<memref<32x64xbf16>, 2>, %arg43: !allo.stream<memref<32x64xbf16>, 2>, %arg44: memref<32x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<32x64xbf16>, %arg47: !allo.stream<memref<32x64xbf16>, 2>, %arg48: !allo.stream<memref<32x64xbf16>, 2>, %arg49: memref<32x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<32x64xbf16>, %arg52: !allo.stream<memref<32x64xbf16>, 2>, %arg53: !allo.stream<memref<32x64xbf16>, 2>, %arg54: memref<32x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<32x64xbf16>, %arg57: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
  func.func @"gemm_0_0_3-gemm_1_0_3-gemm_2_0_3-gemm_3_0_3-gemm_4_0_3-gemm_5_0_3-gemm_6_0_3-gemm_7_0_3-gemm_8_0_3-gemm_9_0_3-gemm_10_0_3-gemm_11_0_3"(%arg0: memref<32x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<32x64xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: memref<32x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<32x64xbf16>, %arg7: !allo.stream<memref<32x64xbf16>, 2>, %arg8: !allo.stream<memref<32x64xbf16>, 2>, %arg9: memref<32x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<32x64xbf16>, %arg12: !allo.stream<memref<32x64xbf16>, 2>, %arg13: !allo.stream<memref<32x64xbf16>, 2>, %arg14: memref<32x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<32x64xbf16>, %arg17: !allo.stream<memref<32x64xbf16>, 2>, %arg18: !allo.stream<memref<32x64xbf16>, 2>, %arg19: memref<32x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<32x64xbf16>, %arg22: !allo.stream<memref<32x64xbf16>, 2>, %arg23: !allo.stream<memref<32x64xbf16>, 2>, %arg24: memref<32x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<32x64xbf16>, %arg27: !allo.stream<memref<32x64xbf16>, 2>, %arg28: !allo.stream<memref<32x64xbf16>, 2>, %arg29: memref<32x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<32x64xbf16>, %arg32: !allo.stream<memref<32x64xbf16>, 2>, %arg33: !allo.stream<memref<32x64xbf16>, 2>, %arg34: memref<32x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<32x64xbf16>, %arg37: !allo.stream<memref<32x64xbf16>, 2>, %arg38: !allo.stream<memref<32x64xbf16>, 2>, %arg39: memref<32x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<32x64xbf16>, %arg42: !allo.stream<memref<32x64xbf16>, 2>, %arg43: !allo.stream<memref<32x64xbf16>, 2>, %arg44: memref<32x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<32x64xbf16>, %arg47: !allo.stream<memref<32x64xbf16>, 2>, %arg48: !allo.stream<memref<32x64xbf16>, 2>, %arg49: memref<32x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<32x64xbf16>, %arg52: !allo.stream<memref<32x64xbf16>, 2>, %arg53: !allo.stream<memref<32x64xbf16>, 2>, %arg54: memref<32x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<32x64xbf16>, %arg57: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
  func.func @"gemm_0_0_4-gemm_1_0_4-gemm_2_0_4-gemm_3_0_4-gemm_4_0_4-gemm_5_0_4-gemm_6_0_4-gemm_7_0_4-gemm_8_0_4-gemm_9_0_4-gemm_10_0_4-gemm_11_0_4"(%arg0: memref<32x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<32x64xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: memref<32x64xbf16>, %arg5: memref<64x64xbf16>, %arg6: memref<32x64xbf16>, %arg7: !allo.stream<memref<32x64xbf16>, 2>, %arg8: !allo.stream<memref<32x64xbf16>, 2>, %arg9: memref<32x64xbf16>, %arg10: memref<64x64xbf16>, %arg11: memref<32x64xbf16>, %arg12: !allo.stream<memref<32x64xbf16>, 2>, %arg13: !allo.stream<memref<32x64xbf16>, 2>, %arg14: memref<32x64xbf16>, %arg15: memref<64x64xbf16>, %arg16: memref<32x64xbf16>, %arg17: !allo.stream<memref<32x64xbf16>, 2>, %arg18: !allo.stream<memref<32x64xbf16>, 2>, %arg19: memref<32x64xbf16>, %arg20: memref<64x64xbf16>, %arg21: memref<32x64xbf16>, %arg22: !allo.stream<memref<32x64xbf16>, 2>, %arg23: !allo.stream<memref<32x64xbf16>, 2>, %arg24: memref<32x64xbf16>, %arg25: memref<64x64xbf16>, %arg26: memref<32x64xbf16>, %arg27: !allo.stream<memref<32x64xbf16>, 2>, %arg28: !allo.stream<memref<32x64xbf16>, 2>, %arg29: memref<32x64xbf16>, %arg30: memref<64x64xbf16>, %arg31: memref<32x64xbf16>, %arg32: !allo.stream<memref<32x64xbf16>, 2>, %arg33: !allo.stream<memref<32x64xbf16>, 2>, %arg34: memref<32x64xbf16>, %arg35: memref<64x64xbf16>, %arg36: memref<32x64xbf16>, %arg37: !allo.stream<memref<32x64xbf16>, 2>, %arg38: !allo.stream<memref<32x64xbf16>, 2>, %arg39: memref<32x64xbf16>, %arg40: memref<64x64xbf16>, %arg41: memref<32x64xbf16>, %arg42: !allo.stream<memref<32x64xbf16>, 2>, %arg43: !allo.stream<memref<32x64xbf16>, 2>, %arg44: memref<32x64xbf16>, %arg45: memref<64x64xbf16>, %arg46: memref<32x64xbf16>, %arg47: !allo.stream<memref<32x64xbf16>, 2>, %arg48: !allo.stream<memref<32x64xbf16>, 2>, %arg49: memref<32x64xbf16>, %arg50: memref<64x64xbf16>, %arg51: memref<32x64xbf16>, %arg52: !allo.stream<memref<32x64xbf16>, 2>, %arg53: !allo.stream<memref<32x64xbf16>, 2>, %arg54: memref<32x64xbf16>, %arg55: memref<64x64xbf16>, %arg56: memref<32x64xbf16>, %arg57: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], output_depth = []} {
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
  func.func @top(%arg0: memref<32x768xbf16>, %arg1: memref<768x320xbf16>, %arg2: memref<32x320xbf16>) attributes {dataflow, itypes = "___"} {
    %0 = allo.stream_construct() {name = "pipe_0_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %1 = allo.stream_construct() {name = "pipe_0_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %2 = allo.stream_construct() {name = "pipe_0_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %3 = allo.stream_construct() {name = "pipe_0_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %4 = allo.stream_construct() {name = "pipe_0_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %5 = allo.stream_construct() {name = "pipe_1_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %6 = allo.stream_construct() {name = "pipe_1_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %7 = allo.stream_construct() {name = "pipe_1_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %8 = allo.stream_construct() {name = "pipe_1_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %9 = allo.stream_construct() {name = "pipe_1_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %10 = allo.stream_construct() {name = "pipe_2_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %11 = allo.stream_construct() {name = "pipe_2_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %12 = allo.stream_construct() {name = "pipe_2_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %13 = allo.stream_construct() {name = "pipe_2_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %14 = allo.stream_construct() {name = "pipe_2_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %15 = allo.stream_construct() {name = "pipe_3_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %16 = allo.stream_construct() {name = "pipe_3_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %17 = allo.stream_construct() {name = "pipe_3_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %18 = allo.stream_construct() {name = "pipe_3_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %19 = allo.stream_construct() {name = "pipe_3_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %20 = allo.stream_construct() {name = "pipe_4_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %21 = allo.stream_construct() {name = "pipe_4_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %22 = allo.stream_construct() {name = "pipe_4_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %23 = allo.stream_construct() {name = "pipe_4_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %24 = allo.stream_construct() {name = "pipe_4_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %25 = allo.stream_construct() {name = "pipe_5_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %26 = allo.stream_construct() {name = "pipe_5_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %27 = allo.stream_construct() {name = "pipe_5_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %28 = allo.stream_construct() {name = "pipe_5_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %29 = allo.stream_construct() {name = "pipe_5_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %30 = allo.stream_construct() {name = "pipe_6_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %31 = allo.stream_construct() {name = "pipe_6_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %32 = allo.stream_construct() {name = "pipe_6_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %33 = allo.stream_construct() {name = "pipe_6_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %34 = allo.stream_construct() {name = "pipe_6_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %35 = allo.stream_construct() {name = "pipe_7_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %36 = allo.stream_construct() {name = "pipe_7_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %37 = allo.stream_construct() {name = "pipe_7_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %38 = allo.stream_construct() {name = "pipe_7_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %39 = allo.stream_construct() {name = "pipe_7_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %40 = allo.stream_construct() {name = "pipe_8_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %41 = allo.stream_construct() {name = "pipe_8_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %42 = allo.stream_construct() {name = "pipe_8_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %43 = allo.stream_construct() {name = "pipe_8_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %44 = allo.stream_construct() {name = "pipe_8_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %45 = allo.stream_construct() {name = "pipe_9_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %46 = allo.stream_construct() {name = "pipe_9_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %47 = allo.stream_construct() {name = "pipe_9_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %48 = allo.stream_construct() {name = "pipe_9_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %49 = allo.stream_construct() {name = "pipe_9_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    %50 = allo.stream_construct() {name = "pipe_10_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %51 = allo.stream_construct() {name = "pipe_10_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %52 = allo.stream_construct() {name = "pipe_10_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %53 = allo.stream_construct() {name = "pipe_10_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %54 = allo.stream_construct() {name = "pipe_10_0_4"} : !allo.stream<memref<32x64xbf16>, 2>
    return
  }
}
