module {
  func.func private @fill_zeros_f32_16_16_vector(memref<16x16xf32>)
  func.func @gemm_0_0(%arg0: memref<16x64xf32>, %arg1: memref<64x16xf32>, %arg2: memref<16x16xf32>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "___", tag = "gemm_()"} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<16x16xf32>
    call @fill_zeros_f32_16_16_vector(%alloc) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<16x64xf32>, memref<64x16xf32>) outs(%alloc : memref<16x16xf32>)
    memref.copy %alloc, %arg2 {to = "local_C"} : memref<16x16xf32> to memref<16x16xf32>
    return
  }
  func.func @gemm_0_1(%arg0: memref<16x64xf32>, %arg1: memref<64x16xf32>, %arg2: memref<16x16xf32>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<16x16xf32>
    call @fill_zeros_f32_16_16_vector(%alloc) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<16x64xf32>, memref<64x16xf32>) outs(%alloc : memref<16x16xf32>)
    memref.copy %alloc, %arg2 {to = "local_C"} : memref<16x16xf32> to memref<16x16xf32>
    return
  }
  func.func @gemm_0_2(%arg0: memref<16x64xf32>, %arg1: memref<64x16xf32>, %arg2: memref<16x16xf32>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<16x16xf32>
    call @fill_zeros_f32_16_16_vector(%alloc) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<16x64xf32>, memref<64x16xf32>) outs(%alloc : memref<16x16xf32>)
    memref.copy %alloc, %arg2 {to = "local_C"} : memref<16x16xf32> to memref<16x16xf32>
    return
  }
  func.func @gemm_0_3(%arg0: memref<16x64xf32>, %arg1: memref<64x16xf32>, %arg2: memref<16x16xf32>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<16x16xf32>
    call @fill_zeros_f32_16_16_vector(%alloc) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<16x64xf32>, memref<64x16xf32>) outs(%alloc : memref<16x16xf32>)
    memref.copy %alloc, %arg2 {to = "local_C"} : memref<16x16xf32> to memref<16x16xf32>
    return
  }
  func.func @gemm_1_0(%arg0: memref<16x64xf32>, %arg1: memref<64x16xf32>, %arg2: memref<16x16xf32>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<16x16xf32>
    call @fill_zeros_f32_16_16_vector(%alloc) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<16x64xf32>, memref<64x16xf32>) outs(%alloc : memref<16x16xf32>)
    memref.copy %alloc, %arg2 {to = "local_C"} : memref<16x16xf32> to memref<16x16xf32>
    return
  }
  func.func @gemm_1_1(%arg0: memref<16x64xf32>, %arg1: memref<64x16xf32>, %arg2: memref<16x16xf32>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<16x16xf32>
    call @fill_zeros_f32_16_16_vector(%alloc) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<16x64xf32>, memref<64x16xf32>) outs(%alloc : memref<16x16xf32>)
    memref.copy %alloc, %arg2 {to = "local_C"} : memref<16x16xf32> to memref<16x16xf32>
    return
  }
  func.func @gemm_1_2(%arg0: memref<16x64xf32>, %arg1: memref<64x16xf32>, %arg2: memref<16x16xf32>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<16x16xf32>
    call @fill_zeros_f32_16_16_vector(%alloc) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<16x64xf32>, memref<64x16xf32>) outs(%alloc : memref<16x16xf32>)
    memref.copy %alloc, %arg2 {to = "local_C"} : memref<16x16xf32> to memref<16x16xf32>
    return
  }
  func.func @gemm_1_3(%arg0: memref<16x64xf32>, %arg1: memref<64x16xf32>, %arg2: memref<16x16xf32>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<16x16xf32>
    call @fill_zeros_f32_16_16_vector(%alloc) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<16x64xf32>, memref<64x16xf32>) outs(%alloc : memref<16x16xf32>)
    memref.copy %alloc, %arg2 {to = "local_C"} : memref<16x16xf32> to memref<16x16xf32>
    return
  }
  func.func @gemm_2_0(%arg0: memref<16x64xf32>, %arg1: memref<64x16xf32>, %arg2: memref<16x16xf32>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<16x16xf32>
    call @fill_zeros_f32_16_16_vector(%alloc) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<16x64xf32>, memref<64x16xf32>) outs(%alloc : memref<16x16xf32>)
    memref.copy %alloc, %arg2 {to = "local_C"} : memref<16x16xf32> to memref<16x16xf32>
    return
  }
  func.func @gemm_2_1(%arg0: memref<16x64xf32>, %arg1: memref<64x16xf32>, %arg2: memref<16x16xf32>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<16x16xf32>
    call @fill_zeros_f32_16_16_vector(%alloc) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<16x64xf32>, memref<64x16xf32>) outs(%alloc : memref<16x16xf32>)
    memref.copy %alloc, %arg2 {to = "local_C"} : memref<16x16xf32> to memref<16x16xf32>
    return
  }
  func.func @gemm_2_2(%arg0: memref<16x64xf32>, %arg1: memref<64x16xf32>, %arg2: memref<16x16xf32>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<16x16xf32>
    call @fill_zeros_f32_16_16_vector(%alloc) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<16x64xf32>, memref<64x16xf32>) outs(%alloc : memref<16x16xf32>)
    memref.copy %alloc, %arg2 {to = "local_C"} : memref<16x16xf32> to memref<16x16xf32>
    return
  }
  func.func @gemm_2_3(%arg0: memref<16x64xf32>, %arg1: memref<64x16xf32>, %arg2: memref<16x16xf32>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<16x16xf32>
    call @fill_zeros_f32_16_16_vector(%alloc) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<16x64xf32>, memref<64x16xf32>) outs(%alloc : memref<16x16xf32>)
    memref.copy %alloc, %arg2 {to = "local_C"} : memref<16x16xf32> to memref<16x16xf32>
    return
  }
  func.func @gemm_3_0(%arg0: memref<16x64xf32>, %arg1: memref<64x16xf32>, %arg2: memref<16x16xf32>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<16x16xf32>
    call @fill_zeros_f32_16_16_vector(%alloc) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<16x64xf32>, memref<64x16xf32>) outs(%alloc : memref<16x16xf32>)
    memref.copy %alloc, %arg2 {to = "local_C"} : memref<16x16xf32> to memref<16x16xf32>
    return
  }
  func.func @gemm_3_1(%arg0: memref<16x64xf32>, %arg1: memref<64x16xf32>, %arg2: memref<16x16xf32>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<16x16xf32>
    call @fill_zeros_f32_16_16_vector(%alloc) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<16x64xf32>, memref<64x16xf32>) outs(%alloc : memref<16x16xf32>)
    memref.copy %alloc, %arg2 {to = "local_C"} : memref<16x16xf32> to memref<16x16xf32>
    return
  }
  func.func @gemm_3_2(%arg0: memref<16x64xf32>, %arg1: memref<64x16xf32>, %arg2: memref<16x16xf32>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<16x16xf32>
    call @fill_zeros_f32_16_16_vector(%alloc) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<16x64xf32>, memref<64x16xf32>) outs(%alloc : memref<16x16xf32>)
    memref.copy %alloc, %arg2 {to = "local_C"} : memref<16x16xf32> to memref<16x16xf32>
    return
  }
  func.func @gemm_3_3(%arg0: memref<16x64xf32>, %arg1: memref<64x16xf32>, %arg2: memref<16x16xf32>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<16x16xf32>
    call @fill_zeros_f32_16_16_vector(%alloc) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<16x64xf32>, memref<64x16xf32>) outs(%alloc : memref<16x16xf32>)
    memref.copy %alloc, %arg2 {to = "local_C"} : memref<16x16xf32> to memref<16x16xf32>
    return
  }
  func.func @linear_matmul_kernel(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) attributes {dataflow, itypes = "___"} {
    return
  }
}
