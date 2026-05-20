module {
  func.func @gemm_0_0_0(%arg0: memref<32x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<32x64xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "___o", tag = "gemm_(None, (), (), None)"} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    %alloc_0 = memref.alloc() : memref<32x64xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc_0 : memref<32x64xbf16>)
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_1 = memref.alloc() : memref<32x64xbf16>
    linalg.fill {op_name = "matmul_init_zero_0"} ins(%cst : bf16) outs(%alloc_1 : memref<32x64xbf16>)
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<32x64xbf16>, memref<64x64xbf16>) outs(%alloc_1 : memref<32x64xbf16>)
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    linalg.add {op_name = "add_2"} ins(%alloc_1, %alloc : memref<32x64xbf16>, memref<32x64xbf16>) outs(%alloc_2 : memref<32x64xbf16>)
    allo.stream_put(%arg3, [], %alloc_2) : !allo.stream<memref<32x64xbf16>, 2> contains memref<32x64xbf16>
    return
  }
  func.func @gemm_1_0_0(%arg0: memref<32x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<32x64xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, itypes = "_____", otypes = "", stypes = "___io", tag = "gemm_((), None, (), None)"} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    %0 = allo.stream_get(%arg3, []) : !allo.stream<memref<32x64xbf16>, 2> -> memref<32x64xbf16>
    memref.copy %0, %alloc {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_0 = memref.alloc() : memref<32x64xbf16>
    linalg.fill {op_name = "matmul_init_zero_0"} ins(%cst : bf16) outs(%alloc_0 : memref<32x64xbf16>)
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<32x64xbf16>, memref<64x64xbf16>) outs(%alloc_0 : memref<32x64xbf16>)
    %alloc_1 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    linalg.add {op_name = "add_2"} ins(%alloc_0, %alloc : memref<32x64xbf16>, memref<32x64xbf16>) outs(%alloc_1 : memref<32x64xbf16>)
    allo.stream_put(%arg4, [], %alloc_1) : !allo.stream<memref<32x64xbf16>, 2> contains memref<32x64xbf16>
    return
  }
  func.func @gemm_11_0_0(%arg0: memref<32x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<32x64xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "___i", tag = "gemm_((), None, None, ())"} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x64xbf16>
    %0 = allo.stream_get(%arg3, []) : !allo.stream<memref<32x64xbf16>, 2> -> memref<32x64xbf16>
    memref.copy %0, %alloc {to = "C_in"} : memref<32x64xbf16> to memref<32x64xbf16>
    %alloc_0 = memref.alloc() : memref<32x64xbf16>
    linalg.fill {op_name = "matmul_init_zero_0"} ins(%cst : bf16) outs(%alloc_0 : memref<32x64xbf16>)
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<32x64xbf16>, memref<64x64xbf16>) outs(%alloc_0 : memref<32x64xbf16>)
    %alloc_1 = memref.alloc() {name = "C_out"} : memref<32x64xbf16>
    linalg.add {op_name = "add_2"} ins(%alloc_0, %alloc : memref<32x64xbf16>, memref<32x64xbf16>) outs(%alloc_1 : memref<32x64xbf16>)
    memref.copy %alloc_1, %arg2 {to = "local_C"} : memref<32x64xbf16> to memref<32x64xbf16>
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
