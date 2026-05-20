module {
  func.func @gemm_0_0_0(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "___o", tag = "gemm_(None, (), (), None)"} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    %alloc_0 = memref.alloc() : memref<32x32xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc_0 : memref<32x32xbf16>)
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_1 = memref.alloc() : memref<32x32xbf16>
    linalg.fill {op_name = "matmul_init_zero_0"} ins(%cst : bf16) outs(%alloc_1 : memref<32x32xbf16>)
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<32x32xbf16>, memref<32x32xbf16>) outs(%alloc_1 : memref<32x32xbf16>)
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    linalg.add {op_name = "add_2"} ins(%alloc_1, %alloc : memref<32x32xbf16>, memref<32x32xbf16>) outs(%alloc_2 : memref<32x32xbf16>)
    allo.stream_put(%arg3, [], %alloc_2) : !allo.stream<memref<32x32xbf16>, 2> contains memref<32x32xbf16>
    return
  }
  func.func @gemm_1_0_0(%arg0: memref<32x32xbf16>, %arg1: memref<32x32xbf16>, %arg2: memref<32x32xbf16>, %arg3: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "___i", tag = "gemm_((), None, None, ())"} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x32xbf16>
    %0 = allo.stream_get(%arg3, []) : !allo.stream<memref<32x32xbf16>, 2> -> memref<32x32xbf16>
    memref.copy %0, %alloc {to = "C_in"} : memref<32x32xbf16> to memref<32x32xbf16>
    %alloc_0 = memref.alloc() : memref<32x32xbf16>
    linalg.fill {op_name = "matmul_init_zero_0"} ins(%cst : bf16) outs(%alloc_0 : memref<32x32xbf16>)
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<32x32xbf16>, memref<32x32xbf16>) outs(%alloc_0 : memref<32x32xbf16>)
    %alloc_1 = memref.alloc() {name = "C_out"} : memref<32x32xbf16>
    linalg.add {op_name = "add_2"} ins(%alloc_0, %alloc : memref<32x32xbf16>, memref<32x32xbf16>) outs(%alloc_1 : memref<32x32xbf16>)
    memref.copy %alloc_1, %arg2 {to = "local_C"} : memref<32x32xbf16> to memref<32x32xbf16>
    return
  }
  func.func @top(%arg0: memref<128x64xbf16>, %arg1: memref<64x128xbf16>, %arg2: memref<128x128xbf16>) attributes {dataflow, itypes = "___"} {
    %0 = allo.stream_construct() {name = "pipe_0_0_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %1 = allo.stream_construct() {name = "pipe_0_0_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %2 = allo.stream_construct() {name = "pipe_0_0_2"} : !allo.stream<memref<32x32xbf16>, 2>
    %3 = allo.stream_construct() {name = "pipe_0_0_3"} : !allo.stream<memref<32x32xbf16>, 2>
    %4 = allo.stream_construct() {name = "pipe_0_1_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %5 = allo.stream_construct() {name = "pipe_0_1_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %6 = allo.stream_construct() {name = "pipe_0_1_2"} : !allo.stream<memref<32x32xbf16>, 2>
    %7 = allo.stream_construct() {name = "pipe_0_1_3"} : !allo.stream<memref<32x32xbf16>, 2>
    %8 = allo.stream_construct() {name = "pipe_0_2_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %9 = allo.stream_construct() {name = "pipe_0_2_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %10 = allo.stream_construct() {name = "pipe_0_2_2"} : !allo.stream<memref<32x32xbf16>, 2>
    %11 = allo.stream_construct() {name = "pipe_0_2_3"} : !allo.stream<memref<32x32xbf16>, 2>
    %12 = allo.stream_construct() {name = "pipe_0_3_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %13 = allo.stream_construct() {name = "pipe_0_3_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %14 = allo.stream_construct() {name = "pipe_0_3_2"} : !allo.stream<memref<32x32xbf16>, 2>
    %15 = allo.stream_construct() {name = "pipe_0_3_3"} : !allo.stream<memref<32x32xbf16>, 2>
    return
  }
}
