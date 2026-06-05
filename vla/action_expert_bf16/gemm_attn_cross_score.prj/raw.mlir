module {
  func.func @gemm_0_0_0(%arg0: memref<32x64xbf16>, %arg1: memref<64x128xbf16>, %arg2: memref<32x128xbf16>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "___", tag = "gemm_(None, (), None, ())"} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "C_in"} : memref<32x128xbf16>
    %alloc_0 = memref.alloc() : memref<32x128xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc_0 : memref<32x128xbf16>)
    memref.copy %alloc_0, %alloc {to = "C_in"} : memref<32x128xbf16> to memref<32x128xbf16>
    %alloc_1 = memref.alloc() : memref<32x128xbf16>
    linalg.fill {op_name = "matmul_init_zero_0"} ins(%cst : bf16) outs(%alloc_1 : memref<32x128xbf16>)
    linalg.matmul {op_name = "matmul_1"} ins(%arg0, %arg1 : memref<32x64xbf16>, memref<64x128xbf16>) outs(%alloc_1 : memref<32x128xbf16>)
    %alloc_2 = memref.alloc() {name = "C_out"} : memref<32x128xbf16>
    linalg.add {op_name = "add_2"} ins(%alloc_1, %alloc : memref<32x128xbf16>, memref<32x128xbf16>) outs(%alloc_2 : memref<32x128xbf16>)
    memref.copy %alloc_2, %arg2 {to = "local_C"} : memref<32x128xbf16> to memref<32x128xbf16>
    return
  }
  func.func @top(%arg0: memref<32x64xbf16>, %arg1: memref<64x128xbf16>, %arg2: memref<32x128xbf16>) attributes {dataflow, itypes = "___"} {
    return
  }
}
