module {
  func.func private @layer_norm_bf16(memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>)
  func.func @norm_no_bias_0(%arg0: memref<4x768xbf16>, %arg1: memref<768xbf16>, %arg2: !allo.stream<memref<4x768xbf16>, 1>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "__o", tag = "norm_no_bias_()"} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "tmp"} : memref<4x768xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<4x768xbf16>)
    call @layer_norm_bf16(%arg0, %arg1, %alloc) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
    allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<4x768xbf16>, 1> contains memref<4x768xbf16>
    return
  }
  func.func @norm_add_bias_0(%arg0: memref<768xbf16>, %arg1: memref<4x768xbf16>, %arg2: !allo.stream<memref<4x768xbf16>, 1>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "__i", tag = "norm_add_bias_()"} {
    %0 = allo.stream_get(%arg2, []) {name = "data"} : !allo.stream<memref<4x768xbf16>, 1> -> memref<4x768xbf16>
    %alloc = memref.alloc() : memref<4x768xbf16>
    linalg.broadcast ins(%arg0 : memref<768xbf16>) outs(%alloc : memref<4x768xbf16>) dimensions = [0] 
    %alloc_0 = memref.alloc() : memref<4x768xbf16>
    linalg.add {op_name = "add_0"} ins(%0, %alloc : memref<4x768xbf16>, memref<4x768xbf16>) outs(%alloc_0 : memref<4x768xbf16>)
    memref.copy %alloc_0, %arg1 {to = "local_output_x"} : memref<4x768xbf16> to memref<4x768xbf16>
    return
  }
  func.func @layer_norm_kernel(%arg0: memref<32x768xbf16>, %arg1: memref<768xbf16>, %arg2: memref<768xbf16>, %arg3: memref<32x768xbf16>) attributes {dataflow, itypes = "____"} {
    %0 = allo.stream_construct() {name = "pipe_0"} : !allo.stream<memref<4x768xbf16>, 1>
    %1 = allo.stream_construct() {name = "pipe_1"} : !allo.stream<memref<4x768xbf16>, 1>
    %2 = allo.stream_construct() {name = "pipe_2"} : !allo.stream<memref<4x768xbf16>, 1>
    %3 = allo.stream_construct() {name = "pipe_3"} : !allo.stream<memref<4x768xbf16>, 1>
    %4 = allo.stream_construct() {name = "pipe_4"} : !allo.stream<memref<4x768xbf16>, 1>
    %5 = allo.stream_construct() {name = "pipe_5"} : !allo.stream<memref<4x768xbf16>, 1>
    %6 = allo.stream_construct() {name = "pipe_6"} : !allo.stream<memref<4x768xbf16>, 1>
    %7 = allo.stream_construct() {name = "pipe_7"} : !allo.stream<memref<4x768xbf16>, 1>
    return
  }
}
