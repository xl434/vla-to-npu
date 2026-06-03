module {
  func.func @send_q_0_0(%arg0: memref<32x64xbf16>, %arg1: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "_o", tag = "send_q_()"} {
    affine.for %arg2 = 0 to 4 {
      allo.stream_put(%arg1, [], %arg0) : !allo.stream<memref<32x64xbf16>, 2> contains memref<32x64xbf16>
    } {loop_name = "i", op_name = "S_i_0"}
    return
  }
  func.func @cal_attn_score_0_0(%arg0: memref<64x32xbf16>, %arg1: !allo.stream<memref<32x64xbf16>, 2>, %arg2: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "_io", tag = "cal_attn_score_()"} {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = allo.stream_get(%arg1, []) : !allo.stream<memref<32x64xbf16>, 2> -> memref<32x64xbf16>
    %alloc = memref.alloc() {name = "score"} : memref<32x32xbf16>
    linalg.fill {op_name = "matmul_init_zero_0"} ins(%cst : bf16) outs(%alloc : memref<32x32xbf16>)
    linalg.matmul {op_name = "matmul_1"} ins(%0, %arg0 : memref<32x64xbf16>, memref<64x32xbf16>) outs(%alloc : memref<32x32xbf16>)
    allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<32x32xbf16>, 2> contains memref<32x32xbf16>
    return
  }
  func.func private @init_softmax(memref<32xbf16>, memref<32xbf16>)
  func.func private @online_softmax(memref<32x32xbf16>, memref<32xbf16>, memref<32xbf16>, memref<32x32xbf16>, memref<32xbf16>, memref<32xbf16>, memref<32xbf16>)
  func.func @cal_softmax_0_0(%arg0: !allo.stream<memref<32x32xbf16>, 2>, %arg1: !allo.stream<memref<32xbf16>, 2>, %arg2: !allo.stream<memref<32x32xbf16>, 2>, %arg3: !allo.stream<memref<32xbf16>, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "iooo", tag = "cal_softmax_()"} {
    %alloc = memref.alloc() {name = "max_logit"} : memref<32xbf16>
    %alloc_0 = memref.alloc() {name = "sum_exp"} : memref<32xbf16>
    call @init_softmax(%alloc, %alloc_0) : (memref<32xbf16>, memref<32xbf16>) -> ()
    affine.for %arg4 = 0 to 4 {
      %alloc_1 = memref.alloc() {name = "attn_weight"} : memref<32x32xbf16>
      %alloc_2 = memref.alloc() {name = "scale_exp"} : memref<32xbf16>
      %0 = allo.stream_get(%arg0, []) : !allo.stream<memref<32x32xbf16>, 2> -> memref<32x32xbf16>
      func.call @online_softmax(%0, %alloc, %alloc_0, %alloc_1, %alloc_2, %alloc, %alloc_0) : (memref<32x32xbf16>, memref<32xbf16>, memref<32xbf16>, memref<32x32xbf16>, memref<32xbf16>, memref<32xbf16>, memref<32xbf16>) -> ()
      allo.stream_put(%arg1, [], %alloc_2) : !allo.stream<memref<32xbf16>, 2> contains memref<32xbf16>
      allo.stream_put(%arg2, [], %alloc_1) : !allo.stream<memref<32x32xbf16>, 2> contains memref<32x32xbf16>
    } {loop_name = "i", op_name = "S_i_0"}
    allo.stream_put(%arg3, [], %alloc_0) : !allo.stream<memref<32xbf16>, 2> contains memref<32xbf16>
    return
  }
  func.func @attn_0_0(%arg0: memref<32x64xbf16>, %arg1: !allo.stream<memref<32x32xbf16>, 2>, %arg2: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "_io", tag = "attn_()"} {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = allo.stream_get(%arg1, []) : !allo.stream<memref<32x32xbf16>, 2> -> memref<32x32xbf16>
    %alloc = memref.alloc() : memref<32x64xbf16>
    linalg.fill {op_name = "matmul_init_zero_0"} ins(%cst : bf16) outs(%alloc : memref<32x64xbf16>)
    linalg.matmul {op_name = "matmul_1"} ins(%0, %arg0 : memref<32x32xbf16>, memref<32x64xbf16>) outs(%alloc : memref<32x64xbf16>)
    allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<32x64xbf16>, 2> contains memref<32x64xbf16>
    return
  }
  func.func private @rescale_attn_output(memref<32x64xbf16>, memref<32xbf16>, memref<32x64xbf16>)
  func.func private @scale_attn_output(memref<32x64xbf16>, memref<32xbf16>, memref<32x64xbf16>)
  func.func @acc_0_0(%arg0: memref<32x64xbf16>, %arg1: !allo.stream<memref<32xbf16>, 2>, %arg2: !allo.stream<memref<32x64xbf16>, 2>, %arg3: !allo.stream<memref<32xbf16>, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "_iii", tag = "acc_()"} {
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {name = "attn_output"} : memref<32x64xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<32x64xbf16>)
    affine.for %arg4 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<memref<32xbf16>, 2> -> memref<32xbf16>
      func.call @rescale_attn_output(%alloc, %1, %alloc) : (memref<32x64xbf16>, memref<32xbf16>, memref<32x64xbf16>) -> ()
      %2 = allo.stream_get(%arg2, []) : !allo.stream<memref<32x64xbf16>, 2> -> memref<32x64xbf16>
      %alloc_0 = memref.alloc() : memref<32x64xbf16>
      linalg.add {op_name = "add_0"} ins(%alloc, %2 : memref<32x64xbf16>, memref<32x64xbf16>) outs(%alloc_0 : memref<32x64xbf16>)
      memref.copy %alloc_0, %alloc {to = "attn_output"} : memref<32x64xbf16> to memref<32x64xbf16>
    } {loop_name = "i", op_name = "S_i_0"}
    %0 = allo.stream_get(%arg3, []) : !allo.stream<memref<32xbf16>, 2> -> memref<32xbf16>
    call @scale_attn_output(%alloc, %0, %arg0) : (memref<32x64xbf16>, memref<32xbf16>, memref<32x64xbf16>) -> ()
    return
  }
  func.func @top(%arg0: memref<32x64xbf16>, %arg1: memref<64x128xbf16>, %arg2: memref<128x64xbf16>, %arg3: memref<32x64xbf16>) attributes {dataflow, itypes = "____"} {
    %0 = allo.stream_construct() {name = "q_pipe_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %1 = allo.stream_construct() {name = "q_pipe_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %2 = allo.stream_construct() {name = "q_pipe_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %3 = allo.stream_construct() {name = "q_pipe_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %4 = allo.stream_construct() {name = "score_pipe_0_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %5 = allo.stream_construct() {name = "score_pipe_0_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %6 = allo.stream_construct() {name = "score_pipe_0_2"} : !allo.stream<memref<32x32xbf16>, 2>
    %7 = allo.stream_construct() {name = "score_pipe_0_3"} : !allo.stream<memref<32x32xbf16>, 2>
    %8 = allo.stream_construct() {name = "weight_pipe_0_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %9 = allo.stream_construct() {name = "weight_pipe_0_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %10 = allo.stream_construct() {name = "weight_pipe_0_2"} : !allo.stream<memref<32x32xbf16>, 2>
    %11 = allo.stream_construct() {name = "weight_pipe_0_3"} : !allo.stream<memref<32x32xbf16>, 2>
    %12 = allo.stream_construct() {name = "o_pipe_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %13 = allo.stream_construct() {name = "o_pipe_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %14 = allo.stream_construct() {name = "o_pipe_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %15 = allo.stream_construct() {name = "o_pipe_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %16 = allo.stream_construct() {name = "exp_sum_pipe_0"} : !allo.stream<memref<32xbf16>, 2>
    %17 = allo.stream_construct() {name = "exp_scale_pipe_0"} : !allo.stream<memref<32xbf16>, 2>
    return
  }
}
