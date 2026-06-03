module {
  func.func private @fill_zeros_bf16_32_32_vector(memref<32x32xbf16>)
  func.func private @matmul_scalar_bf16_bf16_32x64x32(memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>)
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
  func.func private @fill_zeros_bf16_32_64_vector(memref<32x64xbf16>)
  func.func private @matmul_scalar_bf16_bf16_32x32x64(memref<32x32xbf16>, memref<32x64xbf16>, memref<32x64xbf16>)
  func.func private @rescale_attn_output(memref<32x64xbf16>, memref<32xbf16>, memref<32x64xbf16>)
  func.func private @scale_attn_output(memref<32x64xbf16>, memref<32xbf16>, memref<32x64xbf16>)
  func.func private @add_bf16_vector(memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>)
  func.func @acc_0_0(%arg0: memref<32x64xbf16>, %arg1: !allo.stream<memref<32xbf16>, 2>, %arg2: !allo.stream<memref<32x64xbf16>, 2>, %arg3: !allo.stream<memref<32xbf16>, 2>) attributes {df.kernel, itypes = "____", otypes = "", stypes = "_iii", tag = "acc_()"} {
    %alloc = memref.alloc() {name = "attn_output"} : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    affine.for %arg4 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<memref<32xbf16>, 2> -> memref<32xbf16>
      func.call @rescale_attn_output(%alloc, %1, %alloc) : (memref<32x64xbf16>, memref<32xbf16>, memref<32x64xbf16>) -> ()
      %2 = allo.stream_get(%arg2, []) : !allo.stream<memref<32x64xbf16>, 2> -> memref<32x64xbf16>
      func.call @add_bf16_vector(%alloc, %2, %alloc) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    } {loop_name = "i", op_name = "S_i_0"}
    %0 = allo.stream_get(%arg3, []) : !allo.stream<memref<32xbf16>, 2> -> memref<32xbf16>
    call @scale_attn_output(%alloc, %0, %arg0) : (memref<32x64xbf16>, memref<32xbf16>, memref<32x64xbf16>) -> ()
    return
  }
  func.func @cal_softmax_1_0(%arg0: !allo.stream<memref<32x32xbf16>, 2>, %arg1: !allo.stream<memref<32xbf16>, 2>, %arg2: !allo.stream<memref<32x32xbf16>, 2>, %arg3: !allo.stream<memref<32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0], output_depth = []} {
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
  func.func @cal_softmax_2_0(%arg0: !allo.stream<memref<32x32xbf16>, 2>, %arg1: !allo.stream<memref<32xbf16>, 2>, %arg2: !allo.stream<memref<32x32xbf16>, 2>, %arg3: !allo.stream<memref<32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0], output_depth = []} {
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
  func.func @cal_softmax_3_0(%arg0: !allo.stream<memref<32x32xbf16>, 2>, %arg1: !allo.stream<memref<32xbf16>, 2>, %arg2: !allo.stream<memref<32x32xbf16>, 2>, %arg3: !allo.stream<memref<32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0], output_depth = []} {
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
  func.func @acc_1_0(%arg0: memref<32x64xbf16>, %arg1: !allo.stream<memref<32xbf16>, 2>, %arg2: !allo.stream<memref<32x64xbf16>, 2>, %arg3: !allo.stream<memref<32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0], output_depth = []} {
    %alloc = memref.alloc() {name = "attn_output"} : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    affine.for %arg4 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<memref<32xbf16>, 2> -> memref<32xbf16>
      func.call @rescale_attn_output(%alloc, %1, %alloc) : (memref<32x64xbf16>, memref<32xbf16>, memref<32x64xbf16>) -> ()
      %2 = allo.stream_get(%arg2, []) : !allo.stream<memref<32x64xbf16>, 2> -> memref<32x64xbf16>
      func.call @add_bf16_vector(%alloc, %2, %alloc) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    } {loop_name = "i", op_name = "S_i_0"}
    %0 = allo.stream_get(%arg3, []) : !allo.stream<memref<32xbf16>, 2> -> memref<32xbf16>
    call @scale_attn_output(%alloc, %0, %arg0) : (memref<32x64xbf16>, memref<32xbf16>, memref<32x64xbf16>) -> ()
    return
  }
  func.func @acc_2_0(%arg0: memref<32x64xbf16>, %arg1: !allo.stream<memref<32xbf16>, 2>, %arg2: !allo.stream<memref<32x64xbf16>, 2>, %arg3: !allo.stream<memref<32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0], output_depth = []} {
    %alloc = memref.alloc() {name = "attn_output"} : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    affine.for %arg4 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<memref<32xbf16>, 2> -> memref<32xbf16>
      func.call @rescale_attn_output(%alloc, %1, %alloc) : (memref<32x64xbf16>, memref<32xbf16>, memref<32x64xbf16>) -> ()
      %2 = allo.stream_get(%arg2, []) : !allo.stream<memref<32x64xbf16>, 2> -> memref<32x64xbf16>
      func.call @add_bf16_vector(%alloc, %2, %alloc) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    } {loop_name = "i", op_name = "S_i_0"}
    %0 = allo.stream_get(%arg3, []) : !allo.stream<memref<32xbf16>, 2> -> memref<32xbf16>
    call @scale_attn_output(%alloc, %0, %arg0) : (memref<32x64xbf16>, memref<32xbf16>, memref<32x64xbf16>) -> ()
    return
  }
  func.func @acc_3_0(%arg0: memref<32x64xbf16>, %arg1: !allo.stream<memref<32xbf16>, 2>, %arg2: !allo.stream<memref<32x64xbf16>, 2>, %arg3: !allo.stream<memref<32xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0, 0], output_depth = []} {
    %alloc = memref.alloc() {name = "attn_output"} : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    affine.for %arg4 = 0 to 4 {
      %1 = allo.stream_get(%arg1, []) : !allo.stream<memref<32xbf16>, 2> -> memref<32xbf16>
      func.call @rescale_attn_output(%alloc, %1, %alloc) : (memref<32x64xbf16>, memref<32xbf16>, memref<32x64xbf16>) -> ()
      %2 = allo.stream_get(%arg2, []) : !allo.stream<memref<32x64xbf16>, 2> -> memref<32x64xbf16>
      func.call @add_bf16_vector(%alloc, %2, %alloc) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    } {loop_name = "i", op_name = "S_i_0"}
    %0 = allo.stream_get(%arg3, []) : !allo.stream<memref<32xbf16>, 2> -> memref<32xbf16>
    call @scale_attn_output(%alloc, %0, %arg0) : (memref<32x64xbf16>, memref<32xbf16>, memref<32x64xbf16>) -> ()
    return
  }
  func.func private @matmul_bf16_bf16_32x32x64(memref<32x32xbf16>, memref<32x64xbf16>, memref<32x64xbf16>)
  func.func @attn_0_0x4(%arg0: memref<32x64xbf16>, %arg1: !allo.stream<memref<32x32xbf16>, 2>, %arg2: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %0 = allo.stream_get(%arg1, []) : !allo.stream<memref<32x32xbf16>, 2> -> memref<32x32xbf16>
    %alloc = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16_32x32x64(%0, %arg0, %alloc) : (memref<32x32xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<32x64xbf16>, 2> contains memref<32x64xbf16>
    return
  }
  func.func private @matmul_bf16_bf16_32x64x32(memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>)
  func.func @"send_q_0_0-cal_attn_score_0_0x4"(%arg0: memref<32x64xbf16>, %arg1: !allo.stream<memref<32x64xbf16>, 2>, %arg2: memref<64x32xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [1, 1, 0, 0, 0], output_depth = []} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %arg5 = %c0 to %c4 step %c1 {
      %alloc = memref.alloc() {name = "score"} : memref<32x32xbf16>
      func.call @fill_zeros_bf16_32_32_vector(%alloc) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      func.call @matmul_bf16_bf16_32x64x32(%arg0, %arg2, %alloc) : (memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>) -> ()
      allo.stream_put(%arg4, [], %alloc) : !allo.stream<memref<32x32xbf16>, 2> contains memref<32x32xbf16>
    } {task_nest}
    return
  }
  func.func @attn_1_0x4(%arg0: memref<32x64xbf16>, %arg1: !allo.stream<memref<32x32xbf16>, 2>, %arg2: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %0 = allo.stream_get(%arg1, []) : !allo.stream<memref<32x32xbf16>, 2> -> memref<32x32xbf16>
    %alloc = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16_32x32x64(%0, %arg0, %alloc) : (memref<32x32xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<32x64xbf16>, 2> contains memref<32x64xbf16>
    return
  }
  func.func @"send_q_1_0-cal_attn_score_1_0x4"(%arg0: memref<32x64xbf16>, %arg1: !allo.stream<memref<32x64xbf16>, 2>, %arg2: memref<64x32xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [1, 1, 0, 0, 0], output_depth = []} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %arg5 = %c0 to %c4 step %c1 {
      %alloc = memref.alloc() {name = "score"} : memref<32x32xbf16>
      func.call @fill_zeros_bf16_32_32_vector(%alloc) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      func.call @matmul_bf16_bf16_32x64x32(%arg0, %arg2, %alloc) : (memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>) -> ()
      allo.stream_put(%arg4, [], %alloc) : !allo.stream<memref<32x32xbf16>, 2> contains memref<32x32xbf16>
    } {task_nest}
    return
  }
  func.func @attn_2_0x4(%arg0: memref<32x64xbf16>, %arg1: !allo.stream<memref<32x32xbf16>, 2>, %arg2: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %0 = allo.stream_get(%arg1, []) : !allo.stream<memref<32x32xbf16>, 2> -> memref<32x32xbf16>
    %alloc = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16_32x32x64(%0, %arg0, %alloc) : (memref<32x32xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<32x64xbf16>, 2> contains memref<32x64xbf16>
    return
  }
  func.func @"send_q_2_0-cal_attn_score_2_0x4"(%arg0: memref<32x64xbf16>, %arg1: !allo.stream<memref<32x64xbf16>, 2>, %arg2: memref<64x32xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [1, 1, 0, 0, 0], output_depth = []} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %arg5 = %c0 to %c4 step %c1 {
      %alloc = memref.alloc() {name = "score"} : memref<32x32xbf16>
      func.call @fill_zeros_bf16_32_32_vector(%alloc) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      func.call @matmul_bf16_bf16_32x64x32(%arg0, %arg2, %alloc) : (memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>) -> ()
      allo.stream_put(%arg4, [], %alloc) : !allo.stream<memref<32x32xbf16>, 2> contains memref<32x32xbf16>
    } {task_nest}
    return
  }
  func.func @attn_3_0x4(%arg0: memref<32x64xbf16>, %arg1: !allo.stream<memref<32x32xbf16>, 2>, %arg2: !allo.stream<memref<32x64xbf16>, 2>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %0 = allo.stream_get(%arg1, []) : !allo.stream<memref<32x32xbf16>, 2> -> memref<32x32xbf16>
    %alloc = memref.alloc() : memref<32x64xbf16>
    call @fill_zeros_bf16_32_64_vector(%alloc) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
    call @matmul_bf16_bf16_32x32x64(%0, %arg0, %alloc) : (memref<32x32xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<32x64xbf16>, 2> contains memref<32x64xbf16>
    return
  }
  func.func @"send_q_3_0-cal_attn_score_3_0x4"(%arg0: memref<32x64xbf16>, %arg1: !allo.stream<memref<32x64xbf16>, 2>, %arg2: memref<64x32xbf16>, %arg3: !allo.stream<memref<32x64xbf16>, 2>, %arg4: !allo.stream<memref<32x32xbf16>, 2>) attributes {df.kernel, input_depth = [1, 1, 0, 0, 0], output_depth = []} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %arg5 = %c0 to %c4 step %c1 {
      %alloc = memref.alloc() {name = "score"} : memref<32x32xbf16>
      func.call @fill_zeros_bf16_32_32_vector(%alloc) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      func.call @matmul_bf16_bf16_32x64x32(%arg0, %arg2, %alloc) : (memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>) -> ()
      allo.stream_put(%arg4, [], %alloc) : !allo.stream<memref<32x32xbf16>, 2> contains memref<32x32xbf16>
    } {task_nest}
    return
  }
  func.func @top(%arg0: memref<128x64xbf16>, %arg1: memref<64x128xbf16>, %arg2: memref<128x64xbf16>, %arg3: memref<128x64xbf16>) attributes {dataflow, itypes = "____"} {
    %0 = allo.stream_construct() {name = "q_pipe_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %1 = allo.stream_construct() {name = "q_pipe_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %2 = allo.stream_construct() {name = "q_pipe_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %3 = allo.stream_construct() {name = "q_pipe_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %4 = allo.stream_construct() {name = "q_pipe_1_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %5 = allo.stream_construct() {name = "q_pipe_1_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %6 = allo.stream_construct() {name = "q_pipe_1_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %7 = allo.stream_construct() {name = "q_pipe_1_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %8 = allo.stream_construct() {name = "q_pipe_2_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %9 = allo.stream_construct() {name = "q_pipe_2_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %10 = allo.stream_construct() {name = "q_pipe_2_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %11 = allo.stream_construct() {name = "q_pipe_2_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %12 = allo.stream_construct() {name = "q_pipe_3_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %13 = allo.stream_construct() {name = "q_pipe_3_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %14 = allo.stream_construct() {name = "q_pipe_3_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %15 = allo.stream_construct() {name = "q_pipe_3_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %16 = allo.stream_construct() {name = "score_pipe_0_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %17 = allo.stream_construct() {name = "score_pipe_0_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %18 = allo.stream_construct() {name = "score_pipe_0_2"} : !allo.stream<memref<32x32xbf16>, 2>
    %19 = allo.stream_construct() {name = "score_pipe_0_3"} : !allo.stream<memref<32x32xbf16>, 2>
    %20 = allo.stream_construct() {name = "score_pipe_1_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %21 = allo.stream_construct() {name = "score_pipe_1_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %22 = allo.stream_construct() {name = "score_pipe_1_2"} : !allo.stream<memref<32x32xbf16>, 2>
    %23 = allo.stream_construct() {name = "score_pipe_1_3"} : !allo.stream<memref<32x32xbf16>, 2>
    %24 = allo.stream_construct() {name = "score_pipe_2_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %25 = allo.stream_construct() {name = "score_pipe_2_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %26 = allo.stream_construct() {name = "score_pipe_2_2"} : !allo.stream<memref<32x32xbf16>, 2>
    %27 = allo.stream_construct() {name = "score_pipe_2_3"} : !allo.stream<memref<32x32xbf16>, 2>
    %28 = allo.stream_construct() {name = "score_pipe_3_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %29 = allo.stream_construct() {name = "score_pipe_3_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %30 = allo.stream_construct() {name = "score_pipe_3_2"} : !allo.stream<memref<32x32xbf16>, 2>
    %31 = allo.stream_construct() {name = "score_pipe_3_3"} : !allo.stream<memref<32x32xbf16>, 2>
    %32 = allo.stream_construct() {name = "weight_pipe_0_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %33 = allo.stream_construct() {name = "weight_pipe_0_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %34 = allo.stream_construct() {name = "weight_pipe_0_2"} : !allo.stream<memref<32x32xbf16>, 2>
    %35 = allo.stream_construct() {name = "weight_pipe_0_3"} : !allo.stream<memref<32x32xbf16>, 2>
    %36 = allo.stream_construct() {name = "weight_pipe_1_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %37 = allo.stream_construct() {name = "weight_pipe_1_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %38 = allo.stream_construct() {name = "weight_pipe_1_2"} : !allo.stream<memref<32x32xbf16>, 2>
    %39 = allo.stream_construct() {name = "weight_pipe_1_3"} : !allo.stream<memref<32x32xbf16>, 2>
    %40 = allo.stream_construct() {name = "weight_pipe_2_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %41 = allo.stream_construct() {name = "weight_pipe_2_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %42 = allo.stream_construct() {name = "weight_pipe_2_2"} : !allo.stream<memref<32x32xbf16>, 2>
    %43 = allo.stream_construct() {name = "weight_pipe_2_3"} : !allo.stream<memref<32x32xbf16>, 2>
    %44 = allo.stream_construct() {name = "weight_pipe_3_0"} : !allo.stream<memref<32x32xbf16>, 2>
    %45 = allo.stream_construct() {name = "weight_pipe_3_1"} : !allo.stream<memref<32x32xbf16>, 2>
    %46 = allo.stream_construct() {name = "weight_pipe_3_2"} : !allo.stream<memref<32x32xbf16>, 2>
    %47 = allo.stream_construct() {name = "weight_pipe_3_3"} : !allo.stream<memref<32x32xbf16>, 2>
    %48 = allo.stream_construct() {name = "o_pipe_0_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %49 = allo.stream_construct() {name = "o_pipe_0_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %50 = allo.stream_construct() {name = "o_pipe_0_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %51 = allo.stream_construct() {name = "o_pipe_0_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %52 = allo.stream_construct() {name = "o_pipe_1_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %53 = allo.stream_construct() {name = "o_pipe_1_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %54 = allo.stream_construct() {name = "o_pipe_1_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %55 = allo.stream_construct() {name = "o_pipe_1_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %56 = allo.stream_construct() {name = "o_pipe_2_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %57 = allo.stream_construct() {name = "o_pipe_2_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %58 = allo.stream_construct() {name = "o_pipe_2_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %59 = allo.stream_construct() {name = "o_pipe_2_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %60 = allo.stream_construct() {name = "o_pipe_3_0"} : !allo.stream<memref<32x64xbf16>, 2>
    %61 = allo.stream_construct() {name = "o_pipe_3_1"} : !allo.stream<memref<32x64xbf16>, 2>
    %62 = allo.stream_construct() {name = "o_pipe_3_2"} : !allo.stream<memref<32x64xbf16>, 2>
    %63 = allo.stream_construct() {name = "o_pipe_3_3"} : !allo.stream<memref<32x64xbf16>, 2>
    %64 = allo.stream_construct() {name = "exp_sum_pipe_0"} : !allo.stream<memref<32xbf16>, 2>
    %65 = allo.stream_construct() {name = "exp_sum_pipe_1"} : !allo.stream<memref<32xbf16>, 2>
    %66 = allo.stream_construct() {name = "exp_sum_pipe_2"} : !allo.stream<memref<32xbf16>, 2>
    %67 = allo.stream_construct() {name = "exp_sum_pipe_3"} : !allo.stream<memref<32xbf16>, 2>
    %68 = allo.stream_construct() {name = "exp_scale_pipe_0"} : !allo.stream<memref<32xbf16>, 2>
    %69 = allo.stream_construct() {name = "exp_scale_pipe_1"} : !allo.stream<memref<32xbf16>, 2>
    %70 = allo.stream_construct() {name = "exp_scale_pipe_2"} : !allo.stream<memref<32xbf16>, 2>
    %71 = allo.stream_construct() {name = "exp_scale_pipe_3"} : !allo.stream<memref<32xbf16>, 2>
    return
  }
}
