module {
  func.func private @layer_norm(memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>)
  func.func private @fill_zeros_bf16_4_768_vector(memref<4x768xbf16>)
  func.func @norm_no_bias_0(%arg0: memref<4x768xbf16>, %arg1: memref<768xbf16>, %arg2: !allo.stream<memref<4x768xbf16>, 1>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "__o", tag = "norm_no_bias_()"} {
    %alloc = memref.alloc() {name = "tmp"} : memref<4x768xbf16>
    call @fill_zeros_bf16_4_768_vector(%alloc) {lib = "fill_zeros_bf16_4_768_vector"} : (memref<4x768xbf16>) -> ()
    call @layer_norm(%arg0, %arg1, %alloc) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
    allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<4x768xbf16>, 1> contains memref<4x768xbf16>
    return
  }
  func.func private @add_bf16_vector(memref<4x768xbf16>, memref<4x768xbf16>, memref<4x768xbf16>)
  func.func @norm_add_bias_0(%arg0: memref<768xbf16>, %arg1: memref<4x768xbf16>, %arg2: !allo.stream<memref<4x768xbf16>, 1>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "__i", tag = "norm_add_bias_()"} {
    %0 = allo.stream_get(%arg2, []) {name = "data"} : !allo.stream<memref<4x768xbf16>, 1> -> memref<4x768xbf16>
    %alloc = memref.alloc() : memref<4x768xbf16>
    linalg.broadcast ins(%arg0 : memref<768xbf16>) outs(%alloc : memref<4x768xbf16>) dimensions = [0] 
    call @add_bf16_vector(%0, %alloc, %arg1) {lib = "add_bf16_vector"} : (memref<4x768xbf16>, memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @norm_no_bias_1(%arg0: memref<4x768xbf16>, %arg1: memref<768xbf16>, %arg2: !allo.stream<memref<4x768xbf16>, 1>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %alloc = memref.alloc() {name = "tmp"} : memref<4x768xbf16>
    call @fill_zeros_bf16_4_768_vector(%alloc) {lib = "fill_zeros_bf16_4_768_vector"} : (memref<4x768xbf16>) -> ()
    call @layer_norm(%arg0, %arg1, %alloc) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
    allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<4x768xbf16>, 1> contains memref<4x768xbf16>
    return
  }
  func.func @norm_no_bias_2(%arg0: memref<4x768xbf16>, %arg1: memref<768xbf16>, %arg2: !allo.stream<memref<4x768xbf16>, 1>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %alloc = memref.alloc() {name = "tmp"} : memref<4x768xbf16>
    call @fill_zeros_bf16_4_768_vector(%alloc) {lib = "fill_zeros_bf16_4_768_vector"} : (memref<4x768xbf16>) -> ()
    call @layer_norm(%arg0, %arg1, %alloc) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
    allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<4x768xbf16>, 1> contains memref<4x768xbf16>
    return
  }
  func.func @norm_no_bias_3(%arg0: memref<4x768xbf16>, %arg1: memref<768xbf16>, %arg2: !allo.stream<memref<4x768xbf16>, 1>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %alloc = memref.alloc() {name = "tmp"} : memref<4x768xbf16>
    call @fill_zeros_bf16_4_768_vector(%alloc) {lib = "fill_zeros_bf16_4_768_vector"} : (memref<4x768xbf16>) -> ()
    call @layer_norm(%arg0, %arg1, %alloc) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
    allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<4x768xbf16>, 1> contains memref<4x768xbf16>
    return
  }
  func.func @norm_no_bias_4(%arg0: memref<4x768xbf16>, %arg1: memref<768xbf16>, %arg2: !allo.stream<memref<4x768xbf16>, 1>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %alloc = memref.alloc() {name = "tmp"} : memref<4x768xbf16>
    call @fill_zeros_bf16_4_768_vector(%alloc) {lib = "fill_zeros_bf16_4_768_vector"} : (memref<4x768xbf16>) -> ()
    call @layer_norm(%arg0, %arg1, %alloc) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
    allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<4x768xbf16>, 1> contains memref<4x768xbf16>
    return
  }
  func.func @norm_no_bias_5(%arg0: memref<4x768xbf16>, %arg1: memref<768xbf16>, %arg2: !allo.stream<memref<4x768xbf16>, 1>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %alloc = memref.alloc() {name = "tmp"} : memref<4x768xbf16>
    call @fill_zeros_bf16_4_768_vector(%alloc) {lib = "fill_zeros_bf16_4_768_vector"} : (memref<4x768xbf16>) -> ()
    call @layer_norm(%arg0, %arg1, %alloc) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
    allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<4x768xbf16>, 1> contains memref<4x768xbf16>
    return
  }
  func.func @norm_no_bias_6(%arg0: memref<4x768xbf16>, %arg1: memref<768xbf16>, %arg2: !allo.stream<memref<4x768xbf16>, 1>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %alloc = memref.alloc() {name = "tmp"} : memref<4x768xbf16>
    call @fill_zeros_bf16_4_768_vector(%alloc) {lib = "fill_zeros_bf16_4_768_vector"} : (memref<4x768xbf16>) -> ()
    call @layer_norm(%arg0, %arg1, %alloc) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
    allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<4x768xbf16>, 1> contains memref<4x768xbf16>
    return
  }
  func.func @norm_no_bias_7(%arg0: memref<4x768xbf16>, %arg1: memref<768xbf16>, %arg2: !allo.stream<memref<4x768xbf16>, 1>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %alloc = memref.alloc() {name = "tmp"} : memref<4x768xbf16>
    call @fill_zeros_bf16_4_768_vector(%alloc) {lib = "fill_zeros_bf16_4_768_vector"} : (memref<4x768xbf16>) -> ()
    call @layer_norm(%arg0, %arg1, %alloc) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
    allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<4x768xbf16>, 1> contains memref<4x768xbf16>
    return
  }
  func.func @norm_add_bias_1(%arg0: memref<768xbf16>, %arg1: memref<4x768xbf16>, %arg2: !allo.stream<memref<4x768xbf16>, 1>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %0 = allo.stream_get(%arg2, []) {name = "data"} : !allo.stream<memref<4x768xbf16>, 1> -> memref<4x768xbf16>
    %alloc = memref.alloc() : memref<4x768xbf16>
    linalg.broadcast ins(%arg0 : memref<768xbf16>) outs(%alloc : memref<4x768xbf16>) dimensions = [0] 
    call @add_bf16_vector(%0, %alloc, %arg1) {lib = "add_bf16_vector"} : (memref<4x768xbf16>, memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @norm_add_bias_2(%arg0: memref<768xbf16>, %arg1: memref<4x768xbf16>, %arg2: !allo.stream<memref<4x768xbf16>, 1>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %0 = allo.stream_get(%arg2, []) {name = "data"} : !allo.stream<memref<4x768xbf16>, 1> -> memref<4x768xbf16>
    %alloc = memref.alloc() : memref<4x768xbf16>
    linalg.broadcast ins(%arg0 : memref<768xbf16>) outs(%alloc : memref<4x768xbf16>) dimensions = [0] 
    call @add_bf16_vector(%0, %alloc, %arg1) {lib = "add_bf16_vector"} : (memref<4x768xbf16>, memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @norm_add_bias_3(%arg0: memref<768xbf16>, %arg1: memref<4x768xbf16>, %arg2: !allo.stream<memref<4x768xbf16>, 1>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %0 = allo.stream_get(%arg2, []) {name = "data"} : !allo.stream<memref<4x768xbf16>, 1> -> memref<4x768xbf16>
    %alloc = memref.alloc() : memref<4x768xbf16>
    linalg.broadcast ins(%arg0 : memref<768xbf16>) outs(%alloc : memref<4x768xbf16>) dimensions = [0] 
    call @add_bf16_vector(%0, %alloc, %arg1) {lib = "add_bf16_vector"} : (memref<4x768xbf16>, memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @norm_add_bias_4(%arg0: memref<768xbf16>, %arg1: memref<4x768xbf16>, %arg2: !allo.stream<memref<4x768xbf16>, 1>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %0 = allo.stream_get(%arg2, []) {name = "data"} : !allo.stream<memref<4x768xbf16>, 1> -> memref<4x768xbf16>
    %alloc = memref.alloc() : memref<4x768xbf16>
    linalg.broadcast ins(%arg0 : memref<768xbf16>) outs(%alloc : memref<4x768xbf16>) dimensions = [0] 
    call @add_bf16_vector(%0, %alloc, %arg1) {lib = "add_bf16_vector"} : (memref<4x768xbf16>, memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @norm_add_bias_5(%arg0: memref<768xbf16>, %arg1: memref<4x768xbf16>, %arg2: !allo.stream<memref<4x768xbf16>, 1>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %0 = allo.stream_get(%arg2, []) {name = "data"} : !allo.stream<memref<4x768xbf16>, 1> -> memref<4x768xbf16>
    %alloc = memref.alloc() : memref<4x768xbf16>
    linalg.broadcast ins(%arg0 : memref<768xbf16>) outs(%alloc : memref<4x768xbf16>) dimensions = [0] 
    call @add_bf16_vector(%0, %alloc, %arg1) {lib = "add_bf16_vector"} : (memref<4x768xbf16>, memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @norm_add_bias_6(%arg0: memref<768xbf16>, %arg1: memref<4x768xbf16>, %arg2: !allo.stream<memref<4x768xbf16>, 1>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %0 = allo.stream_get(%arg2, []) {name = "data"} : !allo.stream<memref<4x768xbf16>, 1> -> memref<4x768xbf16>
    %alloc = memref.alloc() : memref<4x768xbf16>
    linalg.broadcast ins(%arg0 : memref<768xbf16>) outs(%alloc : memref<4x768xbf16>) dimensions = [0] 
    call @add_bf16_vector(%0, %alloc, %arg1) {lib = "add_bf16_vector"} : (memref<4x768xbf16>, memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @norm_add_bias_7(%arg0: memref<768xbf16>, %arg1: memref<4x768xbf16>, %arg2: !allo.stream<memref<4x768xbf16>, 1>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %0 = allo.stream_get(%arg2, []) {name = "data"} : !allo.stream<memref<4x768xbf16>, 1> -> memref<4x768xbf16>
    %alloc = memref.alloc() : memref<4x768xbf16>
    linalg.broadcast ins(%arg0 : memref<768xbf16>) outs(%alloc : memref<4x768xbf16>) dimensions = [0] 
    call @add_bf16_vector(%0, %alloc, %arg1) {lib = "add_bf16_vector"} : (memref<4x768xbf16>, memref<4x768xbf16>, memref<4x768xbf16>) -> ()
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
