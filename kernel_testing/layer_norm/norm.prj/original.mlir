module {
  func.func private @layer_norm(memref<4x768xf32>, memref<768xf32>, memref<4x768xf32>)
  func.func private @fill_zeros_f32_4_768_vector(memref<4x768xf32>)
  func.func @norm_no_bias_0(%arg0: memref<4x768xf32>, %arg1: memref<768xf32>, %arg2: !allo.stream<memref<4x768xf32>, 1>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "__o", tag = "norm_no_bias_()"} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {name = "tmp"} : memref<4x768xf32>
    call @fill_zeros_f32_4_768_vector(%alloc) {lib = "fill_zeros_f32_4_768_vector"} : (memref<4x768xf32>) -> ()
    call @layer_norm(%arg0, %arg1, %alloc) : (memref<4x768xf32>, memref<768xf32>, memref<4x768xf32>) -> ()
    allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<4x768xf32>, 1> contains memref<4x768xf32>
    return
  }
  func.func private @add_f32_vector(memref<4x768xf32>, memref<4x768xf32>, memref<4x768xf32>)
  func.func @norm_add_bias_0(%arg0: memref<768xf32>, %arg1: memref<4x768xf32>, %arg2: !allo.stream<memref<4x768xf32>, 1>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "__i", tag = "norm_add_bias_()"} {
    %0 = allo.stream_get(%arg2, []) {name = "data"} : !allo.stream<memref<4x768xf32>, 1> -> memref<4x768xf32>
    %alloc = memref.alloc() : memref<4x768xf32>
    linalg.broadcast ins(%arg0 : memref<768xf32>) outs(%alloc : memref<4x768xf32>) dimensions = [0] 
    %alloc_0 = memref.alloc() : memref<4x768xf32>
    call @add_f32_vector(%0, %alloc, %alloc_0) {lib = "add_f32_vector"} : (memref<4x768xf32>, memref<4x768xf32>, memref<4x768xf32>) -> ()
    memref.copy %alloc_0, %arg1 {to = "local_output_x"} : memref<4x768xf32> to memref<4x768xf32>
    return
  }
  func.func @norm_no_bias_1(%arg0: memref<4x768xf32>, %arg1: memref<768xf32>, %arg2: !allo.stream<memref<4x768xf32>, 1>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {name = "tmp"} : memref<4x768xf32>
    call @fill_zeros_f32_4_768_vector(%alloc) {lib = "fill_zeros_f32_4_768_vector"} : (memref<4x768xf32>) -> ()
    call @layer_norm(%arg0, %arg1, %alloc) : (memref<4x768xf32>, memref<768xf32>, memref<4x768xf32>) -> ()
    allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<4x768xf32>, 1> contains memref<4x768xf32>
    return
  }
  func.func @norm_no_bias_2(%arg0: memref<4x768xf32>, %arg1: memref<768xf32>, %arg2: !allo.stream<memref<4x768xf32>, 1>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {name = "tmp"} : memref<4x768xf32>
    call @fill_zeros_f32_4_768_vector(%alloc) {lib = "fill_zeros_f32_4_768_vector"} : (memref<4x768xf32>) -> ()
    call @layer_norm(%arg0, %arg1, %alloc) : (memref<4x768xf32>, memref<768xf32>, memref<4x768xf32>) -> ()
    allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<4x768xf32>, 1> contains memref<4x768xf32>
    return
  }
  func.func @norm_no_bias_3(%arg0: memref<4x768xf32>, %arg1: memref<768xf32>, %arg2: !allo.stream<memref<4x768xf32>, 1>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {name = "tmp"} : memref<4x768xf32>
    call @fill_zeros_f32_4_768_vector(%alloc) {lib = "fill_zeros_f32_4_768_vector"} : (memref<4x768xf32>) -> ()
    call @layer_norm(%arg0, %arg1, %alloc) : (memref<4x768xf32>, memref<768xf32>, memref<4x768xf32>) -> ()
    allo.stream_put(%arg2, [], %alloc) : !allo.stream<memref<4x768xf32>, 1> contains memref<4x768xf32>
    return
  }
  func.func @norm_add_bias_1(%arg0: memref<768xf32>, %arg1: memref<4x768xf32>, %arg2: !allo.stream<memref<4x768xf32>, 1>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %0 = allo.stream_get(%arg2, []) {name = "data"} : !allo.stream<memref<4x768xf32>, 1> -> memref<4x768xf32>
    %alloc = memref.alloc() : memref<4x768xf32>
    linalg.broadcast ins(%arg0 : memref<768xf32>) outs(%alloc : memref<4x768xf32>) dimensions = [0] 
    %alloc_0 = memref.alloc() : memref<4x768xf32>
    call @add_f32_vector(%0, %alloc, %alloc_0) {lib = "add_f32_vector"} : (memref<4x768xf32>, memref<4x768xf32>, memref<4x768xf32>) -> ()
    memref.copy %alloc_0, %arg1 {to = "local_output_x"} : memref<4x768xf32> to memref<4x768xf32>
    return
  }
  func.func @norm_add_bias_2(%arg0: memref<768xf32>, %arg1: memref<4x768xf32>, %arg2: !allo.stream<memref<4x768xf32>, 1>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %0 = allo.stream_get(%arg2, []) {name = "data"} : !allo.stream<memref<4x768xf32>, 1> -> memref<4x768xf32>
    %alloc = memref.alloc() : memref<4x768xf32>
    linalg.broadcast ins(%arg0 : memref<768xf32>) outs(%alloc : memref<4x768xf32>) dimensions = [0] 
    %alloc_0 = memref.alloc() : memref<4x768xf32>
    call @add_f32_vector(%0, %alloc, %alloc_0) {lib = "add_f32_vector"} : (memref<4x768xf32>, memref<4x768xf32>, memref<4x768xf32>) -> ()
    memref.copy %alloc_0, %arg1 {to = "local_output_x"} : memref<4x768xf32> to memref<4x768xf32>
    return
  }
  func.func @norm_add_bias_3(%arg0: memref<768xf32>, %arg1: memref<4x768xf32>, %arg2: !allo.stream<memref<4x768xf32>, 1>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    %0 = allo.stream_get(%arg2, []) {name = "data"} : !allo.stream<memref<4x768xf32>, 1> -> memref<4x768xf32>
    %alloc = memref.alloc() : memref<4x768xf32>
    linalg.broadcast ins(%arg0 : memref<768xf32>) outs(%alloc : memref<4x768xf32>) dimensions = [0] 
    %alloc_0 = memref.alloc() : memref<4x768xf32>
    call @add_f32_vector(%0, %alloc, %alloc_0) {lib = "add_f32_vector"} : (memref<4x768xf32>, memref<4x768xf32>, memref<4x768xf32>) -> ()
    memref.copy %alloc_0, %arg1 {to = "local_output_x"} : memref<4x768xf32> to memref<4x768xf32>
    return
  }
  func.func @layer_norm_kernel(%arg0: memref<16x768xf32>, %arg1: memref<768xf32>, %arg2: memref<768xf32>, %arg3: memref<16x768xf32>) attributes {dataflow, itypes = "____"} {
    %0 = allo.stream_construct() {name = "pipe_0"} : !allo.stream<memref<4x768xf32>, 1>
    %1 = allo.stream_construct() {name = "pipe_1"} : !allo.stream<memref<4x768xf32>, 1>
    %2 = allo.stream_construct() {name = "pipe_2"} : !allo.stream<memref<4x768xf32>, 1>
    %3 = allo.stream_construct() {name = "pipe_3"} : !allo.stream<memref<4x768xf32>, 1>
    return
  }
}
