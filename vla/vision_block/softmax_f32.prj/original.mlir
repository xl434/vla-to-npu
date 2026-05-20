module {
  func.func private @softmax_float32_seq1024(memref<4x512xf32>, memref<4x512xf32>)
  func.func @core_0_0(%arg0: memref<4x512xf32>, %arg1: memref<4x512xf32>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "__", tag = "core_()"} {
    call @softmax_float32_seq1024(%arg0, %arg1) : (memref<4x512xf32>, memref<4x512xf32>) -> ()
    return
  }
  func.func @core_1_0(%arg0: memref<4x512xf32>, %arg1: memref<4x512xf32>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_float32_seq1024(%arg0, %arg1) : (memref<4x512xf32>, memref<4x512xf32>) -> ()
    return
  }
  func.func @core_2_0(%arg0: memref<4x512xf32>, %arg1: memref<4x512xf32>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_float32_seq1024(%arg0, %arg1) : (memref<4x512xf32>, memref<4x512xf32>) -> ()
    return
  }
  func.func @core_3_0(%arg0: memref<4x512xf32>, %arg1: memref<4x512xf32>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_float32_seq1024(%arg0, %arg1) : (memref<4x512xf32>, memref<4x512xf32>) -> ()
    return
  }
  func.func @core_4_0(%arg0: memref<4x512xf32>, %arg1: memref<4x512xf32>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_float32_seq1024(%arg0, %arg1) : (memref<4x512xf32>, memref<4x512xf32>) -> ()
    return
  }
  func.func @core_5_0(%arg0: memref<4x512xf32>, %arg1: memref<4x512xf32>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_float32_seq1024(%arg0, %arg1) : (memref<4x512xf32>, memref<4x512xf32>) -> ()
    return
  }
  func.func @core_6_0(%arg0: memref<4x512xf32>, %arg1: memref<4x512xf32>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_float32_seq1024(%arg0, %arg1) : (memref<4x512xf32>, memref<4x512xf32>) -> ()
    return
  }
  func.func @core_7_0(%arg0: memref<4x512xf32>, %arg1: memref<4x512xf32>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_float32_seq1024(%arg0, %arg1) : (memref<4x512xf32>, memref<4x512xf32>) -> ()
    return
  }
  func.func @core_8_0(%arg0: memref<4x512xf32>, %arg1: memref<4x512xf32>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_float32_seq1024(%arg0, %arg1) : (memref<4x512xf32>, memref<4x512xf32>) -> ()
    return
  }
  func.func @core_9_0(%arg0: memref<4x512xf32>, %arg1: memref<4x512xf32>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_float32_seq1024(%arg0, %arg1) : (memref<4x512xf32>, memref<4x512xf32>) -> ()
    return
  }
  func.func @core_10_0(%arg0: memref<4x512xf32>, %arg1: memref<4x512xf32>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_float32_seq1024(%arg0, %arg1) : (memref<4x512xf32>, memref<4x512xf32>) -> ()
    return
  }
  func.func @core_11_0(%arg0: memref<4x512xf32>, %arg1: memref<4x512xf32>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_float32_seq1024(%arg0, %arg1) : (memref<4x512xf32>, memref<4x512xf32>) -> ()
    return
  }
  func.func @core_12_0(%arg0: memref<4x512xf32>, %arg1: memref<4x512xf32>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_float32_seq1024(%arg0, %arg1) : (memref<4x512xf32>, memref<4x512xf32>) -> ()
    return
  }
  func.func @core_13_0(%arg0: memref<4x512xf32>, %arg1: memref<4x512xf32>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_float32_seq1024(%arg0, %arg1) : (memref<4x512xf32>, memref<4x512xf32>) -> ()
    return
  }
  func.func @core_14_0(%arg0: memref<4x512xf32>, %arg1: memref<4x512xf32>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_float32_seq1024(%arg0, %arg1) : (memref<4x512xf32>, memref<4x512xf32>) -> ()
    return
  }
  func.func @core_15_0(%arg0: memref<4x512xf32>, %arg1: memref<4x512xf32>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_float32_seq1024(%arg0, %arg1) : (memref<4x512xf32>, memref<4x512xf32>) -> ()
    return
  }
  func.func @softmax_kernel(%arg0: memref<64x512xf32>, %arg1: memref<64x512xf32>) attributes {dataflow, itypes = "__"} {
    return
  }
}
