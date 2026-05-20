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
  func.func @top(%arg0: memref<16x512xf32>, %arg1: memref<16x512xf32>) attributes {dataflow, itypes = "__"} {
    return
  }
}
