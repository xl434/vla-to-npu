module {
  func.func private @softmax_bf16(memref<4x1024xbf16>, memref<4x1024xbf16>)
  func.func @core_0x16(%arg0: memref<4x1024xbf16>, %arg1: memref<4x1024xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_bf16(%arg0, %arg1) : (memref<4x1024xbf16>, memref<4x1024xbf16>) -> ()
    return
  }
  func.func @core_1x16(%arg0: memref<4x1024xbf16>, %arg1: memref<4x1024xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_bf16(%arg0, %arg1) : (memref<4x1024xbf16>, memref<4x1024xbf16>) -> ()
    return
  }
  func.func @core_2x16(%arg0: memref<4x1024xbf16>, %arg1: memref<4x1024xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_bf16(%arg0, %arg1) : (memref<4x1024xbf16>, memref<4x1024xbf16>) -> ()
    return
  }
  func.func @core_3x16(%arg0: memref<4x1024xbf16>, %arg1: memref<4x1024xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_bf16(%arg0, %arg1) : (memref<4x1024xbf16>, memref<4x1024xbf16>) -> ()
    return
  }
  func.func @core_4x16(%arg0: memref<4x1024xbf16>, %arg1: memref<4x1024xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_bf16(%arg0, %arg1) : (memref<4x1024xbf16>, memref<4x1024xbf16>) -> ()
    return
  }
  func.func @core_5x16(%arg0: memref<4x1024xbf16>, %arg1: memref<4x1024xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_bf16(%arg0, %arg1) : (memref<4x1024xbf16>, memref<4x1024xbf16>) -> ()
    return
  }
  func.func @core_6x16(%arg0: memref<4x1024xbf16>, %arg1: memref<4x1024xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_bf16(%arg0, %arg1) : (memref<4x1024xbf16>, memref<4x1024xbf16>) -> ()
    return
  }
  func.func @core_7x16(%arg0: memref<4x1024xbf16>, %arg1: memref<4x1024xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_bf16(%arg0, %arg1) : (memref<4x1024xbf16>, memref<4x1024xbf16>) -> ()
    return
  }
  func.func @core_8x16(%arg0: memref<4x1024xbf16>, %arg1: memref<4x1024xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_bf16(%arg0, %arg1) : (memref<4x1024xbf16>, memref<4x1024xbf16>) -> ()
    return
  }
  func.func @core_9x16(%arg0: memref<4x1024xbf16>, %arg1: memref<4x1024xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_bf16(%arg0, %arg1) : (memref<4x1024xbf16>, memref<4x1024xbf16>) -> ()
    return
  }
  func.func @core_10x16(%arg0: memref<4x1024xbf16>, %arg1: memref<4x1024xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_bf16(%arg0, %arg1) : (memref<4x1024xbf16>, memref<4x1024xbf16>) -> ()
    return
  }
  func.func @core_11x16(%arg0: memref<4x1024xbf16>, %arg1: memref<4x1024xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_bf16(%arg0, %arg1) : (memref<4x1024xbf16>, memref<4x1024xbf16>) -> ()
    return
  }
  func.func @core_12x16(%arg0: memref<4x1024xbf16>, %arg1: memref<4x1024xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_bf16(%arg0, %arg1) : (memref<4x1024xbf16>, memref<4x1024xbf16>) -> ()
    return
  }
  func.func @core_13x16(%arg0: memref<4x1024xbf16>, %arg1: memref<4x1024xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_bf16(%arg0, %arg1) : (memref<4x1024xbf16>, memref<4x1024xbf16>) -> ()
    return
  }
  func.func @core_14x16(%arg0: memref<4x1024xbf16>, %arg1: memref<4x1024xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_bf16(%arg0, %arg1) : (memref<4x1024xbf16>, memref<4x1024xbf16>) -> ()
    return
  }
  func.func @core_15x16(%arg0: memref<4x1024xbf16>, %arg1: memref<4x1024xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_bf16(%arg0, %arg1) : (memref<4x1024xbf16>, memref<4x1024xbf16>) -> ()
    return
  }
  func.func @softmax_kernel(%arg0: memref<1024x1024xbf16>, %arg1: memref<1024x1024xbf16>) attributes {dataflow, itypes = "__"} {
    return
  }
}
