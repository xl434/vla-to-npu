module {
  func.func private @softmax_128_bf16(memref<8x128xbf16>, memref<8x128xbf16>)
  func.func @core_0(%arg0: memref<8x128xbf16>, %arg1: memref<8x128xbf16>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "__", tag = "core_()"} {
    call @softmax_128_bf16(%arg0, %arg1) : (memref<8x128xbf16>, memref<8x128xbf16>) -> ()
    return
  }
  func.func @core_1(%arg0: memref<8x128xbf16>, %arg1: memref<8x128xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_128_bf16(%arg0, %arg1) : (memref<8x128xbf16>, memref<8x128xbf16>) -> ()
    return
  }
  func.func @core_2(%arg0: memref<8x128xbf16>, %arg1: memref<8x128xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_128_bf16(%arg0, %arg1) : (memref<8x128xbf16>, memref<8x128xbf16>) -> ()
    return
  }
  func.func @core_3(%arg0: memref<8x128xbf16>, %arg1: memref<8x128xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_128_bf16(%arg0, %arg1) : (memref<8x128xbf16>, memref<8x128xbf16>) -> ()
    return
  }
  func.func @core_4(%arg0: memref<8x128xbf16>, %arg1: memref<8x128xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_128_bf16(%arg0, %arg1) : (memref<8x128xbf16>, memref<8x128xbf16>) -> ()
    return
  }
  func.func @core_5(%arg0: memref<8x128xbf16>, %arg1: memref<8x128xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_128_bf16(%arg0, %arg1) : (memref<8x128xbf16>, memref<8x128xbf16>) -> ()
    return
  }
  func.func @core_6(%arg0: memref<8x128xbf16>, %arg1: memref<8x128xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_128_bf16(%arg0, %arg1) : (memref<8x128xbf16>, memref<8x128xbf16>) -> ()
    return
  }
  func.func @core_7(%arg0: memref<8x128xbf16>, %arg1: memref<8x128xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_128_bf16(%arg0, %arg1) : (memref<8x128xbf16>, memref<8x128xbf16>) -> ()
    return
  }
  func.func @core_8(%arg0: memref<8x128xbf16>, %arg1: memref<8x128xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_128_bf16(%arg0, %arg1) : (memref<8x128xbf16>, memref<8x128xbf16>) -> ()
    return
  }
  func.func @core_9(%arg0: memref<8x128xbf16>, %arg1: memref<8x128xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_128_bf16(%arg0, %arg1) : (memref<8x128xbf16>, memref<8x128xbf16>) -> ()
    return
  }
  func.func @core_10(%arg0: memref<8x128xbf16>, %arg1: memref<8x128xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_128_bf16(%arg0, %arg1) : (memref<8x128xbf16>, memref<8x128xbf16>) -> ()
    return
  }
  func.func @core_11(%arg0: memref<8x128xbf16>, %arg1: memref<8x128xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_128_bf16(%arg0, %arg1) : (memref<8x128xbf16>, memref<8x128xbf16>) -> ()
    return
  }
  func.func @core_12(%arg0: memref<8x128xbf16>, %arg1: memref<8x128xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_128_bf16(%arg0, %arg1) : (memref<8x128xbf16>, memref<8x128xbf16>) -> ()
    return
  }
  func.func @core_13(%arg0: memref<8x128xbf16>, %arg1: memref<8x128xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_128_bf16(%arg0, %arg1) : (memref<8x128xbf16>, memref<8x128xbf16>) -> ()
    return
  }
  func.func @core_14(%arg0: memref<8x128xbf16>, %arg1: memref<8x128xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_128_bf16(%arg0, %arg1) : (memref<8x128xbf16>, memref<8x128xbf16>) -> ()
    return
  }
  func.func @core_15(%arg0: memref<8x128xbf16>, %arg1: memref<8x128xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @softmax_128_bf16(%arg0, %arg1) : (memref<8x128xbf16>, memref<8x128xbf16>) -> ()
    return
  }
  func.func @softmax_kernel(%arg0: memref<128x128xbf16>, %arg1: memref<128x128xbf16>) attributes {dataflow, itypes = "__"} {
    return
  }
}
