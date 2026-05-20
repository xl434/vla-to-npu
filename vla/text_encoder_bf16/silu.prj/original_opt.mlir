module {
  func.func private @silu_160_bf16(memref<16x160xbf16>, memref<16x160xbf16>)
  func.func @core_0_0(%arg0: memref<16x160xbf16>, %arg1: memref<16x160xbf16>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "__", tag = "core_()"} {
    call @silu_160_bf16(%arg0, %arg1) : (memref<16x160xbf16>, memref<16x160xbf16>) -> ()
    return
  }
  func.func @core_0_1(%arg0: memref<16x160xbf16>, %arg1: memref<16x160xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @silu_160_bf16(%arg0, %arg1) : (memref<16x160xbf16>, memref<16x160xbf16>) -> ()
    return
  }
  func.func @core_0_2(%arg0: memref<16x160xbf16>, %arg1: memref<16x160xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @silu_160_bf16(%arg0, %arg1) : (memref<16x160xbf16>, memref<16x160xbf16>) -> ()
    return
  }
  func.func @core_0_3(%arg0: memref<16x160xbf16>, %arg1: memref<16x160xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @silu_160_bf16(%arg0, %arg1) : (memref<16x160xbf16>, memref<16x160xbf16>) -> ()
    return
  }
  func.func @core_0_4(%arg0: memref<16x160xbf16>, %arg1: memref<16x160xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @silu_160_bf16(%arg0, %arg1) : (memref<16x160xbf16>, memref<16x160xbf16>) -> ()
    return
  }
  func.func @core_0_5(%arg0: memref<16x160xbf16>, %arg1: memref<16x160xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @silu_160_bf16(%arg0, %arg1) : (memref<16x160xbf16>, memref<16x160xbf16>) -> ()
    return
  }
  func.func @core_0_6(%arg0: memref<16x160xbf16>, %arg1: memref<16x160xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @silu_160_bf16(%arg0, %arg1) : (memref<16x160xbf16>, memref<16x160xbf16>) -> ()
    return
  }
  func.func @core_0_7(%arg0: memref<16x160xbf16>, %arg1: memref<16x160xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @silu_160_bf16(%arg0, %arg1) : (memref<16x160xbf16>, memref<16x160xbf16>) -> ()
    return
  }
  func.func @core_0_8(%arg0: memref<16x160xbf16>, %arg1: memref<16x160xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @silu_160_bf16(%arg0, %arg1) : (memref<16x160xbf16>, memref<16x160xbf16>) -> ()
    return
  }
  func.func @core_0_9(%arg0: memref<16x160xbf16>, %arg1: memref<16x160xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @silu_160_bf16(%arg0, %arg1) : (memref<16x160xbf16>, memref<16x160xbf16>) -> ()
    return
  }
  func.func @core_0_10(%arg0: memref<16x160xbf16>, %arg1: memref<16x160xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @silu_160_bf16(%arg0, %arg1) : (memref<16x160xbf16>, memref<16x160xbf16>) -> ()
    return
  }
  func.func @core_0_11(%arg0: memref<16x160xbf16>, %arg1: memref<16x160xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @silu_160_bf16(%arg0, %arg1) : (memref<16x160xbf16>, memref<16x160xbf16>) -> ()
    return
  }
  func.func @core_0_12(%arg0: memref<16x160xbf16>, %arg1: memref<16x160xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @silu_160_bf16(%arg0, %arg1) : (memref<16x160xbf16>, memref<16x160xbf16>) -> ()
    return
  }
  func.func @core_0_13(%arg0: memref<16x160xbf16>, %arg1: memref<16x160xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @silu_160_bf16(%arg0, %arg1) : (memref<16x160xbf16>, memref<16x160xbf16>) -> ()
    return
  }
  func.func @core_0_14(%arg0: memref<16x160xbf16>, %arg1: memref<16x160xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @silu_160_bf16(%arg0, %arg1) : (memref<16x160xbf16>, memref<16x160xbf16>) -> ()
    return
  }
  func.func @core_0_15(%arg0: memref<16x160xbf16>, %arg1: memref<16x160xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @silu_160_bf16(%arg0, %arg1) : (memref<16x160xbf16>, memref<16x160xbf16>) -> ()
    return
  }
  func.func @silu_kernel(%arg0: memref<16x2560xbf16>, %arg1: memref<16x2560xbf16>) attributes {dataflow, itypes = "__"} {
    return
  }
}
