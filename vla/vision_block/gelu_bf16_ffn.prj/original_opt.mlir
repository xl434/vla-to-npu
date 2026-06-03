module {
  func.func private @gelu(memref<4x768xbf16>, memref<4x768xbf16>)
  func.func @core_0_0(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "__", tag = "core_()"} {
    call @gelu(%arg0, %arg1) : (memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_0_1(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @gelu(%arg0, %arg1) : (memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_0_2(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @gelu(%arg0, %arg1) : (memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_0_3(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @gelu(%arg0, %arg1) : (memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_1_0(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @gelu(%arg0, %arg1) : (memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_1_1(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @gelu(%arg0, %arg1) : (memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_1_2(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @gelu(%arg0, %arg1) : (memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_1_3(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @gelu(%arg0, %arg1) : (memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_2_0(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @gelu(%arg0, %arg1) : (memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_2_1(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @gelu(%arg0, %arg1) : (memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_2_2(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @gelu(%arg0, %arg1) : (memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_2_3(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @gelu(%arg0, %arg1) : (memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_3_0(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @gelu(%arg0, %arg1) : (memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_3_1(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @gelu(%arg0, %arg1) : (memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_3_2(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @gelu(%arg0, %arg1) : (memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_3_3(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    call @gelu(%arg0, %arg1) : (memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @top(%arg0: memref<16x3072xbf16>, %arg1: memref<16x3072xbf16>) attributes {dataflow, itypes = "__"} {
    return
  }
}
