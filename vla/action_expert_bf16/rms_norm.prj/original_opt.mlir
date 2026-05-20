module {
  func.func private @rms_norm_bf16(memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>)
  func.func @core_0(%arg0: memref<4x768xbf16>, %arg1: memref<768xbf16>, %arg2: memref<4x768xbf16>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "___", tag = "core_()"} {
    call @rms_norm_bf16(%arg0, %arg1, %arg2) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_1(%arg0: memref<4x768xbf16>, %arg1: memref<768xbf16>, %arg2: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @rms_norm_bf16(%arg0, %arg1, %arg2) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_2(%arg0: memref<4x768xbf16>, %arg1: memref<768xbf16>, %arg2: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @rms_norm_bf16(%arg0, %arg1, %arg2) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_3(%arg0: memref<4x768xbf16>, %arg1: memref<768xbf16>, %arg2: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @rms_norm_bf16(%arg0, %arg1, %arg2) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_4(%arg0: memref<4x768xbf16>, %arg1: memref<768xbf16>, %arg2: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @rms_norm_bf16(%arg0, %arg1, %arg2) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_5(%arg0: memref<4x768xbf16>, %arg1: memref<768xbf16>, %arg2: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @rms_norm_bf16(%arg0, %arg1, %arg2) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_6(%arg0: memref<4x768xbf16>, %arg1: memref<768xbf16>, %arg2: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @rms_norm_bf16(%arg0, %arg1, %arg2) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @core_7(%arg0: memref<4x768xbf16>, %arg1: memref<768xbf16>, %arg2: memref<4x768xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @rms_norm_bf16(%arg0, %arg1, %arg2) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @rms_norm_kernel(%arg0: memref<32x768xbf16>, %arg1: memref<768xbf16>, %arg2: memref<32x768xbf16>) attributes {dataflow, itypes = "___"} {
    return
  }
}
