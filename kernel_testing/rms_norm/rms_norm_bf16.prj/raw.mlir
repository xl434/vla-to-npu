module {
  func.func private @rms_norm_bf16(memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>)
  func.func @core_0(%arg0: memref<4x768xbf16>, %arg1: memref<768xbf16>, %arg2: memref<4x768xbf16>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "___", tag = "core_()"} {
    call @rms_norm_bf16(%arg0, %arg1, %arg2) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @rms_norm_kernel(%arg0: memref<16x768xbf16>, %arg1: memref<768xbf16>, %arg2: memref<16x768xbf16>) attributes {dataflow, itypes = "___"} {
    return
  }
}
