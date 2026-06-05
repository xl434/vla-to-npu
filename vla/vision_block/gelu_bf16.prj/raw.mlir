module {
  func.func private @gelu(memref<4x768xbf16>, memref<4x768xbf16>)
  func.func @core_0_0(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "__", tag = "core_()"} {
    call @gelu(%arg0, %arg1) : (memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @gelu_kernel(%arg0: memref<16x3072xbf16>, %arg1: memref<16x3072xbf16>) attributes {dataflow, itypes = "__"} {
    return
  }
}
