module {
  func.func private @silu_bf16(memref<4x768xbf16>, memref<4x768xbf16>)
  func.func @core_0_0(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "__", tag = "core_()"} {
    call @silu_bf16(%arg0, %arg1) : (memref<4x768xbf16>, memref<4x768xbf16>) -> ()
    return
  }
  func.func @top(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) attributes {dataflow, itypes = "__"} {
    return
  }
}
