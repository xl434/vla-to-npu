module {
  func.func private @silu_160_bf16(memref<16x160xbf16>, memref<16x160xbf16>)
  func.func @core_0_0(%arg0: memref<16x160xbf16>, %arg1: memref<16x160xbf16>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "__", tag = "core_()"} {
    call @silu_160_bf16(%arg0, %arg1) : (memref<16x160xbf16>, memref<16x160xbf16>) -> ()
    return
  }
  func.func @silu_kernel(%arg0: memref<16x2560xbf16>, %arg1: memref<16x2560xbf16>) attributes {dataflow, itypes = "__"} {
    return
  }
}
