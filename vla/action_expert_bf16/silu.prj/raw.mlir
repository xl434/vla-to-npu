module {
  func.func private @silu_256_bf16(memref<16x256xbf16>, memref<16x256xbf16>)
  func.func @core_0_0(%arg0: memref<16x256xbf16>, %arg1: memref<16x256xbf16>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "__", tag = "core_()"} {
    call @silu_256_bf16(%arg0, %arg1) : (memref<16x256xbf16>, memref<16x256xbf16>) -> ()
    return
  }
  func.func @silu_kernel(%arg0: memref<16x2048xbf16>, %arg1: memref<16x2048xbf16>) attributes {dataflow, itypes = "__"} {
    return
  }
}
