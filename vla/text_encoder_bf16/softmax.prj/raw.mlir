module {
  func.func private @softmax_128_bf16(memref<8x128xbf16>, memref<8x128xbf16>)
  func.func @core_0(%arg0: memref<8x128xbf16>, %arg1: memref<8x128xbf16>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "__", tag = "core_()"} {
    call @softmax_128_bf16(%arg0, %arg1) : (memref<8x128xbf16>, memref<8x128xbf16>) -> ()
    return
  }
  func.func @softmax_kernel(%arg0: memref<128x128xbf16>, %arg1: memref<128x128xbf16>) attributes {dataflow, itypes = "__"} {
    return
  }
}
