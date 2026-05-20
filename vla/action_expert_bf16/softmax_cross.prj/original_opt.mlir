module {
  func.func private @softmax_bf16_32_128(memref<32x128xbf16>, memref<32x128xbf16>)
  func.func @core_0_0(%arg0: memref<32x128xbf16>, %arg1: memref<32x128xbf16>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "__", tag = "core_()"} {
    call @softmax_bf16_32_128(%arg0, %arg1) : (memref<32x128xbf16>, memref<32x128xbf16>) -> ()
    return
  }
  func.func @softmax_cross_kernel(%arg0: memref<32x128xbf16>, %arg1: memref<32x128xbf16>) attributes {dataflow, itypes = "__"} {
    return
  }
}
