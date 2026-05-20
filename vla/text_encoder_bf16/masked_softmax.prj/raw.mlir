module {
  func.func private @masked_softmax_128_bf16(memref<8x128xbf16>, memref<1xi32>, memref<8x128xbf16>)
  func.func @core_0_0(%arg0: memref<8x128xbf16>, %arg1: memref<1xi32>, %arg2: memref<8x128xbf16>) attributes {df.kernel, itypes = "_s_", otypes = "", stypes = "___", tag = "core_()"} {
    call @masked_softmax_128_bf16(%arg0, %arg1, %arg2) : (memref<8x128xbf16>, memref<1xi32>, memref<8x128xbf16>) -> ()
    return
  }
  func.func @masked_softmax_kernel(%arg0: memref<120x128xbf16>, %arg1: memref<1xi32>, %arg2: memref<120x128xbf16>) attributes {dataflow, itypes = "_s_"} {
    return
  }
}
