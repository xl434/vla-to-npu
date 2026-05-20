module {
  func.func private @softmax_bf16(memref<4x1024xbf16>, memref<4x1024xbf16>)
  func.func @core_0(%arg0: memref<4x1024xbf16>, %arg1: memref<4x1024xbf16>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "__", tag = "core_()"} {
    call @softmax_bf16(%arg0, %arg1) : (memref<4x1024xbf16>, memref<4x1024xbf16>) -> ()
    return
  }
  func.func @softmax_kernel(%arg0: memref<1024x1024xbf16>, %arg1: memref<1024x1024xbf16>) attributes {dataflow, itypes = "__"} {
    return
  }
}
