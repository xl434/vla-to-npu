module {
  func.func private @softmax_bf16_32_64(memref<32x64xbf16>, memref<32x64xbf16>)
  func.func @core_0_0(%arg0: memref<32x64xbf16>, %arg1: memref<32x64xbf16>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "__", tag = "core_()"} {
    call @softmax_bf16_32_64(%arg0, %arg1) : (memref<32x64xbf16>, memref<32x64xbf16>) -> ()
    return
  }
  func.func @top(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>) attributes {dataflow, itypes = "__"} {
    return
  }
}
