module {
  func.func private @add(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
  func.func @core_0_0(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "___", tag = "core_()"} {
    call @add(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
    return
  }
  func.func @add_kernel(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {dataflow, itypes = "___"} {
    return
  }
}
