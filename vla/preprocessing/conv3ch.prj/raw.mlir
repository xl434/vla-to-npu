module {
  func.func private @conv3ch(memref<192x64xbf16>, memref<48x16xbf16>, memref<4x4xbf16>)
  func.func @core_0_0(%arg0: memref<192x64xbf16>, %arg1: memref<48x16xbf16>, %arg2: memref<4x4xbf16>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "___", tag = "core_()"} {
    call @conv3ch(%arg0, %arg1, %arg2) : (memref<192x64xbf16>, memref<48x16xbf16>, memref<4x4xbf16>) -> ()
    return
  }
  func.func @conv3ch_kernel(%arg0: memref<768x256xbf16>, %arg1: memref<48x16xbf16>, %arg2: memref<16x16xbf16>) attributes {dataflow, itypes = "___"} {
    return
  }
}
