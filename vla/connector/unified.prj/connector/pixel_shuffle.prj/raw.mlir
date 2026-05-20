module {
  func.func @mod_0(%arg0: memref<8x768xbf16>, %arg1: memref<8x768xbf16>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "__", tag = "mod_()"} {
    memref.copy %arg0, %arg1 {to = "local_Out"} : memref<8x768xbf16> to memref<8x768xbf16>
    return
  }
  func.func @pixel_shuffle_region(%arg0: memref<32x768xbf16>, %arg1: memref<8x3072xbf16>) attributes {dataflow, itypes = "__"} {
    return
  }
}
