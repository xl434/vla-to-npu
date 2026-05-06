module {
  func.func @mod_0(%arg0: memref<1x32xbf16>, %arg1: memref<1x32xbf16>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "__", tag = "mod_()"} {
    %subview = memref.subview %arg0[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
    %subview_0 = memref.subview %arg1[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
    memref.copy %subview, %subview_0 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
    return
  }
  func.func @copy_region(%arg0: memref<8x32xbf16>, %arg1: memref<1x256xbf16>) attributes {dataflow, itypes = "__"} {
    return
  }
}
