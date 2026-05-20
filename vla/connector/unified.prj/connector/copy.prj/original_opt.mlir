module {
  func.func @mod_0(%arg0: memref<1x768xbf16>, %arg1: memref<1x768xbf16>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "__", tag = "mod_()"} {
    %subview = memref.subview %arg0[0, 0] [1, 768] [1, 1] {from = "local_A"} : memref<1x768xbf16> to memref<768xbf16, strided<[1]>>
    %subview_0 = memref.subview %arg1[0, 0] [1, 768] [1, 1] : memref<1x768xbf16> to memref<768xbf16, strided<[1]>>
    memref.copy %subview, %subview_0 {to = "local_C"} : memref<768xbf16, strided<[1]>> to memref<768xbf16, strided<[1]>>
    return
  }
  func.func @mod_1(%arg0: memref<1x768xbf16>, %arg1: memref<1x768xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    %subview = memref.subview %arg0[0, 0] [1, 768] [1, 1] {from = "local_A"} : memref<1x768xbf16> to memref<768xbf16, strided<[1]>>
    %subview_0 = memref.subview %arg1[0, 0] [1, 768] [1, 1] : memref<1x768xbf16> to memref<768xbf16, strided<[1]>>
    memref.copy %subview, %subview_0 {to = "local_C"} : memref<768xbf16, strided<[1]>> to memref<768xbf16, strided<[1]>>
    return
  }
  func.func @mod_2(%arg0: memref<1x768xbf16>, %arg1: memref<1x768xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    %subview = memref.subview %arg0[0, 0] [1, 768] [1, 1] {from = "local_A"} : memref<1x768xbf16> to memref<768xbf16, strided<[1]>>
    %subview_0 = memref.subview %arg1[0, 0] [1, 768] [1, 1] : memref<1x768xbf16> to memref<768xbf16, strided<[1]>>
    memref.copy %subview, %subview_0 {to = "local_C"} : memref<768xbf16, strided<[1]>> to memref<768xbf16, strided<[1]>>
    return
  }
  func.func @mod_3(%arg0: memref<1x768xbf16>, %arg1: memref<1x768xbf16>) attributes {df.kernel, input_depth = [0, 0], output_depth = []} {
    %subview = memref.subview %arg0[0, 0] [1, 768] [1, 1] {from = "local_A"} : memref<1x768xbf16> to memref<768xbf16, strided<[1]>>
    %subview_0 = memref.subview %arg1[0, 0] [1, 768] [1, 1] : memref<1x768xbf16> to memref<768xbf16, strided<[1]>>
    memref.copy %subview, %subview_0 {to = "local_C"} : memref<768xbf16, strided<[1]>> to memref<768xbf16, strided<[1]>>
    return
  }
  func.func @copy_region(%arg0: memref<4x768xbf16>, %arg1: memref<1x3072xbf16>) attributes {dataflow, itypes = "__"} {
    return
  }
}
