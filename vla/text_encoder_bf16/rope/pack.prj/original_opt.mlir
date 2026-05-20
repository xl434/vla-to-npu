module {
  func.func private @pack32to64_float32(memref<64x32xf32>, memref<64x64xf32>)
  func.func @core_0_0(%arg0: memref<64x32xf32>, %arg1: memref<64x64xf32>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "__", tag = "core_()"} {
    call @pack32to64_float32(%arg0, %arg1) : (memref<64x32xf32>, memref<64x64xf32>) -> ()
    return
  }
  func.func @pack_region(%arg0: memref<64x32xf32>, %arg1: memref<64x64xf32>) attributes {dataflow, itypes = "__"} {
    return
  }
}
