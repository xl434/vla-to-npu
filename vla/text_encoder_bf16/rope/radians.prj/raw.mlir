module {
  func.func private @rope_make_radians_float32(memref<64xf32>, memref<32xf32>, memref<64x32xf32>)
  func.func @core_0_0(%arg0: memref<64xf32>, %arg1: memref<32xf32>, %arg2: memref<64x32xf32>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "___", tag = "core_()"} {
    call @rope_make_radians_float32(%arg0, %arg1, %arg2) : (memref<64xf32>, memref<32xf32>, memref<64x32xf32>) -> ()
    return
  }
  func.func @radians_region(%arg0: memref<64xf32>, %arg1: memref<32xf32>, %arg2: memref<64x32xf32>) attributes {dataflow, itypes = "___"} {
    return
  }
}
