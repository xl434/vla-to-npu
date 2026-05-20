module {
  func.func private @sub32_float32(memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>)
  func.func @core_0_0(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "___", tag = "core_()"} {
    call @sub32_float32(%arg0, %arg1, %arg2) : (memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) -> ()
    return
  }
  func.func @sub32_region(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) attributes {dataflow, itypes = "___"} {
    return
  }
}
