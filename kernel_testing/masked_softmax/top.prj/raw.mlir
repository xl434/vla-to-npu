module {
  func.func private @masked_softmax_float32(memref<32x64xf32>, memref<1xi32>, memref<32x64xf32>)
  func.func @core_0_0(%arg0: memref<32x64xf32>, %arg1: memref<1xi32>, %arg2: memref<32x64xf32>) attributes {df.kernel, itypes = "_s_", otypes = "", stypes = "___", tag = "core_()"} {
    call @masked_softmax_float32(%arg0, %arg1, %arg2) : (memref<32x64xf32>, memref<1xi32>, memref<32x64xf32>) -> ()
    return
  }
  func.func @top(%arg0: memref<64x192xf32>, %arg1: memref<2xi32>, %arg2: memref<64x192xf32>) attributes {dataflow, itypes = "_s_"} {
    return
  }
}
