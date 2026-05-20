module {
  func.func private @conv(memref<64x64xf32>, memref<16x16xf32>, memref<4x4xf32>)
  func.func @core_0_0(%arg0: memref<64x64xf32>, %arg1: memref<16x16xf32>, %arg2: memref<4x4xf32>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "___", tag = "core_()"} {
    call @conv(%arg0, %arg1, %arg2) : (memref<64x64xf32>, memref<16x16xf32>, memref<4x4xf32>) -> ()
    return
  }
  func.func @conv_kernel(%arg0: memref<64x64xf32>, %arg1: memref<16x16xf32>, %arg2: memref<4x4xf32>) attributes {dataflow, itypes = "___"} {
    return
  }
}
