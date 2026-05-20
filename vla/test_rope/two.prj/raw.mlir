module {
  func.func private @rope_fused_float32(memref<32x64xf32>, memref<32x64xf32>, memref<32x64xf32>)
  func.func @core_0_0(%arg0: memref<32x64xf32>, %arg1: memref<32x64xf32>, %arg2: memref<32x64xf32>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "___", tag = "core_()"} {
    call @rope_fused_float32(%arg0, %arg1, %arg2) : (memref<32x64xf32>, memref<32x64xf32>, memref<32x64xf32>) -> ()
    return
  }
  func.func @rope_2head(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) attributes {dataflow, itypes = "___"} {
    return
  }
}
