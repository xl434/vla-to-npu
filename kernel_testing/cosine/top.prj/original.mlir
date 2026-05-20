module {
  func.func private @cos_float32(memref<32x64xf32>, memref<32x64xf32>)
  func.func @core_0_0(%arg0: memref<32x64xf32>, %arg1: memref<32x64xf32>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "__", tag = "core_()"} {
    call @cos_float32(%arg0, %arg1) : (memref<32x64xf32>, memref<32x64xf32>) -> ()
    return
  }
  func.func @top(%arg0: memref<32x64xf32>, %arg1: memref<32x64xf32>) attributes {dataflow, itypes = "__"} {
    return
  }
}
