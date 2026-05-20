module {
  func.func private @copy_left32_from64_float32(memref<32x64xf32>, memref<32x32xf32>)
  func.func @core_0_0(%arg0: memref<32x64xf32>, %arg1: memref<32x32xf32>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "__", tag = "core_()"} {
    call @copy_left32_from64_float32(%arg0, %arg1) : (memref<32x64xf32>, memref<32x32xf32>) -> ()
    return
  }
  func.func @copy_left_region(%arg0: memref<32x64xf32>, %arg1: memref<32x32xf32>) attributes {dataflow, itypes = "__"} {
    return
  }
}
