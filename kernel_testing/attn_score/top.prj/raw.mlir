module {
  func.func private @transpose_matmul_with_scale_bf16(memref<32x64xbf16>, memref<32x64xbf16>, memref<32x32xbf16>)
  func.func @core_0_0(%arg0: memref<32x64xbf16>, %arg1: memref<32x64xbf16>, %arg2: memref<32x32xbf16>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "___", tag = "core_()"} {
    call @transpose_matmul_with_scale_bf16(%arg0, %arg1, %arg2) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x32xbf16>) -> ()
    return
  }
  func.func @top(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {dataflow, itypes = "___"} {
    return
  }
}
