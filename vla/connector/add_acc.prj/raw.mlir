module {
  func.func @core_0_0(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "___", tag = "core_()"} {
    %alloc = memref.alloc() : memref<64x64xbf16>
    linalg.add {op_name = "add_0"} ins(%arg0, %arg1 : memref<64x64xbf16>, memref<64x64xbf16>) outs(%alloc : memref<64x64xbf16>)
    memref.copy %alloc, %arg2 {to = "local_out"} : memref<64x64xbf16> to memref<64x64xbf16>
    return
  }
  func.func @add_acc_region(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) attributes {dataflow, itypes = "___"} {
    return
  }
}
