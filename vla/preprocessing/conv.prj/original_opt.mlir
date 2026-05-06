module {
  func.func private @conv(memref<64x64xbf16>, memref<16x16xbf16>, memref<4x4xbf16>)
  func.func @core_0_0(%arg0: memref<64x64xbf16>, %arg1: memref<16x16xbf16>, %arg2: memref<4x4xbf16>) attributes {df.kernel, itypes = "___", otypes = "", stypes = "___", tag = "core_()"} {
    call @conv(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<16x16xbf16>, memref<4x4xbf16>) -> ()
    return
  }
  func.func @core_0_1(%arg0: memref<64x64xbf16>, %arg1: memref<16x16xbf16>, %arg2: memref<4x4xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @conv(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<16x16xbf16>, memref<4x4xbf16>) -> ()
    return
  }
  func.func @core_0_2(%arg0: memref<64x64xbf16>, %arg1: memref<16x16xbf16>, %arg2: memref<4x4xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @conv(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<16x16xbf16>, memref<4x4xbf16>) -> ()
    return
  }
  func.func @core_0_3(%arg0: memref<64x64xbf16>, %arg1: memref<16x16xbf16>, %arg2: memref<4x4xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @conv(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<16x16xbf16>, memref<4x4xbf16>) -> ()
    return
  }
  func.func @core_1_0(%arg0: memref<64x64xbf16>, %arg1: memref<16x16xbf16>, %arg2: memref<4x4xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @conv(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<16x16xbf16>, memref<4x4xbf16>) -> ()
    return
  }
  func.func @core_1_1(%arg0: memref<64x64xbf16>, %arg1: memref<16x16xbf16>, %arg2: memref<4x4xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @conv(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<16x16xbf16>, memref<4x4xbf16>) -> ()
    return
  }
  func.func @core_1_2(%arg0: memref<64x64xbf16>, %arg1: memref<16x16xbf16>, %arg2: memref<4x4xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @conv(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<16x16xbf16>, memref<4x4xbf16>) -> ()
    return
  }
  func.func @core_1_3(%arg0: memref<64x64xbf16>, %arg1: memref<16x16xbf16>, %arg2: memref<4x4xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @conv(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<16x16xbf16>, memref<4x4xbf16>) -> ()
    return
  }
  func.func @core_2_0(%arg0: memref<64x64xbf16>, %arg1: memref<16x16xbf16>, %arg2: memref<4x4xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @conv(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<16x16xbf16>, memref<4x4xbf16>) -> ()
    return
  }
  func.func @core_2_1(%arg0: memref<64x64xbf16>, %arg1: memref<16x16xbf16>, %arg2: memref<4x4xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @conv(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<16x16xbf16>, memref<4x4xbf16>) -> ()
    return
  }
  func.func @core_2_2(%arg0: memref<64x64xbf16>, %arg1: memref<16x16xbf16>, %arg2: memref<4x4xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @conv(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<16x16xbf16>, memref<4x4xbf16>) -> ()
    return
  }
  func.func @core_2_3(%arg0: memref<64x64xbf16>, %arg1: memref<16x16xbf16>, %arg2: memref<4x4xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @conv(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<16x16xbf16>, memref<4x4xbf16>) -> ()
    return
  }
  func.func @core_3_0(%arg0: memref<64x64xbf16>, %arg1: memref<16x16xbf16>, %arg2: memref<4x4xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @conv(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<16x16xbf16>, memref<4x4xbf16>) -> ()
    return
  }
  func.func @core_3_1(%arg0: memref<64x64xbf16>, %arg1: memref<16x16xbf16>, %arg2: memref<4x4xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @conv(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<16x16xbf16>, memref<4x4xbf16>) -> ()
    return
  }
  func.func @core_3_2(%arg0: memref<64x64xbf16>, %arg1: memref<16x16xbf16>, %arg2: memref<4x4xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @conv(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<16x16xbf16>, memref<4x4xbf16>) -> ()
    return
  }
  func.func @core_3_3(%arg0: memref<64x64xbf16>, %arg1: memref<16x16xbf16>, %arg2: memref<4x4xbf16>) attributes {df.kernel, input_depth = [0, 0, 0], output_depth = []} {
    call @conv(%arg0, %arg1, %arg2) : (memref<64x64xbf16>, memref<16x16xbf16>, memref<4x4xbf16>) -> ()
    return
  }
  func.func @conv_kernel(%arg0: memref<256x256xbf16>, %arg1: memref<16x16xbf16>, %arg2: memref<16x16xbf16>) attributes {dataflow, itypes = "___"} {
    return
  }
}
