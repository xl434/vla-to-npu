module {
  func.func @core_0(%arg0: memref<48x128xbf16>, %arg1: memref<8x768xbf16>) attributes {df.kernel, itypes = "__", otypes = "", stypes = "__", tag = "core_()"} {
    affine.for %arg2 = 0 to 8 {
      affine.for %arg3 = 0 to 3 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            %0 = affine.load %arg0[%arg3 * 16 + %arg4, %arg2 * 16 + %arg5] {from = "local_img"} : memref<48x128xbf16>
            affine.store %0, %arg1[%arg2, %arg3 * 256 + %arg4 * 16 + %arg5] {to = "local_pat"} : memref<8x768xbf16>
          } {loop_name = "kx"}
        } {loop_name = "ky"}
      } {loop_name = "c"}
    } {loop_name = "pw", op_name = "S_pw_c_ky_kx_0"}
    return
  }
  func.func @im2col_region(%arg0: memref<48x128xbf16>, %arg1: memref<8x768xbf16>) attributes {dataflow, itypes = "__"} {
    return
  }
}
