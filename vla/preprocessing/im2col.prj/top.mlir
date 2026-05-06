module {
  aie.device(npu1_4col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @fifo_0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<48x128xbf16>> 
    aie.objectfifo @fifo_1(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x48x128xbf16>> 
    aie.objectfifo @fifo_2(%tile_0_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<8x768xbf16>> 
    aie.objectfifo @fifo_3(%mem_tile_1_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1x1x8x768xbf16>> 
    aie.objectfifo.link [@fifo_1] -> [@fifo_0]([] [])
    aie.objectfifo.link [@fifo_2] -> [@fifo_3]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<48x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<48x128xbf16>> -> memref<48x128xbf16>
        %2 = aie.objectfifo.acquire @fifo_2(Produce, 1) : !aie.objectfifosubview<memref<8x768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x768xbf16>> -> memref<8x768xbf16>
        affine.for %arg1 = 0 to 8 {
          affine.for %arg2 = 0 to 3 {
            affine.for %arg3 = 0 to 16 {
              affine.for %arg4 = 0 to 16 {
                %4 = affine.load %1[%arg2 * 16 + %arg3, %arg1 * 16 + %arg4] {from = "local_img"} : memref<48x128xbf16>
                affine.store %4, %3[%arg1, %arg2 * 256 + %arg3 * 16 + %arg4] {to = "local_pat"} : memref<8x768xbf16>
              } {loop_name = "kx"}
            } {loop_name = "ky"}
          } {loop_name = "c"}
        } {loop_name = "pw", op_name = "S_pw_c_ky_kx_0"}
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_2(Produce, 1)
      }
      aie.end
    }
    aiex.runtime_sequence(%arg0: memref<6144xbf16>, %arg1: memref<6144xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 48, 128][0, 0, 128, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_1} : memref<6144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 8, 768][0, 0, 768, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_3} : memref<6144xbf16>
      aiex.npu.dma_wait {symbol = @fifo_3}
      aie.end
    }
  }
}
