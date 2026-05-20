module {
  aie.device(npu1_4col) {
    func.func private @sin_cos_float32(memref<32x64xf32>, memref<64x64xf32>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @fifo_0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_1(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x64xf32>> 
    aie.objectfifo @fifo_2(%tile_0_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x64xf32>> 
    aie.objectfifo @fifo_3(%mem_tile_1_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1x1x64x64xf32>> 
    aie.objectfifo.link [@fifo_1] -> [@fifo_0]([] [])
    aie.objectfifo.link [@fifo_2] -> [@fifo_3]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %2 = aie.objectfifo.acquire @fifo_2(Produce, 1) : !aie.objectfifosubview<memref<64x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xf32>> -> memref<64x64xf32>
        func.call @sin_cos_float32(%1, %3) : (memref<32x64xf32>, memref<64x64xf32>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_2(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    aiex.runtime_sequence(%arg0: memref<2048xf32>, %arg1: memref<4096xf32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 32, 64][2048, 64, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_1} : memref<2048xf32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 64, 64][4096, 64, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_3} : memref<4096xf32>
      aiex.npu.dma_wait {symbol = @fifo_3}
      aie.end
    }
  }
}
