module {
  aie.device(npu1_4col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @fifo_0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<8x768xbf16>> 
    aie.objectfifo @fifo_1(%mem_tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<8x768xbf16>> 
    aie.objectfifo @fifo_2(%mem_tile_0_1, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<8x768xbf16>> 
    aie.objectfifo @fifo_3(%mem_tile_0_1, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<8x768xbf16>> 
    aie.objectfifo @fifo_4(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x4x8x768xbf16>> 
    aie.objectfifo @fifo_5(%tile_0_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<8x768xbf16>> 
    aie.objectfifo @fifo_6(%tile_0_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<8x768xbf16>> 
    aie.objectfifo @fifo_7(%tile_0_4, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<8x768xbf16>> 
    aie.objectfifo @fifo_8(%tile_0_5, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<8x768xbf16>> 
    aie.objectfifo @fifo_9(%mem_tile_1_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1x4x8x768xbf16>> 
    aie.objectfifo.link [@fifo_4] -> [@fifo_0, @fifo_1, @fifo_2, @fifo_3]([] [0, 6144, 12288, 18432])
    aie.objectfifo.link [@fifo_5, @fifo_6, @fifo_7, @fifo_8] -> [@fifo_9]([0, 6144, 12288, 18432] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<8x768xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x768xbf16>> -> memref<8x768xbf16>
        %2 = aie.objectfifo.acquire @fifo_5(Produce, 1) : !aie.objectfifosubview<memref<8x768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x768xbf16>> -> memref<8x768xbf16>
        memref.copy %1, %3 {to = "local_Out"} : memref<8x768xbf16> to memref<8x768xbf16>
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_5(Produce, 1)
      }
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<8x768xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x768xbf16>> -> memref<8x768xbf16>
        %2 = aie.objectfifo.acquire @fifo_6(Produce, 1) : !aie.objectfifosubview<memref<8x768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x768xbf16>> -> memref<8x768xbf16>
        memref.copy %1, %3 {to = "local_Out"} : memref<8x768xbf16> to memref<8x768xbf16>
        aie.objectfifo.release @fifo_1(Consume, 1)
        aie.objectfifo.release @fifo_6(Produce, 1)
      }
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<8x768xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x768xbf16>> -> memref<8x768xbf16>
        %2 = aie.objectfifo.acquire @fifo_7(Produce, 1) : !aie.objectfifosubview<memref<8x768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x768xbf16>> -> memref<8x768xbf16>
        memref.copy %1, %3 {to = "local_Out"} : memref<8x768xbf16> to memref<8x768xbf16>
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_7(Produce, 1)
      }
      aie.end
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<8x768xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x768xbf16>> -> memref<8x768xbf16>
        %2 = aie.objectfifo.acquire @fifo_8(Produce, 1) : !aie.objectfifosubview<memref<8x768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8x768xbf16>> -> memref<8x768xbf16>
        memref.copy %1, %3 {to = "local_Out"} : memref<8x768xbf16> to memref<8x768xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        aie.objectfifo.release @fifo_8(Produce, 1)
      }
      aie.end
    }
    aiex.runtime_sequence(%arg0: memref<24576xbf16>, %arg1: memref<24576xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 4, 8, 768][0, 6144, 768, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_4} : memref<24576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 4, 8, 768][0, 768, 3072, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_9} : memref<24576xbf16>
      aiex.npu.dma_wait {symbol = @fifo_9}
      aie.end
    }
  }
}
