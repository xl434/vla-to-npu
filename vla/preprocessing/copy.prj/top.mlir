module {
  aie.device(npu1_4col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_5 = aie.tile(1, 5)
    aie.objectfifo @fifo_0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1x32xbf16>> 
    aie.objectfifo @fifo_1(%mem_tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<1x32xbf16>> 
    aie.objectfifo @fifo_2(%mem_tile_0_1, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<1x32xbf16>> 
    aie.objectfifo @fifo_3(%mem_tile_0_1, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<1x32xbf16>> 
    aie.objectfifo @fifo_4(%mem_tile_0_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<1x32xbf16>> 
    aie.objectfifo @fifo_5(%mem_tile_0_1, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<1x32xbf16>> 
    aie.objectfifo @fifo_6(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x6x1x32xbf16>> 
    aie.objectfifo @fifo_7(%mem_tile_1_1, {%tile_1_4}, 2 : i32) : !aie.objectfifo<memref<1x32xbf16>> 
    aie.objectfifo @fifo_8(%mem_tile_1_1, {%tile_1_5}, 2 : i32) : !aie.objectfifo<memref<1x32xbf16>> 
    aie.objectfifo @fifo_9(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x2x1x32xbf16>> 
    aie.objectfifo @fifo_10(%tile_0_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1x32xbf16>> 
    aie.objectfifo @fifo_11(%tile_0_3, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1x32xbf16>> 
    aie.objectfifo @fifo_12(%tile_0_4, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1x32xbf16>> 
    aie.objectfifo @fifo_13(%tile_0_5, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1x32xbf16>> 
    aie.objectfifo @fifo_14(%tile_1_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1x32xbf16>> 
    aie.objectfifo @fifo_15(%tile_1_3, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1x32xbf16>> 
    aie.objectfifo @fifo_16(%mem_tile_2_1, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<1x6x1x32xbf16>> 
    aie.objectfifo @fifo_17(%tile_1_4, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<1x32xbf16>> 
    aie.objectfifo @fifo_18(%tile_1_5, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<1x32xbf16>> 
    aie.objectfifo @fifo_19(%mem_tile_3_1, {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<1x2x1x32xbf16>> 
    aie.objectfifo.link [@fifo_6] -> [@fifo_0, @fifo_1, @fifo_2, @fifo_3, @fifo_4, @fifo_5]([] [0, 32, 64, 96, 128, 160])
    aie.objectfifo.link [@fifo_9] -> [@fifo_7, @fifo_8]([] [0, 32])
    aie.objectfifo.link [@fifo_10, @fifo_11, @fifo_12, @fifo_13, @fifo_14, @fifo_15] -> [@fifo_16]([0, 32, 64, 96, 128, 160] [])
    aie.objectfifo.link [@fifo_17, @fifo_18] -> [@fifo_19]([0, 32] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<1x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1x32xbf16>> -> memref<1x32xbf16>
        %subview = memref.subview %1[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
        %2 = aie.objectfifo.acquire @fifo_10(Produce, 1) : !aie.objectfifosubview<memref<1x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1x32xbf16>> -> memref<1x32xbf16>
        %subview_0 = memref.subview %3[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
        memref.copy %subview, %subview_0 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_10(Produce, 1)
      }
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<1x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1x32xbf16>> -> memref<1x32xbf16>
        %subview = memref.subview %1[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
        %2 = aie.objectfifo.acquire @fifo_11(Produce, 1) : !aie.objectfifosubview<memref<1x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1x32xbf16>> -> memref<1x32xbf16>
        %subview_0 = memref.subview %3[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
        memref.copy %subview, %subview_0 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
        aie.objectfifo.release @fifo_1(Consume, 1)
        aie.objectfifo.release @fifo_11(Produce, 1)
      }
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<1x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1x32xbf16>> -> memref<1x32xbf16>
        %subview = memref.subview %1[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
        %2 = aie.objectfifo.acquire @fifo_12(Produce, 1) : !aie.objectfifosubview<memref<1x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1x32xbf16>> -> memref<1x32xbf16>
        %subview_0 = memref.subview %3[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
        memref.copy %subview, %subview_0 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_12(Produce, 1)
      }
      aie.end
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<1x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1x32xbf16>> -> memref<1x32xbf16>
        %subview = memref.subview %1[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
        %2 = aie.objectfifo.acquire @fifo_13(Produce, 1) : !aie.objectfifosubview<memref<1x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1x32xbf16>> -> memref<1x32xbf16>
        %subview_0 = memref.subview %3[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
        memref.copy %subview, %subview_0 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
        aie.objectfifo.release @fifo_3(Consume, 1)
        aie.objectfifo.release @fifo_13(Produce, 1)
      }
      aie.end
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<1x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1x32xbf16>> -> memref<1x32xbf16>
        %subview = memref.subview %1[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
        %2 = aie.objectfifo.acquire @fifo_14(Produce, 1) : !aie.objectfifosubview<memref<1x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1x32xbf16>> -> memref<1x32xbf16>
        %subview_0 = memref.subview %3[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
        memref.copy %subview, %subview_0 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
        aie.objectfifo.release @fifo_4(Consume, 1)
        aie.objectfifo.release @fifo_14(Produce, 1)
      }
      aie.end
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_5(Consume, 1) : !aie.objectfifosubview<memref<1x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1x32xbf16>> -> memref<1x32xbf16>
        %subview = memref.subview %1[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
        %2 = aie.objectfifo.acquire @fifo_15(Produce, 1) : !aie.objectfifosubview<memref<1x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1x32xbf16>> -> memref<1x32xbf16>
        %subview_0 = memref.subview %3[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
        memref.copy %subview, %subview_0 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
        aie.objectfifo.release @fifo_5(Consume, 1)
        aie.objectfifo.release @fifo_15(Produce, 1)
      }
      aie.end
    }
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_7(Consume, 1) : !aie.objectfifosubview<memref<1x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1x32xbf16>> -> memref<1x32xbf16>
        %subview = memref.subview %1[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
        %2 = aie.objectfifo.acquire @fifo_17(Produce, 1) : !aie.objectfifosubview<memref<1x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1x32xbf16>> -> memref<1x32xbf16>
        %subview_0 = memref.subview %3[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
        memref.copy %subview, %subview_0 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
        aie.objectfifo.release @fifo_7(Consume, 1)
        aie.objectfifo.release @fifo_17(Produce, 1)
      }
      aie.end
    }
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<1x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1x32xbf16>> -> memref<1x32xbf16>
        %subview = memref.subview %1[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
        %2 = aie.objectfifo.acquire @fifo_18(Produce, 1) : !aie.objectfifosubview<memref<1x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1x32xbf16>> -> memref<1x32xbf16>
        %subview_0 = memref.subview %3[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
        memref.copy %subview, %subview_0 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
        aie.objectfifo.release @fifo_8(Consume, 1)
        aie.objectfifo.release @fifo_18(Produce, 1)
      }
      aie.end
    }
    aiex.runtime_sequence(%arg0: memref<256xbf16>, %arg1: memref<256xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 6, 1, 32][0, 32, 32, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_6} : memref<256xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 6, 0, 0][1, 2, 1, 32][0, 32, 32, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_9} : memref<256xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 6, 1, 32][0, 32, 256, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_16} : memref<256xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 6, 0, 0][1, 2, 1, 32][0, 32, 256, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_19} : memref<256xbf16>
      aiex.npu.dma_wait {symbol = @fifo_16}
      aiex.npu.dma_wait {symbol = @fifo_19}
      aie.end
    }
  }
}
