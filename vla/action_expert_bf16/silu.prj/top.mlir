module {
  aie.device(npu1_4col) {
    func.func private @silu_256_bf16(memref<16x256xbf16>, memref<16x256xbf16>)
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
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_2 = aie.tile(3, 2)
    %tile_3_3 = aie.tile(3, 3)
    aie.objectfifo @fifo_0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16x256xbf16>> 
    aie.objectfifo @fifo_1(%mem_tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<16x256xbf16>> 
    aie.objectfifo @fifo_2(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x2x16x256xbf16>> 
    aie.objectfifo @fifo_3(%mem_tile_1_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<16x256xbf16>> 
    aie.objectfifo @fifo_4(%mem_tile_1_1, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<16x256xbf16>> 
    aie.objectfifo @fifo_5(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x2x16x256xbf16>> 
    aie.objectfifo @fifo_6(%mem_tile_2_1, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<16x256xbf16>> 
    aie.objectfifo @fifo_7(%mem_tile_2_1, {%tile_2_3}, 2 : i32) : !aie.objectfifo<memref<16x256xbf16>> 
    aie.objectfifo @fifo_8(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1x2x16x256xbf16>> 
    aie.objectfifo @fifo_9(%mem_tile_3_1, {%tile_3_2}, 2 : i32) : !aie.objectfifo<memref<16x256xbf16>> 
    aie.objectfifo @fifo_10(%mem_tile_3_1, {%tile_3_3}, 2 : i32) : !aie.objectfifo<memref<16x256xbf16>> 
    aie.objectfifo @fifo_11(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<1x2x16x256xbf16>> 
    aie.objectfifo @fifo_12(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16x256xbf16>> 
    aie.objectfifo @fifo_13(%tile_0_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16x256xbf16>> 
    aie.objectfifo @fifo_14(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x2x16x256xbf16>> 
    aie.objectfifo @fifo_15(%tile_1_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<16x256xbf16>> 
    aie.objectfifo @fifo_16(%tile_1_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<16x256xbf16>> 
    aie.objectfifo @fifo_17(%mem_tile_1_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x2x16x256xbf16>> 
    aie.objectfifo @fifo_18(%tile_2_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<16x256xbf16>> 
    aie.objectfifo @fifo_19(%tile_2_3, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<16x256xbf16>> 
    aie.objectfifo @fifo_20(%mem_tile_2_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1x2x16x256xbf16>> 
    aie.objectfifo @fifo_21(%tile_3_2, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<16x256xbf16>> 
    aie.objectfifo @fifo_22(%tile_3_3, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<16x256xbf16>> 
    aie.objectfifo @fifo_23(%mem_tile_3_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1x2x16x256xbf16>> 
    aie.objectfifo.link [@fifo_2] -> [@fifo_0, @fifo_1]([] [0, 4096])
    aie.objectfifo.link [@fifo_12, @fifo_13] -> [@fifo_14]([0, 4096] [])
    aie.objectfifo.link [@fifo_5] -> [@fifo_3, @fifo_4]([] [0, 4096])
    aie.objectfifo.link [@fifo_15, @fifo_16] -> [@fifo_17]([0, 4096] [])
    aie.objectfifo.link [@fifo_8] -> [@fifo_6, @fifo_7]([] [0, 4096])
    aie.objectfifo.link [@fifo_18, @fifo_19] -> [@fifo_20]([0, 4096] [])
    aie.objectfifo.link [@fifo_11] -> [@fifo_9, @fifo_10]([] [0, 4096])
    aie.objectfifo.link [@fifo_21, @fifo_22] -> [@fifo_23]([0, 4096] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<16x256xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x256xbf16>> -> memref<16x256xbf16>
        %2 = aie.objectfifo.acquire @fifo_12(Produce, 1) : !aie.objectfifosubview<memref<16x256xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x256xbf16>> -> memref<16x256xbf16>
        func.call @silu_256_bf16(%1, %3) : (memref<16x256xbf16>, memref<16x256xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_12(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<16x256xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x256xbf16>> -> memref<16x256xbf16>
        %2 = aie.objectfifo.acquire @fifo_13(Produce, 1) : !aie.objectfifosubview<memref<16x256xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x256xbf16>> -> memref<16x256xbf16>
        func.call @silu_256_bf16(%1, %3) : (memref<16x256xbf16>, memref<16x256xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        aie.objectfifo.release @fifo_13(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<16x256xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x256xbf16>> -> memref<16x256xbf16>
        %2 = aie.objectfifo.acquire @fifo_15(Produce, 1) : !aie.objectfifosubview<memref<16x256xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x256xbf16>> -> memref<16x256xbf16>
        func.call @silu_256_bf16(%1, %3) : (memref<16x256xbf16>, memref<16x256xbf16>) -> ()
        aie.objectfifo.release @fifo_3(Consume, 1)
        aie.objectfifo.release @fifo_15(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<16x256xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x256xbf16>> -> memref<16x256xbf16>
        %2 = aie.objectfifo.acquire @fifo_16(Produce, 1) : !aie.objectfifosubview<memref<16x256xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x256xbf16>> -> memref<16x256xbf16>
        func.call @silu_256_bf16(%1, %3) : (memref<16x256xbf16>, memref<16x256xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        aie.objectfifo.release @fifo_16(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<16x256xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x256xbf16>> -> memref<16x256xbf16>
        %2 = aie.objectfifo.acquire @fifo_18(Produce, 1) : !aie.objectfifosubview<memref<16x256xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x256xbf16>> -> memref<16x256xbf16>
        func.call @silu_256_bf16(%1, %3) : (memref<16x256xbf16>, memref<16x256xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        aie.objectfifo.release @fifo_18(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_7(Consume, 1) : !aie.objectfifosubview<memref<16x256xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x256xbf16>> -> memref<16x256xbf16>
        %2 = aie.objectfifo.acquire @fifo_19(Produce, 1) : !aie.objectfifosubview<memref<16x256xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x256xbf16>> -> memref<16x256xbf16>
        func.call @silu_256_bf16(%1, %3) : (memref<16x256xbf16>, memref<16x256xbf16>) -> ()
        aie.objectfifo.release @fifo_7(Consume, 1)
        aie.objectfifo.release @fifo_19(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_9(Consume, 1) : !aie.objectfifosubview<memref<16x256xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x256xbf16>> -> memref<16x256xbf16>
        %2 = aie.objectfifo.acquire @fifo_21(Produce, 1) : !aie.objectfifosubview<memref<16x256xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x256xbf16>> -> memref<16x256xbf16>
        func.call @silu_256_bf16(%1, %3) : (memref<16x256xbf16>, memref<16x256xbf16>) -> ()
        aie.objectfifo.release @fifo_9(Consume, 1)
        aie.objectfifo.release @fifo_21(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<16x256xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x256xbf16>> -> memref<16x256xbf16>
        %2 = aie.objectfifo.acquire @fifo_22(Produce, 1) : !aie.objectfifosubview<memref<16x256xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x256xbf16>> -> memref<16x256xbf16>
        func.call @silu_256_bf16(%1, %3) : (memref<16x256xbf16>, memref<16x256xbf16>) -> ()
        aie.objectfifo.release @fifo_10(Consume, 1)
        aie.objectfifo.release @fifo_22(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    aiex.runtime_sequence(%arg0: memref<32768xbf16>, %arg1: memref<32768xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 2, 16, 256][32768, 256, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_2} : memref<32768xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 2, 0, 0][1, 2, 16, 256][32768, 256, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_5} : memref<32768xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 4, 0, 0][1, 2, 16, 256][32768, 256, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<32768xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 6, 0, 0][1, 2, 16, 256][32768, 256, 2048, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<32768xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 2, 16, 256][32768, 256, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_14} : memref<32768xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 2, 0, 0][1, 2, 16, 256][32768, 256, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_17} : memref<32768xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 4, 0, 0][1, 2, 16, 256][32768, 256, 2048, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_20} : memref<32768xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 6, 0, 0][1, 2, 16, 256][32768, 256, 2048, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_23} : memref<32768xbf16>
      aiex.npu.dma_wait {symbol = @fifo_14}
      aiex.npu.dma_wait {symbol = @fifo_17}
      aiex.npu.dma_wait {symbol = @fifo_20}
      aiex.npu.dma_wait {symbol = @fifo_23}
      aie.end
    }
  }
}
