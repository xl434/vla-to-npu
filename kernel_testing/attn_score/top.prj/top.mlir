module {
  aie.device(npu1_4col) {
    func.func private @transpose_matmul_with_scale_bf16(memref<32x64xbf16>, memref<32x64xbf16>, memref<32x32xbf16>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @fifo_0(%mem_tile_0_1, {%tile_0_2, %tile_0_3}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @fifo_1(%mem_tile_0_1, {%tile_0_5, %tile_0_4}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @fifo_2(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x2x32x64xbf16>> 
    aie.objectfifo @fifo_3(%mem_tile_1_1, {%tile_0_2, %tile_0_4}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @fifo_4(%mem_tile_1_1, {%tile_0_3, %tile_0_5}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @fifo_5(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x2x32x64xbf16>> 
    aie.objectfifo @fifo_6(%tile_0_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_7(%tile_0_3, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_8(%tile_0_4, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_9(%tile_0_5, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_10(%mem_tile_2_1, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<2x2x32x32xbf16>> 
    aie.objectfifo.link [@fifo_2] -> [@fifo_0, @fifo_1]([] [0, 2048])
    aie.objectfifo.link [@fifo_5] -> [@fifo_3, @fifo_4]([] [0, 2048])
    aie.objectfifo.link [@fifo_6, @fifo_7, @fifo_8, @fifo_9] -> [@fifo_10]([0, 1024, 2048, 3072] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        %2 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_6(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @transpose_matmul_with_scale_bf16(%1, %3, %5) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_3(Consume, 1)
        aie.objectfifo.release @fifo_6(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        %2 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_7(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @transpose_matmul_with_scale_bf16(%1, %3, %5) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_4(Consume, 1)
        aie.objectfifo.release @fifo_7(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        %2 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_8(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @transpose_matmul_with_scale_bf16(%1, %3, %5) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        aie.objectfifo.release @fifo_3(Consume, 1)
        aie.objectfifo.release @fifo_8(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        %2 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_9(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @transpose_matmul_with_scale_bf16(%1, %3, %5) : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        aie.objectfifo.release @fifo_4(Consume, 1)
        aie.objectfifo.release @fifo_9(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    aiex.runtime_sequence(%arg0: memref<4096xbf16>, %arg1: memref<4096xbf16>, %arg2: memref<4096xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 2, 32, 64][0, 2048, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_2} : memref<4096xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 2, 32, 64][0, 2048, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_5} : memref<4096xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][2, 2, 32, 32][2048, 32, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_10} : memref<4096xbf16>
      aiex.npu.dma_wait {symbol = @fifo_10}
      aie.end
    }
  }
}
