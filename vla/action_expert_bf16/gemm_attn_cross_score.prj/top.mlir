module {
  aie.device(npu1_4col) {
    func.func private @fill_zeros_bf16_32_32_vector(memref<32x32xbf16>)
    func.func private @matmul_scalar_bf16_bf16(memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>)
    func.func private @add_bf16_vector(memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>)
    func.func private @matmul_bf16_bf16(memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>)
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
    aie.objectfifo @fifo_0(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%tile_0_5, %tile_0_3, %tile_0_2, %tile_0_4}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_1(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo @fifo_2(%mem_tile_1_1 dimensionsToStream [<size = 4, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_3(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo @fifo_4(%mem_tile_2_1 dimensionsToStream [<size = 4, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>], {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_5(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo @fifo_6(%mem_tile_3_1 dimensionsToStream [<size = 4, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>], {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_7(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo @fifo_8(%mem_tile_0_1 dimensionsToStream [<size = 4, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>], {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_9(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo @fifo_10(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_11(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo @fifo_12(%tile_0_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_13(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo @fifo_14(%tile_0_4, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_15(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo @fifo_16(%tile_0_5, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_17(%mem_tile_3_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo.link [@fifo_1] -> [@fifo_0]([] [])
    aie.objectfifo.link [@fifo_9] -> [@fifo_8]([] [])
    aie.objectfifo.link [@fifo_10] -> [@fifo_11]([] [])
    aie.objectfifo.link [@fifo_3] -> [@fifo_2]([] [])
    aie.objectfifo.link [@fifo_12] -> [@fifo_13]([] [])
    aie.objectfifo.link [@fifo_5] -> [@fifo_4]([] [])
    aie.objectfifo.link [@fifo_14] -> [@fifo_15]([] [])
    aie.objectfifo.link [@fifo_7] -> [@fifo_6]([] [])
    aie.objectfifo.link [@fifo_16] -> [@fifo_17]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_10(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        aie.objectfifo.release @fifo_2(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_10(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_12(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        aie.objectfifo.release @fifo_4(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_4(Consume, 1)
        aie.objectfifo.release @fifo_12(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_14(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        aie.objectfifo.release @fifo_6(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_6(Consume, 1)
        aie.objectfifo.release @fifo_14(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_16(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_8(Consume, 1)
        aie.objectfifo.release @fifo_16(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    aiex.runtime_sequence(%arg0: memref<2048xbf16>, %arg1: memref<4096xbf16>, %arg2: memref<8192xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 2, 32, 32][2048, 32, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_1} : memref<2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 2, 32, 32][32, 4096, 128, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_3} : memref<8192xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[1, 0, 0, 0][1, 2, 32, 32][32, 4096, 128, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_5} : memref<8192xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[2, 0, 0, 0][1, 2, 32, 32][32, 4096, 128, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_7} : memref<8192xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[3, 0, 0, 0][1, 2, 32, 32][32, 4096, 128, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_9} : memref<8192xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 32, 32][4096, 32, 128, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_11} : memref<4096xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 1, 0, 0][1, 1, 32, 32][4096, 32, 128, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_13} : memref<4096xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 2, 0, 0][1, 1, 32, 32][4096, 32, 128, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_15} : memref<4096xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 3, 0, 0][1, 1, 32, 32][4096, 32, 128, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_17} : memref<4096xbf16>
      aiex.npu.dma_wait {symbol = @fifo_11}
      aiex.npu.dma_wait {symbol = @fifo_13}
      aiex.npu.dma_wait {symbol = @fifo_15}
      aiex.npu.dma_wait {symbol = @fifo_17}
      aie.end
    }
  }
}
