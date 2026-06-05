module {
  aie.device(npu1_4col) {
    func.func private @fill_zeros_bf16_32_32_vector(memref<32x32xbf16>)
    func.func private @matmul_scalar_bf16_bf16(memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>)
    func.func private @add_bf16_vector(memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>)
    func.func private @matmul_bf16_bf16(memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    aie.objectfifo @fifo_0(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 768>, <size = 24, stride = 8>, <size = 4, stride = 192>, <size = 8, stride = 1>], {%tile_0_2, %tile_0_3}, 2 : i32) : !aie.objectfifo<memref<32x192xbf16>> 
    aie.objectfifo @fifo_1(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x192xbf16>> 
    aie.objectfifo @fifo_2(%mem_tile_1_1 dimensionsToStream [<size = 24, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<192x32xbf16>> 
    aie.objectfifo @fifo_3(%mem_tile_1_1 dimensionsToStream [<size = 24, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>], {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<192x32xbf16>> 
    aie.objectfifo @fifo_4(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x2x192x32xbf16>> 
    aie.objectfifo @fifo_5(%tile_0_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_6(%tile_0_3, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_7(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<1x2x32x32xbf16>> 
    aie.objectfifo.link [@fifo_1] -> [@fifo_0]([] [])
    aie.objectfifo.link [@fifo_4] -> [@fifo_2, @fifo_3]([] [0, 6144])
    aie.objectfifo.link [@fifo_5, @fifo_6] -> [@fifo_7]([0, 1024] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_5(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x192xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x192xbf16>> -> memref<32x192xbf16>
        %4 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<192x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<192x32xbf16>> -> memref<192x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x192xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x192xbf16>> -> memref<32x192xbf16>
        aie.objectfifo.release @fifo_2(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<192x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<192x32xbf16>> -> memref<192x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x192xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x192xbf16>> -> memref<32x192xbf16>
        aie.objectfifo.release @fifo_2(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<192x32xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<192x32xbf16>> -> memref<192x32xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x192xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x192xbf16>> -> memref<32x192xbf16>
        aie.objectfifo.release @fifo_2(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<192x32xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<192x32xbf16>> -> memref<192x32xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_5(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_6(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x192xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x192xbf16>> -> memref<32x192xbf16>
        %4 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<192x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<192x32xbf16>> -> memref<192x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x192xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x192xbf16>> -> memref<32x192xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<192x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<192x32xbf16>> -> memref<192x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x192xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x192xbf16>> -> memref<32x192xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<192x32xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<192x32xbf16>> -> memref<192x32xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x192xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x192xbf16>> -> memref<32x192xbf16>
        aie.objectfifo.release @fifo_3(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<192x32xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<192x32xbf16>> -> memref<192x32xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_3(Consume, 1)
        aie.objectfifo.release @fifo_6(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    aiex.runtime_sequence(%arg0: memref<24576xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<49152xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 4, 32, 192][24576, 192, 768, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_1} : memref<24576xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][4, 2, 192, 32][12288, 32, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_4} : memref<49152xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 2, 32, 32][2048, 32, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_7} : memref<2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_7}
      aie.end
    }
  }
}
