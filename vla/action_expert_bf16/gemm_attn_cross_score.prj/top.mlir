module {
  aie.device(npu1_4col) {
    func.func private @fill_zeros_bf16_32_128_vector(memref<32x128xbf16>)
    func.func private @matmul_scalar_bf16_bf16(memref<32x64xbf16>, memref<64x128xbf16>, memref<32x128xbf16>)
    func.func private @add_bf16_vector(memref<32x128xbf16>, memref<32x128xbf16>, memref<32x128xbf16>)
    func.func private @matmul_bf16_bf16(memref<32x64xbf16>, memref<64x128xbf16>, memref<32x128xbf16>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @fifo_0(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @fifo_1(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x64xbf16>> 
    aie.objectfifo @fifo_2(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 1024>, <size = 32, stride = 4>, <size = 8, stride = 128>, <size = 4, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x128xbf16>> 
    aie.objectfifo @fifo_3(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x1x64x128xbf16>> 
    aie.objectfifo @fifo_4(%tile_0_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<32x128xbf16>> 
    aie.objectfifo @fifo_5(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 512>, <size = 4, stride = 4>, <size = 32, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<1x1x32x128xbf16>> 
    aie.objectfifo.link [@fifo_1] -> [@fifo_0]([] [])
    aie.objectfifo.link [@fifo_3] -> [@fifo_2]([] [])
    aie.objectfifo.link [@fifo_4] -> [@fifo_5]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_4(Produce, 1) : !aie.objectfifosubview<memref<32x128xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x128xbf16>> -> memref<32x128xbf16>
        func.call @fill_zeros_bf16_32_128_vector(%1) {lib = "fill_zeros_bf16_32_128_vector"} : (memref<32x128xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x128xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x128xbf16>> -> memref<64x128xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x64xbf16>, memref<64x128xbf16>, memref<32x128xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_4(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    aiex.runtime_sequence(%arg0: memref<2048xbf16>, %arg1: memref<4096xbf16>, %arg2: memref<8192xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 32, 64][2048, 64, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_1} : memref<2048xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 1, 64, 128][8192, 128, 128, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_3} : memref<8192xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 32, 128][4096, 128, 128, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_5} : memref<4096xbf16>
      aiex.npu.dma_wait {symbol = @fifo_5}
      aie.end
    }
  }
}
