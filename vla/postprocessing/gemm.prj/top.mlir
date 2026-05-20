module {
  aie.device(npu1_4col) {
    func.func private @fill_zeros_bf16_32_32_vector(memref<32x32xbf16>)
    func.func private @matmul_scalar_bf16_bf16(memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>)
    func.func private @add_bf16_vector(memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>)
    func.func private @matmul_bf16_bf16(memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @fifo_0(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @fifo_1(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x64xbf16>> 
    aie.objectfifo @fifo_2(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x32xbf16>> 
    aie.objectfifo @fifo_3(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x1x64x32xbf16>> 
    aie.objectfifo @fifo_4(%tile_0_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_5(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo.link [@fifo_1] -> [@fifo_0]([] [])
    aie.objectfifo.link [@fifo_3] -> [@fifo_2]([] [])
    aie.objectfifo.link [@fifo_4] -> [@fifo_5]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_4(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        %4 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        aie.objectfifo.release @fifo_2(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %10 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        aie.objectfifo.release @fifo_2(Consume, 1)
        %12 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
        func.call @matmul_bf16_bf16(%11, %13, %1) : (memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %14 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        aie.objectfifo.release @fifo_2(Consume, 1)
        %16 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
        func.call @matmul_bf16_bf16(%15, %17, %1) : (memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %18 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        aie.objectfifo.release @fifo_2(Consume, 1)
        %20 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
        %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
        func.call @matmul_bf16_bf16(%19, %21, %1) : (memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %22 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %23 = aie.objectfifo.subview.access %22[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        aie.objectfifo.release @fifo_2(Consume, 1)
        %24 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
        %25 = aie.objectfifo.subview.access %24[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
        func.call @matmul_bf16_bf16(%23, %25, %1) : (memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %26 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %27 = aie.objectfifo.subview.access %26[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        aie.objectfifo.release @fifo_2(Consume, 1)
        %28 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
        %29 = aie.objectfifo.subview.access %28[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
        func.call @matmul_bf16_bf16(%27, %29, %1) : (memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %30 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %31 = aie.objectfifo.subview.access %30[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        aie.objectfifo.release @fifo_2(Consume, 1)
        %32 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
        %33 = aie.objectfifo.subview.access %32[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
        func.call @matmul_bf16_bf16(%31, %33, %1) : (memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %34 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %35 = aie.objectfifo.subview.access %34[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        aie.objectfifo.release @fifo_2(Consume, 1)
        %36 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
        %37 = aie.objectfifo.subview.access %36[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
        func.call @matmul_bf16_bf16(%35, %37, %1) : (memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %38 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %39 = aie.objectfifo.subview.access %38[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        aie.objectfifo.release @fifo_2(Consume, 1)
        %40 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
        %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
        func.call @matmul_bf16_bf16(%39, %41, %1) : (memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %42 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        aie.objectfifo.release @fifo_2(Consume, 1)
        %44 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
        %45 = aie.objectfifo.subview.access %44[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
        func.call @matmul_bf16_bf16(%43, %45, %1) : (memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %46 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        aie.objectfifo.release @fifo_2(Consume, 1)
        %48 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
        %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
        func.call @matmul_bf16_bf16(%47, %49, %1) : (memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_4(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    aiex.runtime_sequence(%arg0: memref<24576xbf16>, %arg1: memref<1024xbf16>, %arg2: memref<24576xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 12, 32, 64][24576, 64, 768, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_1} : memref<24576xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 12, 64, 32][32, 2048, 32, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_3} : memref<24576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 32, 32][1024, 32, 32, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_5} : memref<1024xbf16>
      aiex.npu.dma_wait {symbol = @fifo_5}
      aie.end
    }
  }
}
