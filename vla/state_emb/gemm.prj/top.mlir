module {
  aie.device(npu1_4col) {
    func.func private @fill_zeros_bf16_16_64_vector(memref<16x64xbf16>)
    func.func private @matmul_scalar_bf16_bf16(memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>)
    func.func private @add_bf16_vector(memref<16x64xbf16>, memref<16x64xbf16>, memref<16x64xbf16>)
    func.func private @matmul_bf16_bf16(memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>)
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
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_5 = aie.tile(1, 5)
    aie.objectfifo @fifo_0(%mem_tile_0_1 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%tile_0_2, %tile_0_3, %tile_0_4}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>> 
    aie.objectfifo @fifo_1(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x16x32xbf16>> 
    aie.objectfifo @fifo_2(%mem_tile_1_1 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%tile_1_5, %tile_1_3, %tile_1_2, %tile_1_4}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>> 
    aie.objectfifo @fifo_3(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x1x16x32xbf16>> 
    aie.objectfifo @fifo_4(%mem_tile_2_1 dimensionsToStream [<size = 4, stride = 512>, <size = 16, stride = 4>, <size = 8, stride = 64>, <size = 4, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @fifo_5(%mem_tile_2_1 dimensionsToStream [<size = 4, stride = 512>, <size = 16, stride = 4>, <size = 8, stride = 64>, <size = 4, stride = 1>], {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @fifo_6(%mem_tile_2_1 dimensionsToStream [<size = 4, stride = 512>, <size = 16, stride = 4>, <size = 8, stride = 64>, <size = 4, stride = 1>], {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @fifo_7(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1x3x32x64xbf16>> 
    aie.objectfifo @fifo_8(%mem_tile_3_1 dimensionsToStream [<size = 4, stride = 512>, <size = 16, stride = 4>, <size = 8, stride = 64>, <size = 4, stride = 1>], {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @fifo_9(%mem_tile_3_1 dimensionsToStream [<size = 4, stride = 512>, <size = 16, stride = 4>, <size = 8, stride = 64>, <size = 4, stride = 1>], {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @fifo_10(%mem_tile_3_1 dimensionsToStream [<size = 4, stride = 512>, <size = 16, stride = 4>, <size = 8, stride = 64>, <size = 4, stride = 1>], {%tile_1_4}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @fifo_11(%mem_tile_3_1 dimensionsToStream [<size = 4, stride = 512>, <size = 16, stride = 4>, <size = 8, stride = 64>, <size = 4, stride = 1>], {%tile_1_5}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @fifo_12(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<1x4x32x64xbf16>> 
    aie.objectfifo @fifo_13(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>> 
    aie.objectfifo @fifo_14(%tile_0_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>> 
    aie.objectfifo @fifo_15(%tile_0_4, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>> 
    aie.objectfifo @fifo_16(%mem_tile_0_1 dimensionsToStream [<size = 4, stride = 256>, <size = 4, stride = 4>, <size = 16, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x3x16x64xbf16>> 
    aie.objectfifo @fifo_17(%tile_1_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>> 
    aie.objectfifo @fifo_18(%tile_1_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>> 
    aie.objectfifo @fifo_19(%tile_1_4, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>> 
    aie.objectfifo @fifo_20(%tile_1_5, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>> 
    aie.objectfifo @fifo_21(%mem_tile_1_1 dimensionsToStream [<size = 4, stride = 256>, <size = 4, stride = 4>, <size = 16, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x4x16x64xbf16>> 
    aie.objectfifo.link [@fifo_1] -> [@fifo_0]([] [])
    aie.objectfifo.link [@fifo_13, @fifo_14, @fifo_15] -> [@fifo_16]([0, 1024, 2048] [])
    aie.objectfifo.link [@fifo_3] -> [@fifo_2]([] [])
    aie.objectfifo.link [@fifo_17, @fifo_18, @fifo_19, @fifo_20] -> [@fifo_21]([0, 1024, 2048, 3072] [])
    aie.objectfifo.link [@fifo_7] -> [@fifo_4, @fifo_5, @fifo_6]([] [0, 2048, 4096])
    aie.objectfifo.link [@fifo_12] -> [@fifo_8, @fifo_9, @fifo_10, @fifo_11]([] [0, 2048, 4096, 6144])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_13(Produce, 1) : !aie.objectfifosubview<memref<16x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x64xbf16>> -> memref<16x64xbf16>
        func.call @fill_zeros_bf16_16_64_vector(%1) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_4(Consume, 1)
        aie.objectfifo.release @fifo_13(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_14(Produce, 1) : !aie.objectfifosubview<memref<16x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x64xbf16>> -> memref<16x64xbf16>
        func.call @fill_zeros_bf16_16_64_vector(%1) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_5(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_5(Consume, 1)
        aie.objectfifo.release @fifo_14(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_15(Produce, 1) : !aie.objectfifosubview<memref<16x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x64xbf16>> -> memref<16x64xbf16>
        func.call @fill_zeros_bf16_16_64_vector(%1) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_6(Consume, 1)
        aie.objectfifo.release @fifo_15(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_17(Produce, 1) : !aie.objectfifosubview<memref<16x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x64xbf16>> -> memref<16x64xbf16>
        func.call @fill_zeros_bf16_16_64_vector(%1) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_8(Consume, 1)
        aie.objectfifo.release @fifo_17(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_18(Produce, 1) : !aie.objectfifosubview<memref<16x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x64xbf16>> -> memref<16x64xbf16>
        func.call @fill_zeros_bf16_16_64_vector(%1) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_9(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_9(Consume, 1)
        aie.objectfifo.release @fifo_18(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_19(Produce, 1) : !aie.objectfifosubview<memref<16x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x64xbf16>> -> memref<16x64xbf16>
        func.call @fill_zeros_bf16_16_64_vector(%1) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_10(Consume, 1)
        aie.objectfifo.release @fifo_19(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_20(Produce, 1) : !aie.objectfifosubview<memref<16x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x64xbf16>> -> memref<16x64xbf16>
        func.call @fill_zeros_bf16_16_64_vector(%1) {lib = "fill_zeros_bf16_16_64_vector"} : (memref<16x64xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_11(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<16x32xbf16>, memref<32x64xbf16>, memref<16x64xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_11(Consume, 1)
        aie.objectfifo.release @fifo_20(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    aiex.runtime_sequence(%arg0: memref<512xbf16>, %arg1: memref<15360xbf16>, %arg2: memref<30720xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 16, 32][512, 32, 32, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_1} : memref<512xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 12, 0, 0][1, 3, 32, 64][30720, 64, 960, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_7} : memref<30720xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 12, 0, 0][1, 3, 16, 64][15360, 64, 960, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_16} : memref<15360xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 16, 32][512, 32, 32, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_3} : memref<512xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 16, 32][512, 32, 32, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_3} : memref<512xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 16, 32][512, 32, 32, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_3} : memref<512xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 4, 32, 64][30720, 64, 960, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_12} : memref<30720xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 4, 0, 0][1, 4, 32, 64][30720, 64, 960, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_12} : memref<30720xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 8, 0, 0][1, 4, 32, 64][30720, 64, 960, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_12} : memref<30720xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 4, 16, 64][15360, 64, 960, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_21} : memref<15360xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 4, 0, 0][1, 4, 16, 64][15360, 64, 960, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_21} : memref<15360xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 8, 0, 0][1, 4, 16, 64][15360, 64, 960, 1]) {id = 4 : i64, issue_token = true, metadata = @fifo_21} : memref<15360xbf16>
      aiex.npu.dma_wait {symbol = @fifo_16}
      aiex.npu.dma_wait {symbol = @fifo_21}
      aiex.npu.dma_wait {symbol = @fifo_21}
      aiex.npu.dma_wait {symbol = @fifo_21}
      aie.end
    }
  }
}
