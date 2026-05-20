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
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)
    %tile_2_4 = aie.tile(2, 4)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_2 = aie.tile(3, 2)
    %tile_3_3 = aie.tile(3, 3)
    %tile_3_4 = aie.tile(3, 4)
    %tile_3_5 = aie.tile(3, 5)
    aie.objectfifo @fifo_0(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%tile_0_5, %tile_0_3, %tile_0_2, %tile_0_4}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_1(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo @fifo_2(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%tile_1_3, %tile_1_2, %tile_1_4, %tile_1_5}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_3(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo @fifo_4(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%tile_2_3, %tile_2_4, %tile_2_5, %tile_2_2}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_5(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo @fifo_6(%mem_tile_3_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%tile_3_4, %tile_3_5, %tile_3_2, %tile_3_3}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_7(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo @fifo_8(%mem_tile_0_1 dimensionsToStream [<size = 4, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>], {%tile_2_2, %tile_1_2, %tile_3_2, %tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_9(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo @fifo_10(%mem_tile_1_1 dimensionsToStream [<size = 4, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>], {%tile_2_3, %tile_1_3, %tile_0_3, %tile_3_3}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_11(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo @fifo_12(%mem_tile_2_1 dimensionsToStream [<size = 4, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>], {%tile_0_4, %tile_1_4, %tile_3_4, %tile_2_4}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_13(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo @fifo_14(%mem_tile_3_1 dimensionsToStream [<size = 4, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>], {%tile_1_5, %tile_0_5, %tile_3_5, %tile_2_5}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_15(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x32xbf16>> 
    aie.objectfifo @fifo_16(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_17(%tile_0_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_18(%tile_0_4, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_19(%tile_0_5, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_20(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x4x32x32xbf16>> 
    aie.objectfifo @fifo_21(%tile_1_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_22(%tile_1_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_23(%tile_1_4, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_24(%tile_1_5, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_25(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x4x32x32xbf16>> 
    aie.objectfifo @fifo_26(%tile_2_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_27(%tile_2_3, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_28(%tile_2_4, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_29(%tile_2_5, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_30(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1x4x32x32xbf16>> 
    aie.objectfifo @fifo_31(%tile_3_2, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_32(%tile_3_3, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_33(%tile_3_4, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_34(%tile_3_5, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @fifo_35(%mem_tile_3_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1x4x32x32xbf16>> 
    aie.objectfifo.link [@fifo_1] -> [@fifo_0]([] [])
    aie.objectfifo.link [@fifo_9] -> [@fifo_8]([] [])
    aie.objectfifo.link [@fifo_16, @fifo_17, @fifo_18, @fifo_19] -> [@fifo_20]([0, 1024, 2048, 3072] [])
    aie.objectfifo.link [@fifo_3] -> [@fifo_2]([] [])
    aie.objectfifo.link [@fifo_11] -> [@fifo_10]([] [])
    aie.objectfifo.link [@fifo_21, @fifo_22, @fifo_23, @fifo_24] -> [@fifo_25]([0, 1024, 2048, 3072] [])
    aie.objectfifo.link [@fifo_5] -> [@fifo_4]([] [])
    aie.objectfifo.link [@fifo_13] -> [@fifo_12]([] [])
    aie.objectfifo.link [@fifo_26, @fifo_27, @fifo_28, @fifo_29] -> [@fifo_30]([0, 1024, 2048, 3072] [])
    aie.objectfifo.link [@fifo_7] -> [@fifo_6]([] [])
    aie.objectfifo.link [@fifo_15] -> [@fifo_14]([] [])
    aie.objectfifo.link [@fifo_31, @fifo_32, @fifo_33, @fifo_34] -> [@fifo_35]([0, 1024, 2048, 3072] [])
    %core_0_2 = aie.core(%tile_0_2) {
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
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_17(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_10(Consume, 1)
        aie.objectfifo.release @fifo_17(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_18(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_12(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        aie.objectfifo.release @fifo_12(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_12(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_12(Consume, 1)
        aie.objectfifo.release @fifo_18(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_19(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_14(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        aie.objectfifo.release @fifo_14(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_14(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_14(Consume, 1)
        aie.objectfifo.release @fifo_19(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_21(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_8(Consume, 1)
        aie.objectfifo.release @fifo_21(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_22(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_10(Consume, 1)
        aie.objectfifo.release @fifo_22(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_23(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_12(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        aie.objectfifo.release @fifo_12(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_12(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_12(Consume, 1)
        aie.objectfifo.release @fifo_23(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_24(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_14(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        aie.objectfifo.release @fifo_14(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_14(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_14(Consume, 1)
        aie.objectfifo.release @fifo_24(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_26(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        aie.objectfifo.release @fifo_8(Consume, 1)
        aie.objectfifo.release @fifo_26(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_27(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        aie.objectfifo.release @fifo_10(Consume, 1)
        aie.objectfifo.release @fifo_27(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_28(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_12(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        aie.objectfifo.release @fifo_12(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_12(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        aie.objectfifo.release @fifo_12(Consume, 1)
        aie.objectfifo.release @fifo_28(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_29(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_14(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        aie.objectfifo.release @fifo_14(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_14(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        aie.objectfifo.release @fifo_14(Consume, 1)
        aie.objectfifo.release @fifo_29(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_31(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        aie.objectfifo.release @fifo_8(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        aie.objectfifo.release @fifo_8(Consume, 1)
        aie.objectfifo.release @fifo_31(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_32(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        aie.objectfifo.release @fifo_10(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        aie.objectfifo.release @fifo_10(Consume, 1)
        aie.objectfifo.release @fifo_32(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_33(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_12(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        aie.objectfifo.release @fifo_12(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_12(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        aie.objectfifo.release @fifo_12(Consume, 1)
        aie.objectfifo.release @fifo_33(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_3_5 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_34(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @fill_zeros_bf16_32_32_vector(%1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %4 = aie.objectfifo.acquire @fifo_14(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        %6 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        aie.objectfifo.release @fifo_14(Consume, 1)
        %8 = aie.objectfifo.acquire @fifo_14(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        func.call @matmul_bf16_bf16(%7, %9, %1) : (memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        aie.objectfifo.release @fifo_14(Consume, 1)
        aie.objectfifo.release @fifo_34(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    aiex.runtime_sequence(%arg0: memref<8192xbf16>, %arg1: memref<16384xbf16>, %arg2: memref<8192xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 2, 32, 32][2048, 32, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_1} : memref<8192xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[1, 0, 0, 0][1, 2, 32, 32][2048, 32, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_3} : memref<8192xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[2, 0, 0, 0][1, 2, 32, 32][2048, 32, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_5} : memref<8192xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[3, 0, 0, 0][1, 2, 32, 32][2048, 32, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_7} : memref<8192xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 2, 32, 32][32, 4096, 128, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_9} : memref<8192xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[1, 0, 0, 0][1, 2, 32, 32][32, 4096, 128, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_11} : memref<8192xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[2, 0, 0, 0][1, 2, 32, 32][32, 4096, 128, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_13} : memref<8192xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[3, 0, 0, 0][1, 2, 32, 32][32, 4096, 128, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_15} : memref<8192xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 4, 32, 32][4096, 32, 128, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_20} : memref<16384xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[1, 0, 0, 0][1, 4, 32, 32][4096, 32, 128, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_25} : memref<16384xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[2, 0, 0, 0][1, 4, 32, 32][4096, 32, 128, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_30} : memref<16384xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[3, 0, 0, 0][1, 4, 32, 32][4096, 32, 128, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_35} : memref<16384xbf16>
      aiex.npu.dma_wait {symbol = @fifo_20}
      aiex.npu.dma_wait {symbol = @fifo_25}
      aiex.npu.dma_wait {symbol = @fifo_30}
      aiex.npu.dma_wait {symbol = @fifo_35}
      aie.end
    }
  }
}
