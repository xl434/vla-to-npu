module {
  aie.device(npu1_4col) {
    func.func private @rope_fused_float32(memref<32x64xf32>, memref<32x64xf32>, memref<32x64xf32>)
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
    aie.objectfifo @fifo_0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_1(%mem_tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_2(%mem_tile_0_1, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_3(%mem_tile_0_1, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_4(%mem_tile_0_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_5(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<5x1x32x64xf32>> 
    aie.objectfifo @fifo_6(%mem_tile_1_1, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_7(%mem_tile_1_1, {%tile_1_4}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_8(%mem_tile_1_1, {%tile_1_5}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_9(%mem_tile_1_1, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_10(%mem_tile_1_1, {%tile_2_3}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_11(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<5x1x32x64xf32>> 
    aie.objectfifo @fifo_12(%mem_tile_2_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_13(%mem_tile_2_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_14(%mem_tile_2_1, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_15(%mem_tile_2_1, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_16(%mem_tile_2_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_17(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<5x1x32x64xf32>> 
    aie.objectfifo @fifo_18(%mem_tile_3_1, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_19(%mem_tile_3_1, {%tile_1_4}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_20(%mem_tile_3_1, {%tile_1_5}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_21(%mem_tile_3_1, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_22(%mem_tile_3_1, {%tile_2_3}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_23(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<5x1x32x64xf32>> 
    aie.objectfifo @fifo_24(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_25(%tile_0_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_26(%tile_0_4, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_27(%tile_0_5, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_28(%tile_1_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_29(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<5x1x32x64xf32>> 
    aie.objectfifo @fifo_30(%tile_1_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_31(%tile_1_4, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_32(%tile_1_5, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_33(%tile_2_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_34(%tile_2_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<32x64xf32>> 
    aie.objectfifo @fifo_35(%mem_tile_1_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<5x1x32x64xf32>> 
    aie.objectfifo.link [@fifo_5] -> [@fifo_0, @fifo_1, @fifo_2, @fifo_3, @fifo_4]([] [0, 2048, 4096, 6144, 8192])
    aie.objectfifo.link [@fifo_24, @fifo_25, @fifo_26, @fifo_27, @fifo_28] -> [@fifo_29]([0, 2048, 4096, 6144, 8192] [])
    aie.objectfifo.link [@fifo_11] -> [@fifo_6, @fifo_7, @fifo_8, @fifo_9, @fifo_10]([] [0, 2048, 4096, 6144, 8192])
    aie.objectfifo.link [@fifo_30, @fifo_31, @fifo_32, @fifo_33, @fifo_34] -> [@fifo_35]([0, 2048, 4096, 6144, 8192] [])
    aie.objectfifo.link [@fifo_17] -> [@fifo_12, @fifo_13, @fifo_14, @fifo_15, @fifo_16]([] [0, 2048, 4096, 6144, 8192])
    aie.objectfifo.link [@fifo_23] -> [@fifo_18, @fifo_19, @fifo_20, @fifo_21, @fifo_22]([] [0, 2048, 4096, 6144, 8192])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %2 = aie.objectfifo.acquire @fifo_12(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %4 = aie.objectfifo.acquire @fifo_24(Produce, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        func.call @rope_fused_float32(%1, %3, %5) : (memref<32x64xf32>, memref<32x64xf32>, memref<32x64xf32>) -> ()
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_12(Consume, 1)
        aie.objectfifo.release @fifo_24(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %2 = aie.objectfifo.acquire @fifo_13(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %4 = aie.objectfifo.acquire @fifo_25(Produce, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        func.call @rope_fused_float32(%1, %3, %5) : (memref<32x64xf32>, memref<32x64xf32>, memref<32x64xf32>) -> ()
        aie.objectfifo.release @fifo_1(Consume, 1)
        aie.objectfifo.release @fifo_13(Consume, 1)
        aie.objectfifo.release @fifo_25(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %2 = aie.objectfifo.acquire @fifo_14(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %4 = aie.objectfifo.acquire @fifo_26(Produce, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        func.call @rope_fused_float32(%1, %3, %5) : (memref<32x64xf32>, memref<32x64xf32>, memref<32x64xf32>) -> ()
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_14(Consume, 1)
        aie.objectfifo.release @fifo_26(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %2 = aie.objectfifo.acquire @fifo_15(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %4 = aie.objectfifo.acquire @fifo_27(Produce, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        func.call @rope_fused_float32(%1, %3, %5) : (memref<32x64xf32>, memref<32x64xf32>, memref<32x64xf32>) -> ()
        aie.objectfifo.release @fifo_3(Consume, 1)
        aie.objectfifo.release @fifo_15(Consume, 1)
        aie.objectfifo.release @fifo_27(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %2 = aie.objectfifo.acquire @fifo_16(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %4 = aie.objectfifo.acquire @fifo_28(Produce, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        func.call @rope_fused_float32(%1, %3, %5) : (memref<32x64xf32>, memref<32x64xf32>, memref<32x64xf32>) -> ()
        aie.objectfifo.release @fifo_4(Consume, 1)
        aie.objectfifo.release @fifo_16(Consume, 1)
        aie.objectfifo.release @fifo_28(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %2 = aie.objectfifo.acquire @fifo_18(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %4 = aie.objectfifo.acquire @fifo_30(Produce, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        func.call @rope_fused_float32(%1, %3, %5) : (memref<32x64xf32>, memref<32x64xf32>, memref<32x64xf32>) -> ()
        aie.objectfifo.release @fifo_6(Consume, 1)
        aie.objectfifo.release @fifo_18(Consume, 1)
        aie.objectfifo.release @fifo_30(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_7(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %2 = aie.objectfifo.acquire @fifo_19(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %4 = aie.objectfifo.acquire @fifo_31(Produce, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        func.call @rope_fused_float32(%1, %3, %5) : (memref<32x64xf32>, memref<32x64xf32>, memref<32x64xf32>) -> ()
        aie.objectfifo.release @fifo_7(Consume, 1)
        aie.objectfifo.release @fifo_19(Consume, 1)
        aie.objectfifo.release @fifo_31(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %2 = aie.objectfifo.acquire @fifo_20(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %4 = aie.objectfifo.acquire @fifo_32(Produce, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        func.call @rope_fused_float32(%1, %3, %5) : (memref<32x64xf32>, memref<32x64xf32>, memref<32x64xf32>) -> ()
        aie.objectfifo.release @fifo_8(Consume, 1)
        aie.objectfifo.release @fifo_20(Consume, 1)
        aie.objectfifo.release @fifo_32(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_9(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %2 = aie.objectfifo.acquire @fifo_21(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %4 = aie.objectfifo.acquire @fifo_33(Produce, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        func.call @rope_fused_float32(%1, %3, %5) : (memref<32x64xf32>, memref<32x64xf32>, memref<32x64xf32>) -> ()
        aie.objectfifo.release @fifo_9(Consume, 1)
        aie.objectfifo.release @fifo_21(Consume, 1)
        aie.objectfifo.release @fifo_33(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %2 = aie.objectfifo.acquire @fifo_22(Consume, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        %4 = aie.objectfifo.acquire @fifo_34(Produce, 1) : !aie.objectfifosubview<memref<32x64xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xf32>> -> memref<32x64xf32>
        func.call @rope_fused_float32(%1, %3, %5) : (memref<32x64xf32>, memref<32x64xf32>, memref<32x64xf32>) -> ()
        aie.objectfifo.release @fifo_10(Consume, 1)
        aie.objectfifo.release @fifo_22(Consume, 1)
        aie.objectfifo.release @fifo_34(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    aiex.runtime_sequence(%arg0: memref<20480xf32>, %arg1: memref<20480xf32>, %arg2: memref<20480xf32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 5, 32, 64][64, 2048, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_5} : memref<20480xf32>
      aiex.npu.dma_memcpy_nd(%arg0[0, 5, 0, 0][1, 5, 32, 64][64, 2048, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<20480xf32>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 5, 32, 64][64, 2048, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_17} : memref<20480xf32>
      aiex.npu.dma_memcpy_nd(%arg2[0, 5, 0, 0][1, 5, 32, 64][64, 2048, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_23} : memref<20480xf32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 5, 32, 64][64, 2048, 64, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_29} : memref<20480xf32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 5, 0, 0][1, 5, 32, 64][64, 2048, 64, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_35} : memref<20480xf32>
      aiex.npu.dma_wait {symbol = @fifo_29}
      aiex.npu.dma_wait {symbol = @fifo_35}
      aie.end
    }
  }
}
