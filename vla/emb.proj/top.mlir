module {
  aie.device(npu1_4col) {
    func.func private @fill_zeros_f32_16_16_vector(memref<16x16xf32>)
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
    aie.objectfifo @fifo_0(%mem_tile_0_1, {%tile_0_3, %tile_0_5, %tile_0_2, %tile_0_4}, 2 : i32) : !aie.objectfifo<memref<16x64xf32>> 
    aie.objectfifo @fifo_1(%mem_tile_0_1, {%tile_1_2, %tile_1_3, %tile_1_4, %tile_1_5}, 2 : i32) : !aie.objectfifo<memref<16x64xf32>> 
    aie.objectfifo @fifo_2(%mem_tile_0_1, {%tile_2_2, %tile_2_3, %tile_2_5, %tile_2_4}, 2 : i32) : !aie.objectfifo<memref<16x64xf32>> 
    aie.objectfifo @fifo_3(%mem_tile_0_1, {%tile_3_2, %tile_3_4, %tile_3_3, %tile_3_5}, 2 : i32) : !aie.objectfifo<memref<16x64xf32>> 
    aie.objectfifo @fifo_4(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x4x16x64xf32>> 
    aie.objectfifo @fifo_5(%mem_tile_1_1, {%tile_1_2, %tile_3_2, %tile_2_2, %tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x16xf32>> 
    aie.objectfifo @fifo_6(%mem_tile_1_1, {%tile_3_3, %tile_0_3, %tile_2_3, %tile_1_3}, 2 : i32) : !aie.objectfifo<memref<64x16xf32>> 
    aie.objectfifo @fifo_7(%mem_tile_1_1, {%tile_3_4, %tile_1_4, %tile_2_4, %tile_0_4}, 2 : i32) : !aie.objectfifo<memref<64x16xf32>> 
    aie.objectfifo @fifo_8(%mem_tile_1_1, {%tile_3_5, %tile_1_5, %tile_0_5, %tile_2_5}, 2 : i32) : !aie.objectfifo<memref<64x16xf32>> 
    aie.objectfifo @fifo_9(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x4x64x16xf32>> 
    aie.objectfifo @fifo_10(%tile_0_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<16x16xf32>> 
    aie.objectfifo @fifo_11(%tile_0_3, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<16x16xf32>> 
    aie.objectfifo @fifo_12(%tile_0_4, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<16x16xf32>> 
    aie.objectfifo @fifo_13(%tile_0_5, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<16x16xf32>> 
    aie.objectfifo @fifo_14(%mem_tile_2_1, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<1x4x16x16xf32>> 
    aie.objectfifo @fifo_15(%tile_1_2, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<16x16xf32>> 
    aie.objectfifo @fifo_16(%tile_1_3, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<16x16xf32>> 
    aie.objectfifo @fifo_17(%tile_1_4, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<16x16xf32>> 
    aie.objectfifo @fifo_18(%tile_1_5, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<16x16xf32>> 
    aie.objectfifo @fifo_19(%mem_tile_3_1, {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<1x4x16x16xf32>> 
    aie.objectfifo @fifo_20(%tile_2_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16x16xf32>> 
    aie.objectfifo @fifo_21(%tile_2_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16x16xf32>> 
    aie.objectfifo @fifo_22(%tile_2_4, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16x16xf32>> 
    aie.objectfifo @fifo_23(%tile_2_5, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16x16xf32>> 
    aie.objectfifo @fifo_24(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x4x16x16xf32>> 
    aie.objectfifo @fifo_25(%tile_3_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<16x16xf32>> 
    aie.objectfifo @fifo_26(%tile_3_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<16x16xf32>> 
    aie.objectfifo @fifo_27(%tile_3_4, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<16x16xf32>> 
    aie.objectfifo @fifo_28(%tile_3_5, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<16x16xf32>> 
    aie.objectfifo @fifo_29(%mem_tile_1_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x4x16x16xf32>> 
    aie.objectfifo.link [@fifo_4] -> [@fifo_0, @fifo_1, @fifo_2, @fifo_3]([] [0, 1024, 2048, 3072])
    aie.objectfifo.link [@fifo_20, @fifo_21, @fifo_22, @fifo_23] -> [@fifo_24]([0, 256, 512, 768] [])
    aie.objectfifo.link [@fifo_9] -> [@fifo_5, @fifo_6, @fifo_7, @fifo_8]([] [0, 1024, 2048, 3072])
    aie.objectfifo.link [@fifo_25, @fifo_26, @fifo_27, @fifo_28] -> [@fifo_29]([0, 256, 512, 768] [])
    aie.objectfifo.link [@fifo_10, @fifo_11, @fifo_12, @fifo_13] -> [@fifo_14]([0, 256, 512, 768] [])
    aie.objectfifo.link [@fifo_15, @fifo_16, @fifo_17, @fifo_18] -> [@fifo_19]([0, 256, 512, 768] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_10(Produce, 1) : !aie.objectfifosubview<memref<16x16xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x16xf32>> -> memref<16x16xf32>
        func.call @fill_zeros_f32_16_16_vector(%1) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<16x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x64xf32>> -> memref<16x64xf32>
        %4 = aie.objectfifo.acquire @fifo_5(Consume, 1) : !aie.objectfifosubview<memref<64x16xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16xf32>> -> memref<64x16xf32>
        affine.for %arg1 = 0 to 16 {
          affine.for %arg2 = 0 to 16 {
            affine.for %arg3 = 0 to 64 {
              %6 = affine.load %3[%arg1, %arg3] : memref<16x64xf32>
              %7 = affine.load %5[%arg3, %arg2] : memref<64x16xf32>
              %8 = affine.load %1[%arg1, %arg2] : memref<16x16xf32>
              %9 = arith.mulf %6, %7 : f32
              %10 = arith.addf %8, %9 : f32
              affine.store %10, %1[%arg1, %arg2] : memref<16x16xf32>
            }
          }
        }
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_5(Consume, 1)
        aie.objectfifo.release @fifo_10(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_11(Produce, 1) : !aie.objectfifosubview<memref<16x16xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x16xf32>> -> memref<16x16xf32>
        func.call @fill_zeros_f32_16_16_vector(%1) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<16x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x64xf32>> -> memref<16x64xf32>
        %4 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x16xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16xf32>> -> memref<64x16xf32>
        affine.for %arg1 = 0 to 16 {
          affine.for %arg2 = 0 to 16 {
            affine.for %arg3 = 0 to 64 {
              %6 = affine.load %3[%arg1, %arg3] : memref<16x64xf32>
              %7 = affine.load %5[%arg3, %arg2] : memref<64x16xf32>
              %8 = affine.load %1[%arg1, %arg2] : memref<16x16xf32>
              %9 = arith.mulf %6, %7 : f32
              %10 = arith.addf %8, %9 : f32
              affine.store %10, %1[%arg1, %arg2] : memref<16x16xf32>
            }
          }
        }
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_6(Consume, 1)
        aie.objectfifo.release @fifo_11(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_12(Produce, 1) : !aie.objectfifosubview<memref<16x16xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x16xf32>> -> memref<16x16xf32>
        func.call @fill_zeros_f32_16_16_vector(%1) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<16x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x64xf32>> -> memref<16x64xf32>
        %4 = aie.objectfifo.acquire @fifo_7(Consume, 1) : !aie.objectfifosubview<memref<64x16xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16xf32>> -> memref<64x16xf32>
        affine.for %arg1 = 0 to 16 {
          affine.for %arg2 = 0 to 16 {
            affine.for %arg3 = 0 to 64 {
              %6 = affine.load %3[%arg1, %arg3] : memref<16x64xf32>
              %7 = affine.load %5[%arg3, %arg2] : memref<64x16xf32>
              %8 = affine.load %1[%arg1, %arg2] : memref<16x16xf32>
              %9 = arith.mulf %6, %7 : f32
              %10 = arith.addf %8, %9 : f32
              affine.store %10, %1[%arg1, %arg2] : memref<16x16xf32>
            }
          }
        }
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_7(Consume, 1)
        aie.objectfifo.release @fifo_12(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_13(Produce, 1) : !aie.objectfifosubview<memref<16x16xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x16xf32>> -> memref<16x16xf32>
        func.call @fill_zeros_f32_16_16_vector(%1) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<16x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x64xf32>> -> memref<16x64xf32>
        %4 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x16xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16xf32>> -> memref<64x16xf32>
        affine.for %arg1 = 0 to 16 {
          affine.for %arg2 = 0 to 16 {
            affine.for %arg3 = 0 to 64 {
              %6 = affine.load %3[%arg1, %arg3] : memref<16x64xf32>
              %7 = affine.load %5[%arg3, %arg2] : memref<64x16xf32>
              %8 = affine.load %1[%arg1, %arg2] : memref<16x16xf32>
              %9 = arith.mulf %6, %7 : f32
              %10 = arith.addf %8, %9 : f32
              affine.store %10, %1[%arg1, %arg2] : memref<16x16xf32>
            }
          }
        }
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_8(Consume, 1)
        aie.objectfifo.release @fifo_13(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_15(Produce, 1) : !aie.objectfifosubview<memref<16x16xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x16xf32>> -> memref<16x16xf32>
        func.call @fill_zeros_f32_16_16_vector(%1) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
        %2 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<16x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x64xf32>> -> memref<16x64xf32>
        %4 = aie.objectfifo.acquire @fifo_5(Consume, 1) : !aie.objectfifosubview<memref<64x16xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16xf32>> -> memref<64x16xf32>
        affine.for %arg1 = 0 to 16 {
          affine.for %arg2 = 0 to 16 {
            affine.for %arg3 = 0 to 64 {
              %6 = affine.load %3[%arg1, %arg3] : memref<16x64xf32>
              %7 = affine.load %5[%arg3, %arg2] : memref<64x16xf32>
              %8 = affine.load %1[%arg1, %arg2] : memref<16x16xf32>
              %9 = arith.mulf %6, %7 : f32
              %10 = arith.addf %8, %9 : f32
              affine.store %10, %1[%arg1, %arg2] : memref<16x16xf32>
            }
          }
        }
        aie.objectfifo.release @fifo_1(Consume, 1)
        aie.objectfifo.release @fifo_5(Consume, 1)
        aie.objectfifo.release @fifo_15(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_16(Produce, 1) : !aie.objectfifosubview<memref<16x16xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x16xf32>> -> memref<16x16xf32>
        func.call @fill_zeros_f32_16_16_vector(%1) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
        %2 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<16x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x64xf32>> -> memref<16x64xf32>
        %4 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x16xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16xf32>> -> memref<64x16xf32>
        affine.for %arg1 = 0 to 16 {
          affine.for %arg2 = 0 to 16 {
            affine.for %arg3 = 0 to 64 {
              %6 = affine.load %3[%arg1, %arg3] : memref<16x64xf32>
              %7 = affine.load %5[%arg3, %arg2] : memref<64x16xf32>
              %8 = affine.load %1[%arg1, %arg2] : memref<16x16xf32>
              %9 = arith.mulf %6, %7 : f32
              %10 = arith.addf %8, %9 : f32
              affine.store %10, %1[%arg1, %arg2] : memref<16x16xf32>
            }
          }
        }
        aie.objectfifo.release @fifo_1(Consume, 1)
        aie.objectfifo.release @fifo_6(Consume, 1)
        aie.objectfifo.release @fifo_16(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_17(Produce, 1) : !aie.objectfifosubview<memref<16x16xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x16xf32>> -> memref<16x16xf32>
        func.call @fill_zeros_f32_16_16_vector(%1) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
        %2 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<16x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x64xf32>> -> memref<16x64xf32>
        %4 = aie.objectfifo.acquire @fifo_7(Consume, 1) : !aie.objectfifosubview<memref<64x16xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16xf32>> -> memref<64x16xf32>
        affine.for %arg1 = 0 to 16 {
          affine.for %arg2 = 0 to 16 {
            affine.for %arg3 = 0 to 64 {
              %6 = affine.load %3[%arg1, %arg3] : memref<16x64xf32>
              %7 = affine.load %5[%arg3, %arg2] : memref<64x16xf32>
              %8 = affine.load %1[%arg1, %arg2] : memref<16x16xf32>
              %9 = arith.mulf %6, %7 : f32
              %10 = arith.addf %8, %9 : f32
              affine.store %10, %1[%arg1, %arg2] : memref<16x16xf32>
            }
          }
        }
        aie.objectfifo.release @fifo_1(Consume, 1)
        aie.objectfifo.release @fifo_7(Consume, 1)
        aie.objectfifo.release @fifo_17(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_18(Produce, 1) : !aie.objectfifosubview<memref<16x16xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x16xf32>> -> memref<16x16xf32>
        func.call @fill_zeros_f32_16_16_vector(%1) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
        %2 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<16x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x64xf32>> -> memref<16x64xf32>
        %4 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x16xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16xf32>> -> memref<64x16xf32>
        affine.for %arg1 = 0 to 16 {
          affine.for %arg2 = 0 to 16 {
            affine.for %arg3 = 0 to 64 {
              %6 = affine.load %3[%arg1, %arg3] : memref<16x64xf32>
              %7 = affine.load %5[%arg3, %arg2] : memref<64x16xf32>
              %8 = affine.load %1[%arg1, %arg2] : memref<16x16xf32>
              %9 = arith.mulf %6, %7 : f32
              %10 = arith.addf %8, %9 : f32
              affine.store %10, %1[%arg1, %arg2] : memref<16x16xf32>
            }
          }
        }
        aie.objectfifo.release @fifo_1(Consume, 1)
        aie.objectfifo.release @fifo_8(Consume, 1)
        aie.objectfifo.release @fifo_18(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_20(Produce, 1) : !aie.objectfifosubview<memref<16x16xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x16xf32>> -> memref<16x16xf32>
        func.call @fill_zeros_f32_16_16_vector(%1) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
        %2 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<16x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x64xf32>> -> memref<16x64xf32>
        %4 = aie.objectfifo.acquire @fifo_5(Consume, 1) : !aie.objectfifosubview<memref<64x16xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16xf32>> -> memref<64x16xf32>
        affine.for %arg1 = 0 to 16 {
          affine.for %arg2 = 0 to 16 {
            affine.for %arg3 = 0 to 64 {
              %6 = affine.load %3[%arg1, %arg3] : memref<16x64xf32>
              %7 = affine.load %5[%arg3, %arg2] : memref<64x16xf32>
              %8 = affine.load %1[%arg1, %arg2] : memref<16x16xf32>
              %9 = arith.mulf %6, %7 : f32
              %10 = arith.addf %8, %9 : f32
              affine.store %10, %1[%arg1, %arg2] : memref<16x16xf32>
            }
          }
        }
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_5(Consume, 1)
        aie.objectfifo.release @fifo_20(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_21(Produce, 1) : !aie.objectfifosubview<memref<16x16xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x16xf32>> -> memref<16x16xf32>
        func.call @fill_zeros_f32_16_16_vector(%1) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
        %2 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<16x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x64xf32>> -> memref<16x64xf32>
        %4 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x16xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16xf32>> -> memref<64x16xf32>
        affine.for %arg1 = 0 to 16 {
          affine.for %arg2 = 0 to 16 {
            affine.for %arg3 = 0 to 64 {
              %6 = affine.load %3[%arg1, %arg3] : memref<16x64xf32>
              %7 = affine.load %5[%arg3, %arg2] : memref<64x16xf32>
              %8 = affine.load %1[%arg1, %arg2] : memref<16x16xf32>
              %9 = arith.mulf %6, %7 : f32
              %10 = arith.addf %8, %9 : f32
              affine.store %10, %1[%arg1, %arg2] : memref<16x16xf32>
            }
          }
        }
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_6(Consume, 1)
        aie.objectfifo.release @fifo_21(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_22(Produce, 1) : !aie.objectfifosubview<memref<16x16xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x16xf32>> -> memref<16x16xf32>
        func.call @fill_zeros_f32_16_16_vector(%1) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
        %2 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<16x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x64xf32>> -> memref<16x64xf32>
        %4 = aie.objectfifo.acquire @fifo_7(Consume, 1) : !aie.objectfifosubview<memref<64x16xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16xf32>> -> memref<64x16xf32>
        affine.for %arg1 = 0 to 16 {
          affine.for %arg2 = 0 to 16 {
            affine.for %arg3 = 0 to 64 {
              %6 = affine.load %3[%arg1, %arg3] : memref<16x64xf32>
              %7 = affine.load %5[%arg3, %arg2] : memref<64x16xf32>
              %8 = affine.load %1[%arg1, %arg2] : memref<16x16xf32>
              %9 = arith.mulf %6, %7 : f32
              %10 = arith.addf %8, %9 : f32
              affine.store %10, %1[%arg1, %arg2] : memref<16x16xf32>
            }
          }
        }
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_7(Consume, 1)
        aie.objectfifo.release @fifo_22(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_23(Produce, 1) : !aie.objectfifosubview<memref<16x16xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x16xf32>> -> memref<16x16xf32>
        func.call @fill_zeros_f32_16_16_vector(%1) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
        %2 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<16x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x64xf32>> -> memref<16x64xf32>
        %4 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x16xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16xf32>> -> memref<64x16xf32>
        affine.for %arg1 = 0 to 16 {
          affine.for %arg2 = 0 to 16 {
            affine.for %arg3 = 0 to 64 {
              %6 = affine.load %3[%arg1, %arg3] : memref<16x64xf32>
              %7 = affine.load %5[%arg3, %arg2] : memref<64x16xf32>
              %8 = affine.load %1[%arg1, %arg2] : memref<16x16xf32>
              %9 = arith.mulf %6, %7 : f32
              %10 = arith.addf %8, %9 : f32
              affine.store %10, %1[%arg1, %arg2] : memref<16x16xf32>
            }
          }
        }
        aie.objectfifo.release @fifo_2(Consume, 1)
        aie.objectfifo.release @fifo_8(Consume, 1)
        aie.objectfifo.release @fifo_23(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_25(Produce, 1) : !aie.objectfifosubview<memref<16x16xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x16xf32>> -> memref<16x16xf32>
        func.call @fill_zeros_f32_16_16_vector(%1) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
        %2 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<16x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x64xf32>> -> memref<16x64xf32>
        %4 = aie.objectfifo.acquire @fifo_5(Consume, 1) : !aie.objectfifosubview<memref<64x16xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16xf32>> -> memref<64x16xf32>
        affine.for %arg1 = 0 to 16 {
          affine.for %arg2 = 0 to 16 {
            affine.for %arg3 = 0 to 64 {
              %6 = affine.load %3[%arg1, %arg3] : memref<16x64xf32>
              %7 = affine.load %5[%arg3, %arg2] : memref<64x16xf32>
              %8 = affine.load %1[%arg1, %arg2] : memref<16x16xf32>
              %9 = arith.mulf %6, %7 : f32
              %10 = arith.addf %8, %9 : f32
              affine.store %10, %1[%arg1, %arg2] : memref<16x16xf32>
            }
          }
        }
        aie.objectfifo.release @fifo_3(Consume, 1)
        aie.objectfifo.release @fifo_5(Consume, 1)
        aie.objectfifo.release @fifo_25(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_26(Produce, 1) : !aie.objectfifosubview<memref<16x16xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x16xf32>> -> memref<16x16xf32>
        func.call @fill_zeros_f32_16_16_vector(%1) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
        %2 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<16x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x64xf32>> -> memref<16x64xf32>
        %4 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<64x16xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16xf32>> -> memref<64x16xf32>
        affine.for %arg1 = 0 to 16 {
          affine.for %arg2 = 0 to 16 {
            affine.for %arg3 = 0 to 64 {
              %6 = affine.load %3[%arg1, %arg3] : memref<16x64xf32>
              %7 = affine.load %5[%arg3, %arg2] : memref<64x16xf32>
              %8 = affine.load %1[%arg1, %arg2] : memref<16x16xf32>
              %9 = arith.mulf %6, %7 : f32
              %10 = arith.addf %8, %9 : f32
              affine.store %10, %1[%arg1, %arg2] : memref<16x16xf32>
            }
          }
        }
        aie.objectfifo.release @fifo_3(Consume, 1)
        aie.objectfifo.release @fifo_6(Consume, 1)
        aie.objectfifo.release @fifo_26(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_27(Produce, 1) : !aie.objectfifosubview<memref<16x16xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x16xf32>> -> memref<16x16xf32>
        func.call @fill_zeros_f32_16_16_vector(%1) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
        %2 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<16x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x64xf32>> -> memref<16x64xf32>
        %4 = aie.objectfifo.acquire @fifo_7(Consume, 1) : !aie.objectfifosubview<memref<64x16xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16xf32>> -> memref<64x16xf32>
        affine.for %arg1 = 0 to 16 {
          affine.for %arg2 = 0 to 16 {
            affine.for %arg3 = 0 to 64 {
              %6 = affine.load %3[%arg1, %arg3] : memref<16x64xf32>
              %7 = affine.load %5[%arg3, %arg2] : memref<64x16xf32>
              %8 = affine.load %1[%arg1, %arg2] : memref<16x16xf32>
              %9 = arith.mulf %6, %7 : f32
              %10 = arith.addf %8, %9 : f32
              affine.store %10, %1[%arg1, %arg2] : memref<16x16xf32>
            }
          }
        }
        aie.objectfifo.release @fifo_3(Consume, 1)
        aie.objectfifo.release @fifo_7(Consume, 1)
        aie.objectfifo.release @fifo_27(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_3_5 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_28(Produce, 1) : !aie.objectfifosubview<memref<16x16xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x16xf32>> -> memref<16x16xf32>
        func.call @fill_zeros_f32_16_16_vector(%1) {lib = "fill_zeros_f32_16_16_vector"} : (memref<16x16xf32>) -> ()
        %2 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<16x64xf32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x64xf32>> -> memref<16x64xf32>
        %4 = aie.objectfifo.acquire @fifo_8(Consume, 1) : !aie.objectfifosubview<memref<64x16xf32>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x16xf32>> -> memref<64x16xf32>
        affine.for %arg1 = 0 to 16 {
          affine.for %arg2 = 0 to 16 {
            affine.for %arg3 = 0 to 64 {
              %6 = affine.load %3[%arg1, %arg3] : memref<16x64xf32>
              %7 = affine.load %5[%arg3, %arg2] : memref<64x16xf32>
              %8 = affine.load %1[%arg1, %arg2] : memref<16x16xf32>
              %9 = arith.mulf %6, %7 : f32
              %10 = arith.addf %8, %9 : f32
              affine.store %10, %1[%arg1, %arg2] : memref<16x16xf32>
            }
          }
        }
        aie.objectfifo.release @fifo_3(Consume, 1)
        aie.objectfifo.release @fifo_8(Consume, 1)
        aie.objectfifo.release @fifo_28(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    aiex.runtime_sequence(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 4, 16, 64][0, 1024, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_4} : memref<4096xf32>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 4, 64, 16][0, 16, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_9} : memref<4096xf32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 4, 16, 16][1024, 16, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_14} : memref<4096xf32>
      aiex.npu.dma_memcpy_nd(%arg1[1, 0, 0, 0][1, 4, 16, 16][1024, 16, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_19} : memref<4096xf32>
      aiex.npu.dma_memcpy_nd(%arg1[2, 0, 0, 0][1, 4, 16, 16][1024, 16, 64, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_24} : memref<4096xf32>
      aiex.npu.dma_memcpy_nd(%arg1[3, 0, 0, 0][1, 4, 16, 16][1024, 16, 64, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_29} : memref<4096xf32>
      aiex.npu.dma_wait {symbol = @fifo_14}
      aiex.npu.dma_wait {symbol = @fifo_19}
      aiex.npu.dma_wait {symbol = @fifo_24}
      aiex.npu.dma_wait {symbol = @fifo_29}
      aie.end
    }
  }
}
