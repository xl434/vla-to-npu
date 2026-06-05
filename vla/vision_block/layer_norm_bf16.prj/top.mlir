module {
  aie.device(npu1_4col) {
    func.func private @layer_norm(memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>)
    func.func private @fill_zeros_bf16_4_768_vector(memref<4x768xbf16>)
    func.func private @add_bf16_vector(memref<4x768xbf16>, memref<4x768xbf16>, memref<4x768xbf16>)
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
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_4 = aie.tile(2, 4)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_4 = aie.tile(3, 4)
    %tile_3_5 = aie.tile(3, 5)
    aie.objectfifo @pipe_0(%tile_0_3, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @pipe_1(%tile_1_3, {%tile_1_2}, 1 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @pipe_2(%tile_2_3, {%tile_2_2}, 1 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @pipe_3(%tile_3_3, {%tile_3_2}, 1 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @pipe_4(%tile_0_5, {%tile_0_4}, 1 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @pipe_5(%tile_1_5, {%tile_1_4}, 1 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @pipe_6(%tile_2_5, {%tile_2_4}, 1 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @pipe_7(%tile_3_5, {%tile_3_4}, 1 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @fifo_0(%mem_tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @fifo_1(%mem_tile_0_1, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @fifo_2(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x2x4x768xbf16>> 
    aie.objectfifo @fifo_3(%mem_tile_1_1, {%tile_2_3}, 2 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @fifo_4(%mem_tile_1_1, {%tile_3_3}, 2 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @fifo_5(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x2x4x768xbf16>> 
    aie.objectfifo @fifo_6(%mem_tile_2_1, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @fifo_7(%mem_tile_2_1, {%tile_1_5}, 2 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @fifo_8(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1x2x4x768xbf16>> 
    aie.objectfifo @fifo_9(%mem_tile_3_1, {%tile_2_5}, 2 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @fifo_10(%mem_tile_3_1, {%tile_3_5}, 2 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @fifo_11(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<1x2x4x768xbf16>> 
    aie.objectfifo @fifo_12(%mem_tile_0_1, {%tile_3_5, %tile_3_3, %tile_1_5, %tile_2_3, %tile_0_5, %tile_0_3, %tile_2_5, %tile_1_3}, 2 : i32) : !aie.objectfifo<memref<768xbf16>> 
    aie.objectfifo @fifo_13(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x1x768xbf16>> 
    aie.objectfifo @fifo_14(%mem_tile_0_1, {%tile_3_2, %tile_0_4, %tile_1_2, %tile_1_4, %tile_2_2, %tile_2_4, %tile_3_4, %tile_0_2}, 2 : i32) : !aie.objectfifo<memref<768xbf16>> 
    aie.objectfifo @fifo_15(%shim_noc_tile_1_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x1x768xbf16>> 
    aie.objectfifo @fifo_16(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @fifo_17(%tile_1_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @fifo_18(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x2x4x768xbf16>> 
    aie.objectfifo @fifo_19(%tile_2_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @fifo_20(%tile_3_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @fifo_21(%mem_tile_1_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1x2x4x768xbf16>> 
    aie.objectfifo @fifo_22(%tile_0_4, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @fifo_23(%tile_1_4, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @fifo_24(%mem_tile_2_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1x2x4x768xbf16>> 
    aie.objectfifo @fifo_25(%tile_2_4, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @fifo_26(%tile_3_4, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<4x768xbf16>> 
    aie.objectfifo @fifo_27(%mem_tile_3_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1x2x4x768xbf16>> 
    aie.objectfifo.link [@fifo_2] -> [@fifo_0, @fifo_1]([] [0, 3072])
    aie.objectfifo.link [@fifo_13] -> [@fifo_12]([] [])
    aie.objectfifo.link [@fifo_15] -> [@fifo_14]([] [])
    aie.objectfifo.link [@fifo_16, @fifo_17] -> [@fifo_18]([0, 3072] [])
    aie.objectfifo.link [@fifo_5] -> [@fifo_3, @fifo_4]([] [0, 3072])
    aie.objectfifo.link [@fifo_19, @fifo_20] -> [@fifo_21]([0, 3072] [])
    aie.objectfifo.link [@fifo_8] -> [@fifo_6, @fifo_7]([] [0, 3072])
    aie.objectfifo.link [@fifo_22, @fifo_23] -> [@fifo_24]([0, 3072] [])
    aie.objectfifo.link [@fifo_11] -> [@fifo_9, @fifo_10]([] [0, 3072])
    aie.objectfifo.link [@fifo_25, @fifo_26] -> [@fifo_27]([0, 3072] [])
    %buffer_0_2 = aie.buffer(%tile_0_2) : memref<4x768xbf16> 
    %buffer_0_2_0 = aie.buffer(%tile_0_2) : memref<4x768xbf16> 
    %buffer_1_2 = aie.buffer(%tile_1_2) : memref<4x768xbf16> 
    %buffer_1_2_1 = aie.buffer(%tile_1_2) : memref<4x768xbf16> 
    %buffer_2_2 = aie.buffer(%tile_2_2) : memref<4x768xbf16> 
    %buffer_2_2_2 = aie.buffer(%tile_2_2) : memref<4x768xbf16> 
    %buffer_3_2 = aie.buffer(%tile_3_2) : memref<4x768xbf16> 
    %buffer_3_2_3 = aie.buffer(%tile_3_2) : memref<4x768xbf16> 
    %buffer_0_4 = aie.buffer(%tile_0_4) : memref<4x768xbf16> 
    %buffer_0_4_4 = aie.buffer(%tile_0_4) : memref<4x768xbf16> 
    %buffer_1_4 = aie.buffer(%tile_1_4) : memref<4x768xbf16> 
    %buffer_1_4_5 = aie.buffer(%tile_1_4) : memref<4x768xbf16> 
    %buffer_2_4 = aie.buffer(%tile_2_4) : memref<4x768xbf16> 
    %buffer_2_4_6 = aie.buffer(%tile_2_4) : memref<4x768xbf16> 
    %buffer_3_4 = aie.buffer(%tile_3_4) : memref<4x768xbf16> 
    %buffer_3_4_7 = aie.buffer(%tile_3_4) : memref<4x768xbf16> 
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @pipe_0(Produce, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        func.call @fill_zeros_bf16_4_768_vector(%1) {lib = "fill_zeros_bf16_4_768_vector"} : (memref<4x768xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        %4 = aie.objectfifo.acquire @fifo_12(Consume, 1) : !aie.objectfifosubview<memref<768xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<768xbf16>> -> memref<768xbf16>
        func.call @layer_norm(%3, %5, %1) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
        aie.objectfifo.release @pipe_0(Produce, 1)
        aie.objectfifo.release @fifo_0(Consume, 1)
        aie.objectfifo.release @fifo_12(Consume, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @pipe_0(Consume, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        %2 = aie.objectfifo.acquire @fifo_14(Consume, 1) : !aie.objectfifosubview<memref<768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<768xbf16>> -> memref<768xbf16>
        affine.for %arg1 = 0 to 4 {
          affine.for %arg2 = 0 to 768 {
            %6 = affine.load %3[%arg2] : memref<768xbf16>
            affine.store %6, %buffer_0_2_0[%arg1, %arg2] : memref<4x768xbf16>
          }
        }
        %4 = aie.objectfifo.acquire @fifo_16(Produce, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        func.call @add_bf16_vector(%1, %buffer_0_2_0, %5) {lib = "add_bf16_vector"} : (memref<4x768xbf16>, memref<4x768xbf16>, memref<4x768xbf16>) -> ()
        aie.objectfifo.release @pipe_0(Consume, 1)
        aie.objectfifo.release @fifo_14(Consume, 1)
        aie.objectfifo.release @fifo_16(Produce, 1)
      }
      aie.end
    } {link_with = "external1.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @pipe_1(Produce, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        func.call @fill_zeros_bf16_4_768_vector(%1) {lib = "fill_zeros_bf16_4_768_vector"} : (memref<4x768xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_1(Consume, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        %4 = aie.objectfifo.acquire @fifo_12(Consume, 1) : !aie.objectfifosubview<memref<768xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<768xbf16>> -> memref<768xbf16>
        func.call @layer_norm(%3, %5, %1) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
        aie.objectfifo.release @pipe_1(Produce, 1)
        aie.objectfifo.release @fifo_1(Consume, 1)
        aie.objectfifo.release @fifo_12(Consume, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @pipe_2(Produce, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        func.call @fill_zeros_bf16_4_768_vector(%1) {lib = "fill_zeros_bf16_4_768_vector"} : (memref<4x768xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_3(Consume, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        %4 = aie.objectfifo.acquire @fifo_12(Consume, 1) : !aie.objectfifosubview<memref<768xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<768xbf16>> -> memref<768xbf16>
        func.call @layer_norm(%3, %5, %1) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
        aie.objectfifo.release @pipe_2(Produce, 1)
        aie.objectfifo.release @fifo_3(Consume, 1)
        aie.objectfifo.release @fifo_12(Consume, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @pipe_3(Produce, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        func.call @fill_zeros_bf16_4_768_vector(%1) {lib = "fill_zeros_bf16_4_768_vector"} : (memref<4x768xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        %4 = aie.objectfifo.acquire @fifo_12(Consume, 1) : !aie.objectfifosubview<memref<768xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<768xbf16>> -> memref<768xbf16>
        func.call @layer_norm(%3, %5, %1) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
        aie.objectfifo.release @pipe_3(Produce, 1)
        aie.objectfifo.release @fifo_4(Consume, 1)
        aie.objectfifo.release @fifo_12(Consume, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @pipe_4(Produce, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        func.call @fill_zeros_bf16_4_768_vector(%1) {lib = "fill_zeros_bf16_4_768_vector"} : (memref<4x768xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_6(Consume, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        %4 = aie.objectfifo.acquire @fifo_12(Consume, 1) : !aie.objectfifosubview<memref<768xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<768xbf16>> -> memref<768xbf16>
        func.call @layer_norm(%3, %5, %1) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
        aie.objectfifo.release @pipe_4(Produce, 1)
        aie.objectfifo.release @fifo_6(Consume, 1)
        aie.objectfifo.release @fifo_12(Consume, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @pipe_5(Produce, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        func.call @fill_zeros_bf16_4_768_vector(%1) {lib = "fill_zeros_bf16_4_768_vector"} : (memref<4x768xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_7(Consume, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        %4 = aie.objectfifo.acquire @fifo_12(Consume, 1) : !aie.objectfifosubview<memref<768xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<768xbf16>> -> memref<768xbf16>
        func.call @layer_norm(%3, %5, %1) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
        aie.objectfifo.release @pipe_5(Produce, 1)
        aie.objectfifo.release @fifo_7(Consume, 1)
        aie.objectfifo.release @fifo_12(Consume, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @pipe_6(Produce, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        func.call @fill_zeros_bf16_4_768_vector(%1) {lib = "fill_zeros_bf16_4_768_vector"} : (memref<4x768xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_9(Consume, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        %4 = aie.objectfifo.acquire @fifo_12(Consume, 1) : !aie.objectfifosubview<memref<768xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<768xbf16>> -> memref<768xbf16>
        func.call @layer_norm(%3, %5, %1) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
        aie.objectfifo.release @pipe_6(Produce, 1)
        aie.objectfifo.release @fifo_9(Consume, 1)
        aie.objectfifo.release @fifo_12(Consume, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_3_5 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @pipe_7(Produce, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        func.call @fill_zeros_bf16_4_768_vector(%1) {lib = "fill_zeros_bf16_4_768_vector"} : (memref<4x768xbf16>) -> ()
        %2 = aie.objectfifo.acquire @fifo_10(Consume, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        %4 = aie.objectfifo.acquire @fifo_12(Consume, 1) : !aie.objectfifosubview<memref<768xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<768xbf16>> -> memref<768xbf16>
        func.call @layer_norm(%3, %5, %1) : (memref<4x768xbf16>, memref<768xbf16>, memref<4x768xbf16>) -> ()
        aie.objectfifo.release @pipe_7(Produce, 1)
        aie.objectfifo.release @fifo_10(Consume, 1)
        aie.objectfifo.release @fifo_12(Consume, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @pipe_1(Consume, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        %2 = aie.objectfifo.acquire @fifo_14(Consume, 1) : !aie.objectfifosubview<memref<768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<768xbf16>> -> memref<768xbf16>
        affine.for %arg1 = 0 to 4 {
          affine.for %arg2 = 0 to 768 {
            %6 = affine.load %3[%arg2] : memref<768xbf16>
            affine.store %6, %buffer_1_2_1[%arg1, %arg2] : memref<4x768xbf16>
          }
        }
        %4 = aie.objectfifo.acquire @fifo_17(Produce, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        func.call @add_bf16_vector(%1, %buffer_1_2_1, %5) {lib = "add_bf16_vector"} : (memref<4x768xbf16>, memref<4x768xbf16>, memref<4x768xbf16>) -> ()
        aie.objectfifo.release @pipe_1(Consume, 1)
        aie.objectfifo.release @fifo_14(Consume, 1)
        aie.objectfifo.release @fifo_17(Produce, 1)
      }
      aie.end
    } {link_with = "external1.o"}
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @pipe_2(Consume, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        %2 = aie.objectfifo.acquire @fifo_14(Consume, 1) : !aie.objectfifosubview<memref<768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<768xbf16>> -> memref<768xbf16>
        affine.for %arg1 = 0 to 4 {
          affine.for %arg2 = 0 to 768 {
            %6 = affine.load %3[%arg2] : memref<768xbf16>
            affine.store %6, %buffer_2_2_2[%arg1, %arg2] : memref<4x768xbf16>
          }
        }
        %4 = aie.objectfifo.acquire @fifo_19(Produce, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        func.call @add_bf16_vector(%1, %buffer_2_2_2, %5) {lib = "add_bf16_vector"} : (memref<4x768xbf16>, memref<4x768xbf16>, memref<4x768xbf16>) -> ()
        aie.objectfifo.release @pipe_2(Consume, 1)
        aie.objectfifo.release @fifo_14(Consume, 1)
        aie.objectfifo.release @fifo_19(Produce, 1)
      }
      aie.end
    } {link_with = "external1.o"}
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @pipe_3(Consume, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        %2 = aie.objectfifo.acquire @fifo_14(Consume, 1) : !aie.objectfifosubview<memref<768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<768xbf16>> -> memref<768xbf16>
        affine.for %arg1 = 0 to 4 {
          affine.for %arg2 = 0 to 768 {
            %6 = affine.load %3[%arg2] : memref<768xbf16>
            affine.store %6, %buffer_3_2_3[%arg1, %arg2] : memref<4x768xbf16>
          }
        }
        %4 = aie.objectfifo.acquire @fifo_20(Produce, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        func.call @add_bf16_vector(%1, %buffer_3_2_3, %5) {lib = "add_bf16_vector"} : (memref<4x768xbf16>, memref<4x768xbf16>, memref<4x768xbf16>) -> ()
        aie.objectfifo.release @pipe_3(Consume, 1)
        aie.objectfifo.release @fifo_14(Consume, 1)
        aie.objectfifo.release @fifo_20(Produce, 1)
      }
      aie.end
    } {link_with = "external1.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @pipe_4(Consume, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        %2 = aie.objectfifo.acquire @fifo_14(Consume, 1) : !aie.objectfifosubview<memref<768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<768xbf16>> -> memref<768xbf16>
        affine.for %arg1 = 0 to 4 {
          affine.for %arg2 = 0 to 768 {
            %6 = affine.load %3[%arg2] : memref<768xbf16>
            affine.store %6, %buffer_0_4_4[%arg1, %arg2] : memref<4x768xbf16>
          }
        }
        %4 = aie.objectfifo.acquire @fifo_22(Produce, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        func.call @add_bf16_vector(%1, %buffer_0_4_4, %5) {lib = "add_bf16_vector"} : (memref<4x768xbf16>, memref<4x768xbf16>, memref<4x768xbf16>) -> ()
        aie.objectfifo.release @pipe_4(Consume, 1)
        aie.objectfifo.release @fifo_14(Consume, 1)
        aie.objectfifo.release @fifo_22(Produce, 1)
      }
      aie.end
    } {link_with = "external1.o"}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @pipe_5(Consume, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        %2 = aie.objectfifo.acquire @fifo_14(Consume, 1) : !aie.objectfifosubview<memref<768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<768xbf16>> -> memref<768xbf16>
        affine.for %arg1 = 0 to 4 {
          affine.for %arg2 = 0 to 768 {
            %6 = affine.load %3[%arg2] : memref<768xbf16>
            affine.store %6, %buffer_1_4_5[%arg1, %arg2] : memref<4x768xbf16>
          }
        }
        %4 = aie.objectfifo.acquire @fifo_23(Produce, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        func.call @add_bf16_vector(%1, %buffer_1_4_5, %5) {lib = "add_bf16_vector"} : (memref<4x768xbf16>, memref<4x768xbf16>, memref<4x768xbf16>) -> ()
        aie.objectfifo.release @pipe_5(Consume, 1)
        aie.objectfifo.release @fifo_14(Consume, 1)
        aie.objectfifo.release @fifo_23(Produce, 1)
      }
      aie.end
    } {link_with = "external1.o"}
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @pipe_6(Consume, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        %2 = aie.objectfifo.acquire @fifo_14(Consume, 1) : !aie.objectfifosubview<memref<768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<768xbf16>> -> memref<768xbf16>
        affine.for %arg1 = 0 to 4 {
          affine.for %arg2 = 0 to 768 {
            %6 = affine.load %3[%arg2] : memref<768xbf16>
            affine.store %6, %buffer_2_4_6[%arg1, %arg2] : memref<4x768xbf16>
          }
        }
        %4 = aie.objectfifo.acquire @fifo_25(Produce, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        func.call @add_bf16_vector(%1, %buffer_2_4_6, %5) {lib = "add_bf16_vector"} : (memref<4x768xbf16>, memref<4x768xbf16>, memref<4x768xbf16>) -> ()
        aie.objectfifo.release @pipe_6(Consume, 1)
        aie.objectfifo.release @fifo_14(Consume, 1)
        aie.objectfifo.release @fifo_25(Produce, 1)
      }
      aie.end
    } {link_with = "external1.o"}
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @pipe_7(Consume, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        %2 = aie.objectfifo.acquire @fifo_14(Consume, 1) : !aie.objectfifosubview<memref<768xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<768xbf16>> -> memref<768xbf16>
        affine.for %arg1 = 0 to 4 {
          affine.for %arg2 = 0 to 768 {
            %6 = affine.load %3[%arg2] : memref<768xbf16>
            affine.store %6, %buffer_3_4_7[%arg1, %arg2] : memref<4x768xbf16>
          }
        }
        %4 = aie.objectfifo.acquire @fifo_26(Produce, 1) : !aie.objectfifosubview<memref<4x768xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<4x768xbf16>> -> memref<4x768xbf16>
        func.call @add_bf16_vector(%1, %buffer_3_4_7, %5) {lib = "add_bf16_vector"} : (memref<4x768xbf16>, memref<4x768xbf16>, memref<4x768xbf16>) -> ()
        aie.objectfifo.release @pipe_7(Consume, 1)
        aie.objectfifo.release @fifo_14(Consume, 1)
        aie.objectfifo.release @fifo_26(Produce, 1)
      }
      aie.end
    } {link_with = "external1.o"}
    aiex.runtime_sequence(%arg0: memref<25344xbf16>, %arg1: memref<24576xbf16>, %arg2: memref<768xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 2, 4, 768][0, 3072, 768, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_2} : memref<25344xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 2, 0, 0][1, 2, 4, 768][0, 3072, 768, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_5} : memref<25344xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 4, 0, 0][1, 2, 4, 768][0, 3072, 768, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_8} : memref<25344xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 6, 0, 0][1, 2, 4, 768][0, 3072, 768, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<25344xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 1, 1, 768][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_13} : memref<768xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 24576][1, 1, 1, 768][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @fifo_15} : memref<25344xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 2, 4, 768][0, 3072, 768, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_18} : memref<24576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 2, 0, 0][1, 2, 4, 768][0, 3072, 768, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_21} : memref<24576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 4, 0, 0][1, 2, 4, 768][0, 3072, 768, 1]) {id = 2 : i64, issue_token = true, metadata = @fifo_24} : memref<24576xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 6, 0, 0][1, 2, 4, 768][0, 3072, 768, 1]) {id = 3 : i64, issue_token = true, metadata = @fifo_27} : memref<24576xbf16>
      aiex.npu.dma_wait {symbol = @fifo_18}
      aiex.npu.dma_wait {symbol = @fifo_21}
      aiex.npu.dma_wait {symbol = @fifo_24}
      aiex.npu.dma_wait {symbol = @fifo_27}
      aie.end
    }
  }
}
