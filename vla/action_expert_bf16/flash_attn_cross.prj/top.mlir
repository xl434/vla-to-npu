module {
  aie.device(npu1_4col) {
    func.func private @fill_zeros_bf16_32_32_vector(memref<32x32xbf16>)
    func.func private @matmul_scalar_bf16_bf16_32x64x32(memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>)
    func.func private @init_softmax(memref<32xbf16>, memref<32xbf16>)
    func.func private @online_softmax(memref<32x32xbf16>, memref<32xbf16>, memref<32xbf16>, memref<32x32xbf16>, memref<32xbf16>, memref<32xbf16>, memref<32xbf16>)
    func.func private @fill_zeros_bf16_32_64_vector(memref<32x64xbf16>)
    func.func private @matmul_scalar_bf16_bf16_32x32x64(memref<32x32xbf16>, memref<32x64xbf16>, memref<32x64xbf16>)
    func.func private @rescale_attn_output(memref<32x64xbf16>, memref<32xbf16>, memref<32x64xbf16>)
    func.func private @scale_attn_output(memref<32x64xbf16>, memref<32xbf16>, memref<32x64xbf16>)
    func.func private @add_bf16_vector(memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>)
    func.func private @matmul_bf16_bf16_32x32x64(memref<32x32xbf16>, memref<32x64xbf16>, memref<32x64xbf16>)
    func.func private @matmul_bf16_bf16_32x64x32(memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>)
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
    aie.objectfifo @score_pipe_0_0_src(%tile_0_5, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @score_pipe_0_0_dst(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>], {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo.link [@score_pipe_0_0_src] -> [@score_pipe_0_0_dst]([] [])
    aie.objectfifo @exp_scale_pipe_0(%tile_0_4, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<32xbf16>> 
    aie.objectfifo @weight_pipe_0_0_src(%tile_0_4, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo @weight_pipe_0_0_dst(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>> 
    aie.objectfifo.link [@weight_pipe_0_0_src] -> [@weight_pipe_0_0_dst]([] [])
    aie.objectfifo @exp_sum_pipe_0(%tile_0_4, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<32xbf16>> 
    aie.objectfifo @o_pipe_0_0_src(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @o_pipe_0_0_dst(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 256>, <size = 4, stride = 4>, <size = 16, stride = 16>, <size = 4, stride = 1>], {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo.link [@o_pipe_0_0_src] -> [@o_pipe_0_0_dst]([] [])
    aie.objectfifo @fifo_0(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @fifo_1(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x64xbf16>> 
    aie.objectfifo @fifo_2(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>], {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<64x32xbf16>> 
    aie.objectfifo @fifo_3(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1x1x64x32xbf16>> 
    aie.objectfifo @fifo_4(%mem_tile_2_1 dimensionsToStream [<size = 4, stride = 512>, <size = 16, stride = 4>, <size = 8, stride = 64>, <size = 4, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @fifo_5(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1x1x32x64xbf16>> 
    aie.objectfifo @fifo_6(%tile_0_3, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>> 
    aie.objectfifo @fifo_7(%mem_tile_3_1, {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<1x1x32x64xbf16>> 
    aie.objectfifo.link [@fifo_1] -> [@fifo_0]([] [])
    aie.objectfifo.link [@fifo_3] -> [@fifo_2]([] [])
    aie.objectfifo.link [@fifo_5] -> [@fifo_4]([] [])
    aie.objectfifo.link [@fifo_6] -> [@fifo_7]([] [])
    %buffer_0_4 = aie.buffer(%tile_0_4) : memref<32xbf16> 
    %buffer_0_4_0 = aie.buffer(%tile_0_4) : memref<32x32xbf16> 
    %buffer_0_3 = aie.buffer(%tile_0_3) : memref<32x64xbf16> 
    %buffer_0_3_1 = aie.buffer(%tile_0_3) : memref<32xbf16> 
    %buffer_0_3_2 = aie.buffer(%tile_0_3) : memref<32x64xbf16> 
    %buffer_0_3_3 = aie.buffer(%tile_0_3) : memref<32xbf16> 
    %buffer_0_2 = aie.buffer(%tile_0_2) : memref<32x32xbf16> 
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @exp_sum_pipe_0(Produce, 1) : !aie.objectfifosubview<memref<32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32xbf16>> -> memref<32xbf16>
        func.call @init_softmax(%buffer_0_4, %1) : (memref<32xbf16>, memref<32xbf16>) -> ()
        affine.for %arg1 = 0 to 4 {
          %2 = aie.objectfifo.acquire @weight_pipe_0_0_src(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          %4 = aie.objectfifo.acquire @exp_scale_pipe_0(Produce, 1) : !aie.objectfifosubview<memref<32xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32xbf16>> -> memref<32xbf16>
          %6 = aie.objectfifo.acquire @score_pipe_0_0_dst(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @online_softmax(%7, %buffer_0_4, %1, %3, %5, %buffer_0_4, %1) : (memref<32x32xbf16>, memref<32xbf16>, memref<32xbf16>, memref<32x32xbf16>, memref<32xbf16>, memref<32xbf16>, memref<32xbf16>) -> ()
          aie.objectfifo.release @score_pipe_0_0_dst(Consume, 1)
          aie.objectfifo.release @exp_scale_pipe_0(Produce, 1)
          aie.objectfifo.release @weight_pipe_0_0_src(Produce, 1)
        } {loop_name = "i", op_name = "S_i_0"}
        aie.objectfifo.release @exp_sum_pipe_0(Produce, 1)
      }
      aie.end
    } {link_with = "external0.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        func.call @fill_zeros_bf16_32_64_vector(%buffer_0_3) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
        affine.for %arg1 = 0 to 4 {
          %4 = aie.objectfifo.acquire @exp_scale_pipe_0(Consume, 1) : !aie.objectfifosubview<memref<32xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32xbf16>> -> memref<32xbf16>
          func.call @rescale_attn_output(%buffer_0_3, %5, %buffer_0_3) : (memref<32x64xbf16>, memref<32xbf16>, memref<32x64xbf16>) -> ()
          %6 = aie.objectfifo.acquire @o_pipe_0_0_dst(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
          func.call @add_bf16_vector(%buffer_0_3, %7, %buffer_0_3) {lib = "add_bf16_vector"} : (memref<32x64xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
          aie.objectfifo.release @exp_scale_pipe_0(Consume, 1)
          aie.objectfifo.release @o_pipe_0_0_dst(Consume, 1)
        } {loop_name = "i", op_name = "S_i_0"}
        %0 = aie.objectfifo.acquire @exp_sum_pipe_0(Consume, 1) : !aie.objectfifosubview<memref<32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32xbf16>> -> memref<32xbf16>
        %2 = aie.objectfifo.acquire @fifo_6(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        func.call @scale_attn_output(%buffer_0_3, %1, %3) : (memref<32x64xbf16>, memref<32xbf16>, memref<32x64xbf16>) -> ()
        aie.objectfifo.release @exp_sum_pipe_0(Consume, 1)
        aie.objectfifo.release @fifo_6(Produce, 1)
      }
      aie.end
    } {link_with = "external1.o"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @weight_pipe_0_0_dst(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
        %2 = aie.objectfifo.acquire @o_pipe_0_0_src(Produce, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        func.call @fill_zeros_bf16_32_64_vector(%3) {lib = "fill_zeros_bf16_32_64_vector"} : (memref<32x64xbf16>) -> ()
        %4 = aie.objectfifo.acquire @fifo_4(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        func.call @matmul_bf16_bf16_32x32x64(%1, %5, %3) : (memref<32x32xbf16>, memref<32x64xbf16>, memref<32x64xbf16>) -> ()
        aie.objectfifo.release @weight_pipe_0_0_dst(Consume, 1)
        aie.objectfifo.release @o_pipe_0_0_src(Produce, 1)
        aie.objectfifo.release @fifo_4(Consume, 1)
      }
      aie.end
    } {link_with = "external2.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
        scf.for %arg1 = %c0 to %c4 step %c1 {
          %2 = aie.objectfifo.acquire @score_pipe_0_0_src(Produce, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @fill_zeros_bf16_32_32_vector(%3) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
          %4 = aie.objectfifo.acquire @fifo_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
          func.call @matmul_bf16_bf16_32x64x32(%1, %5, %3) : (memref<32x64xbf16>, memref<64x32xbf16>, memref<32x32xbf16>) -> ()
          aie.objectfifo.release @score_pipe_0_0_src(Produce, 1)
          aie.objectfifo.release @fifo_2(Consume, 1)
        } {task_nest}
        aie.objectfifo.release @fifo_0(Consume, 1)
      }
      aie.end
    } {link_with = "external3.o"}
    aiex.runtime_sequence(%arg0: memref<10240xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<8192xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 32, 64][0, 2048, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_1} : memref<10240xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 4, 64, 32][0, 32, 128, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_3} : memref<8192xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 2048][4, 1, 32, 64][2048, 2048, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_5} : memref<10240xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 32, 64][0, 2048, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_7} : memref<2048xbf16>
      aiex.npu.dma_wait {symbol = @fifo_7}
      aie.end
    }
  }
}
