module {
  aie.device(npu1_4col) {
    memref.global "public" @fifo_3_cons : memref<1x1x8x768xbf16>
    memref.global "public" @fifo_3 : memref<1x1x8x768xbf16>
    memref.global "public" @fifo_2_cons : memref<8x768xbf16>
    memref.global "public" @fifo_2 : memref<8x768xbf16>
    memref.global "public" @fifo_1_cons : memref<1x1x48x128xbf16>
    memref.global "public" @fifo_1 : memref<1x1x48x128xbf16>
    memref.global "public" @fifo_0_cons : memref<48x128xbf16>
    memref.global "public" @fifo_0 : memref<48x128xbf16>
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %fifo_3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 1 : i32, sym_name = "fifo_3_cons_prod_lock_0"}
    %fifo_3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "fifo_3_cons_cons_lock_0"}
    %fifo_2_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, sym_name = "fifo_2_cons_buff_0"} : memref<8x768xbf16> 
    %fifo_2_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 12288 : i32, sym_name = "fifo_2_cons_buff_1"} : memref<8x768xbf16> 
    %fifo_2_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 2 : i32, sym_name = "fifo_2_cons_prod_lock_0"}
    %fifo_2_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "fifo_2_cons_cons_lock_0"}
    %fifo_2_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "fifo_2_buff_0"} : memref<8x768xbf16> 
    %fifo_2_buff_1 = aie.buffer(%tile_0_2) {address = 13312 : i32, sym_name = "fifo_2_buff_1"} : memref<8x768xbf16> 
    %fifo_2_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "fifo_2_prod_lock_0"}
    %fifo_2_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "fifo_2_cons_lock_0"}
    %fifo_1_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "fifo_1_cons_buff_0"} : memref<1x1x48x128xbf16> 
    %fifo_1_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 12288 : i32, sym_name = "fifo_1_cons_buff_1"} : memref<1x1x48x128xbf16> 
    %fifo_1_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "fifo_1_cons_prod_lock_0"}
    %fifo_1_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "fifo_1_cons_cons_lock_0"}
    %fifo_1_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 1 : i32, sym_name = "fifo_1_prod_lock_0"}
    %fifo_1_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "fifo_1_cons_lock_0"}
    %fifo_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 25600 : i32, sym_name = "fifo_0_cons_buff_0"} : memref<48x128xbf16> 
    %fifo_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 37888 : i32, sym_name = "fifo_0_cons_buff_1"} : memref<48x128xbf16> 
    %fifo_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "fifo_0_cons_prod_lock_0"}
    %fifo_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "fifo_0_cons_cons_lock_0"}
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb26
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb27
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_prod_lock_0, AcquireGreaterEqual, 1)
      %c0_0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1_1 = arith.constant 1 : index
      cf.br ^bb3(%c0_0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb13
      %3 = arith.cmpi slt, %2, %c8 : index
      cf.cond_br %3, ^bb4, ^bb14
    ^bb4:  // pred: ^bb3
      %c0_2 = arith.constant 0 : index
      %c3 = arith.constant 3 : index
      %c1_3 = arith.constant 1 : index
      cf.br ^bb5(%c0_2 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb12
      %5 = arith.cmpi slt, %4, %c3 : index
      cf.cond_br %5, ^bb6, ^bb13
    ^bb6:  // pred: ^bb5
      %c0_4 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1_5 = arith.constant 1 : index
      cf.br ^bb7(%c0_4 : index)
    ^bb7(%6: index):  // 2 preds: ^bb6, ^bb11
      %7 = arith.cmpi slt, %6, %c16 : index
      cf.cond_br %7, ^bb8, ^bb12
    ^bb8:  // pred: ^bb7
      %c0_6 = arith.constant 0 : index
      %c16_7 = arith.constant 16 : index
      %c1_8 = arith.constant 1 : index
      cf.br ^bb9(%c0_6 : index)
    ^bb9(%8: index):  // 2 preds: ^bb8, ^bb10
      %9 = arith.cmpi slt, %8, %c16_7 : index
      cf.cond_br %9, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      %c16_9 = arith.constant 16 : index
      %10 = arith.muli %4, %c16_9 overflow<nsw> : index
      %11 = arith.addi %10, %6 : index
      %c16_10 = arith.constant 16 : index
      %12 = arith.muli %2, %c16_10 overflow<nsw> : index
      %13 = arith.addi %12, %8 : index
      %14 = memref.load %fifo_0_cons_buff_0[%11, %13] : memref<48x128xbf16>
      %c256 = arith.constant 256 : index
      %15 = arith.muli %4, %c256 overflow<nsw> : index
      %c16_11 = arith.constant 16 : index
      %16 = arith.muli %6, %c16_11 overflow<nsw> : index
      %17 = arith.addi %15, %16 : index
      %18 = arith.addi %17, %8 : index
      memref.store %14, %fifo_2_buff_0[%2, %18] : memref<8x768xbf16>
      %19 = arith.addi %8, %c1_8 : index
      cf.br ^bb9(%19 : index)
    ^bb11:  // pred: ^bb9
      %20 = arith.addi %6, %c1_5 : index
      cf.br ^bb7(%20 : index)
    ^bb12:  // pred: ^bb7
      %21 = arith.addi %4, %c1_3 : index
      cf.br ^bb5(%21 : index)
    ^bb13:  // pred: ^bb5
      %22 = arith.addi %2, %c1_1 : index
      cf.br ^bb3(%22 : index)
    ^bb14:  // pred: ^bb3
      aie.use_lock(%fifo_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_prod_lock_0, AcquireGreaterEqual, 1)
      %c0_12 = arith.constant 0 : index
      %c8_13 = arith.constant 8 : index
      %c1_14 = arith.constant 1 : index
      cf.br ^bb15(%c0_12 : index)
    ^bb15(%23: index):  // 2 preds: ^bb14, ^bb25
      %24 = arith.cmpi slt, %23, %c8_13 : index
      cf.cond_br %24, ^bb16, ^bb26
    ^bb16:  // pred: ^bb15
      %c0_15 = arith.constant 0 : index
      %c3_16 = arith.constant 3 : index
      %c1_17 = arith.constant 1 : index
      cf.br ^bb17(%c0_15 : index)
    ^bb17(%25: index):  // 2 preds: ^bb16, ^bb24
      %26 = arith.cmpi slt, %25, %c3_16 : index
      cf.cond_br %26, ^bb18, ^bb25
    ^bb18:  // pred: ^bb17
      %c0_18 = arith.constant 0 : index
      %c16_19 = arith.constant 16 : index
      %c1_20 = arith.constant 1 : index
      cf.br ^bb19(%c0_18 : index)
    ^bb19(%27: index):  // 2 preds: ^bb18, ^bb23
      %28 = arith.cmpi slt, %27, %c16_19 : index
      cf.cond_br %28, ^bb20, ^bb24
    ^bb20:  // pred: ^bb19
      %c0_21 = arith.constant 0 : index
      %c16_22 = arith.constant 16 : index
      %c1_23 = arith.constant 1 : index
      cf.br ^bb21(%c0_21 : index)
    ^bb21(%29: index):  // 2 preds: ^bb20, ^bb22
      %30 = arith.cmpi slt, %29, %c16_22 : index
      cf.cond_br %30, ^bb22, ^bb23
    ^bb22:  // pred: ^bb21
      %c16_24 = arith.constant 16 : index
      %31 = arith.muli %25, %c16_24 overflow<nsw> : index
      %32 = arith.addi %31, %27 : index
      %c16_25 = arith.constant 16 : index
      %33 = arith.muli %23, %c16_25 overflow<nsw> : index
      %34 = arith.addi %33, %29 : index
      %35 = memref.load %fifo_0_cons_buff_1[%32, %34] : memref<48x128xbf16>
      %c256_26 = arith.constant 256 : index
      %36 = arith.muli %25, %c256_26 overflow<nsw> : index
      %c16_27 = arith.constant 16 : index
      %37 = arith.muli %27, %c16_27 overflow<nsw> : index
      %38 = arith.addi %36, %37 : index
      %39 = arith.addi %38, %29 : index
      memref.store %35, %fifo_2_buff_1[%23, %39] : memref<8x768xbf16>
      %40 = arith.addi %29, %c1_23 : index
      cf.br ^bb21(%40 : index)
    ^bb23:  // pred: ^bb21
      %41 = arith.addi %27, %c1_20 : index
      cf.br ^bb19(%41 : index)
    ^bb24:  // pred: ^bb19
      %42 = arith.addi %25, %c1_17 : index
      cf.br ^bb17(%42 : index)
    ^bb25:  // pred: ^bb17
      %43 = arith.addi %23, %c1_14 : index
      cf.br ^bb15(%43 : index)
    ^bb26:  // pred: ^bb15
      aie.use_lock(%fifo_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_lock_0, Release, 1)
      %44 = arith.addi %0, %c2 : index
      cf.br ^bb1(%44 : index)
    ^bb27:  // pred: ^bb1
      aie.use_lock(%fifo_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_prod_lock_0, AcquireGreaterEqual, 1)
      %c0_28 = arith.constant 0 : index
      %c8_29 = arith.constant 8 : index
      %c1_30 = arith.constant 1 : index
      cf.br ^bb28(%c0_28 : index)
    ^bb28(%45: index):  // 2 preds: ^bb27, ^bb38
      %46 = arith.cmpi slt, %45, %c8_29 : index
      cf.cond_br %46, ^bb29, ^bb39
    ^bb29:  // pred: ^bb28
      %c0_31 = arith.constant 0 : index
      %c3_32 = arith.constant 3 : index
      %c1_33 = arith.constant 1 : index
      cf.br ^bb30(%c0_31 : index)
    ^bb30(%47: index):  // 2 preds: ^bb29, ^bb37
      %48 = arith.cmpi slt, %47, %c3_32 : index
      cf.cond_br %48, ^bb31, ^bb38
    ^bb31:  // pred: ^bb30
      %c0_34 = arith.constant 0 : index
      %c16_35 = arith.constant 16 : index
      %c1_36 = arith.constant 1 : index
      cf.br ^bb32(%c0_34 : index)
    ^bb32(%49: index):  // 2 preds: ^bb31, ^bb36
      %50 = arith.cmpi slt, %49, %c16_35 : index
      cf.cond_br %50, ^bb33, ^bb37
    ^bb33:  // pred: ^bb32
      %c0_37 = arith.constant 0 : index
      %c16_38 = arith.constant 16 : index
      %c1_39 = arith.constant 1 : index
      cf.br ^bb34(%c0_37 : index)
    ^bb34(%51: index):  // 2 preds: ^bb33, ^bb35
      %52 = arith.cmpi slt, %51, %c16_38 : index
      cf.cond_br %52, ^bb35, ^bb36
    ^bb35:  // pred: ^bb34
      %c16_40 = arith.constant 16 : index
      %53 = arith.muli %47, %c16_40 overflow<nsw> : index
      %54 = arith.addi %53, %49 : index
      %c16_41 = arith.constant 16 : index
      %55 = arith.muli %45, %c16_41 overflow<nsw> : index
      %56 = arith.addi %55, %51 : index
      %57 = memref.load %fifo_0_cons_buff_0[%54, %56] : memref<48x128xbf16>
      %c256_42 = arith.constant 256 : index
      %58 = arith.muli %47, %c256_42 overflow<nsw> : index
      %c16_43 = arith.constant 16 : index
      %59 = arith.muli %49, %c16_43 overflow<nsw> : index
      %60 = arith.addi %58, %59 : index
      %61 = arith.addi %60, %51 : index
      memref.store %57, %fifo_2_buff_0[%45, %61] : memref<8x768xbf16>
      %62 = arith.addi %51, %c1_39 : index
      cf.br ^bb34(%62 : index)
    ^bb36:  // pred: ^bb34
      %63 = arith.addi %49, %c1_36 : index
      cf.br ^bb32(%63 : index)
    ^bb37:  // pred: ^bb32
      %64 = arith.addi %47, %c1_33 : index
      cf.br ^bb30(%64 : index)
    ^bb38:  // pred: ^bb30
      %65 = arith.addi %45, %c1_30 : index
      cf.br ^bb28(%65 : index)
    ^bb39:  // pred: ^bb28
      aie.use_lock(%fifo_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_lock_0, Release, 1)
      aie.end
    }
    aiex.runtime_sequence(%arg0: memref<6144xbf16>, %arg1: memref<6144xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 48, 128][0, 0, 128, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_1} : memref<6144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 8, 768][0, 0, 768, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_3} : memref<6144xbf16>
      aiex.npu.dma_wait {symbol = @fifo_3}
      aie.end
    }
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_0 : memref<1x1x48x128xbf16>, 0, 6144) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_1 : memref<1x1x48x128xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_0 : memref<1x1x48x128xbf16>, 0, 6144) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_1 : memref<1x1x48x128xbf16>, 0, 6144) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_0_cons_buff_0 : memref<48x128xbf16>, 0, 6144) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_0_cons_buff_1 : memref<48x128xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_2_buff_0 : memref<8x768xbf16>, 0, 6144) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_2_buff_1 : memref<8x768xbf16>, 0, 6144) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @fifo_1(MM2S, 0, 0)
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_2_cons_buff_0 : memref<8x768xbf16>, 0, 6144) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_2_cons_buff_1 : memref<8x768xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_2_cons_buff_0 : memref<8x768xbf16>, 0, 6144) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_2_cons_buff_1 : memref<8x768xbf16>, 0, 6144) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @fifo_3(S2MM, 0, 1)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
