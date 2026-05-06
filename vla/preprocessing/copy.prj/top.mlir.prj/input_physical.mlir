module {
  aie.device(npu1_4col) {
    memref.global "public" @fifo_19_cons : memref<1x2x1x32xbf16>
    memref.global "public" @fifo_19 : memref<1x2x1x32xbf16>
    memref.global "public" @fifo_18_cons : memref<1x32xbf16>
    memref.global "public" @fifo_18 : memref<1x32xbf16>
    memref.global "public" @fifo_17_cons : memref<1x32xbf16>
    memref.global "public" @fifo_17 : memref<1x32xbf16>
    memref.global "public" @fifo_16_cons : memref<1x6x1x32xbf16>
    memref.global "public" @fifo_16 : memref<1x6x1x32xbf16>
    memref.global "public" @fifo_15_cons : memref<1x32xbf16>
    memref.global "public" @fifo_15 : memref<1x32xbf16>
    memref.global "public" @fifo_14_cons : memref<1x32xbf16>
    memref.global "public" @fifo_14 : memref<1x32xbf16>
    memref.global "public" @fifo_13_cons : memref<1x32xbf16>
    memref.global "public" @fifo_13 : memref<1x32xbf16>
    memref.global "public" @fifo_12_cons : memref<1x32xbf16>
    memref.global "public" @fifo_12 : memref<1x32xbf16>
    memref.global "public" @fifo_11_cons : memref<1x32xbf16>
    memref.global "public" @fifo_11 : memref<1x32xbf16>
    memref.global "public" @fifo_10_cons : memref<1x32xbf16>
    memref.global "public" @fifo_10 : memref<1x32xbf16>
    memref.global "public" @fifo_9_cons : memref<1x2x1x32xbf16>
    memref.global "public" @fifo_9 : memref<1x2x1x32xbf16>
    memref.global "public" @fifo_8_cons : memref<1x32xbf16>
    memref.global "public" @fifo_8 : memref<1x32xbf16>
    memref.global "public" @fifo_7_cons : memref<1x32xbf16>
    memref.global "public" @fifo_7 : memref<1x32xbf16>
    memref.global "public" @fifo_6_cons : memref<1x6x1x32xbf16>
    memref.global "public" @fifo_6 : memref<1x6x1x32xbf16>
    memref.global "public" @fifo_5_cons : memref<1x32xbf16>
    memref.global "public" @fifo_5 : memref<1x32xbf16>
    memref.global "public" @fifo_4_cons : memref<1x32xbf16>
    memref.global "public" @fifo_4 : memref<1x32xbf16>
    memref.global "public" @fifo_3_cons : memref<1x32xbf16>
    memref.global "public" @fifo_3 : memref<1x32xbf16>
    memref.global "public" @fifo_2_cons : memref<1x32xbf16>
    memref.global "public" @fifo_2 : memref<1x32xbf16>
    memref.global "public" @fifo_1_cons : memref<1x32xbf16>
    memref.global "public" @fifo_1 : memref<1x32xbf16>
    memref.global "public" @fifo_0_cons : memref<1x32xbf16>
    memref.global "public" @fifo_0 : memref<1x32xbf16>
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_2_1 = aie.tile(2, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_3_1 = aie.tile(3, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_1_4 = aie.tile(1, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_1_5 = aie.tile(1, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %fifo_19_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 1 : i32, sym_name = "fifo_19_cons_prod_lock_0"}
    %fifo_19_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "fifo_19_cons_cons_lock_0"}
    %fifo_19_buff_0 = aie.buffer(%mem_tile_3_1) {address = 0 : i32, sym_name = "fifo_19_buff_0"} : memref<1x2x1x32xbf16> 
    %fifo_19_buff_1 = aie.buffer(%mem_tile_3_1) {address = 128 : i32, sym_name = "fifo_19_buff_1"} : memref<1x2x1x32xbf16> 
    %fifo_19_prod_lock_0 = aie.lock(%mem_tile_3_1, 0) {init = 2 : i32, sym_name = "fifo_19_prod_lock_0"}
    %fifo_19_cons_lock_0 = aie.lock(%mem_tile_3_1, 1) {init = 0 : i32, sym_name = "fifo_19_cons_lock_0"}
    %fifo_19_prod_lock_1 = aie.lock(%mem_tile_3_1, 2) {init = 2 : i32, sym_name = "fifo_19_prod_lock_1"}
    %fifo_19_cons_lock_1 = aie.lock(%mem_tile_3_1, 3) {init = 0 : i32, sym_name = "fifo_19_cons_lock_1"}
    %fifo_18_buff_0 = aie.buffer(%tile_1_5) {address = 1024 : i32, sym_name = "fifo_18_buff_0"} : memref<1x32xbf16> 
    %fifo_18_buff_1 = aie.buffer(%tile_1_5) {address = 1088 : i32, sym_name = "fifo_18_buff_1"} : memref<1x32xbf16> 
    %fifo_18_prod_lock_0 = aie.lock(%tile_1_5, 2) {init = 2 : i32, sym_name = "fifo_18_prod_lock_0"}
    %fifo_18_cons_lock_0 = aie.lock(%tile_1_5, 3) {init = 0 : i32, sym_name = "fifo_18_cons_lock_0"}
    %fifo_17_buff_0 = aie.buffer(%tile_1_4) {address = 1024 : i32, sym_name = "fifo_17_buff_0"} : memref<1x32xbf16> 
    %fifo_17_buff_1 = aie.buffer(%tile_1_4) {address = 1088 : i32, sym_name = "fifo_17_buff_1"} : memref<1x32xbf16> 
    %fifo_17_prod_lock_0 = aie.lock(%tile_1_4, 2) {init = 2 : i32, sym_name = "fifo_17_prod_lock_0"}
    %fifo_17_cons_lock_0 = aie.lock(%tile_1_4, 3) {init = 0 : i32, sym_name = "fifo_17_cons_lock_0"}
    %fifo_16_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 1 : i32, sym_name = "fifo_16_cons_prod_lock_0"}
    %fifo_16_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "fifo_16_cons_cons_lock_0"}
    %fifo_16_buff_0 = aie.buffer(%mem_tile_2_1) {address = 0 : i32, sym_name = "fifo_16_buff_0"} : memref<1x6x1x32xbf16> 
    %fifo_16_buff_1 = aie.buffer(%mem_tile_2_1) {address = 384 : i32, sym_name = "fifo_16_buff_1"} : memref<1x6x1x32xbf16> 
    %fifo_16_prod_lock_0 = aie.lock(%mem_tile_2_1, 0) {init = 2 : i32, sym_name = "fifo_16_prod_lock_0"}
    %fifo_16_cons_lock_0 = aie.lock(%mem_tile_2_1, 1) {init = 0 : i32, sym_name = "fifo_16_cons_lock_0"}
    %fifo_16_prod_lock_1 = aie.lock(%mem_tile_2_1, 2) {init = 2 : i32, sym_name = "fifo_16_prod_lock_1"}
    %fifo_16_cons_lock_1 = aie.lock(%mem_tile_2_1, 3) {init = 0 : i32, sym_name = "fifo_16_cons_lock_1"}
    %fifo_16_prod_lock_2 = aie.lock(%mem_tile_2_1, 4) {init = 2 : i32, sym_name = "fifo_16_prod_lock_2"}
    %fifo_16_cons_lock_2 = aie.lock(%mem_tile_2_1, 5) {init = 0 : i32, sym_name = "fifo_16_cons_lock_2"}
    %fifo_16_prod_lock_3 = aie.lock(%mem_tile_2_1, 6) {init = 2 : i32, sym_name = "fifo_16_prod_lock_3"}
    %fifo_16_cons_lock_3 = aie.lock(%mem_tile_2_1, 7) {init = 0 : i32, sym_name = "fifo_16_cons_lock_3"}
    %fifo_16_prod_lock_4 = aie.lock(%mem_tile_2_1, 8) {init = 2 : i32, sym_name = "fifo_16_prod_lock_4"}
    %fifo_16_cons_lock_4 = aie.lock(%mem_tile_2_1, 9) {init = 0 : i32, sym_name = "fifo_16_cons_lock_4"}
    %fifo_16_prod_lock_5 = aie.lock(%mem_tile_2_1, 10) {init = 2 : i32, sym_name = "fifo_16_prod_lock_5"}
    %fifo_16_cons_lock_5 = aie.lock(%mem_tile_2_1, 11) {init = 0 : i32, sym_name = "fifo_16_cons_lock_5"}
    %fifo_15_buff_0 = aie.buffer(%tile_1_3) {address = 1024 : i32, sym_name = "fifo_15_buff_0"} : memref<1x32xbf16> 
    %fifo_15_buff_1 = aie.buffer(%tile_1_3) {address = 1088 : i32, sym_name = "fifo_15_buff_1"} : memref<1x32xbf16> 
    %fifo_15_prod_lock_0 = aie.lock(%tile_1_3, 2) {init = 2 : i32, sym_name = "fifo_15_prod_lock_0"}
    %fifo_15_cons_lock_0 = aie.lock(%tile_1_3, 3) {init = 0 : i32, sym_name = "fifo_15_cons_lock_0"}
    %fifo_14_buff_0 = aie.buffer(%tile_1_2) {address = 1024 : i32, sym_name = "fifo_14_buff_0"} : memref<1x32xbf16> 
    %fifo_14_buff_1 = aie.buffer(%tile_1_2) {address = 1088 : i32, sym_name = "fifo_14_buff_1"} : memref<1x32xbf16> 
    %fifo_14_prod_lock_0 = aie.lock(%tile_1_2, 2) {init = 2 : i32, sym_name = "fifo_14_prod_lock_0"}
    %fifo_14_cons_lock_0 = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "fifo_14_cons_lock_0"}
    %fifo_13_buff_0 = aie.buffer(%tile_0_5) {address = 1024 : i32, sym_name = "fifo_13_buff_0"} : memref<1x32xbf16> 
    %fifo_13_buff_1 = aie.buffer(%tile_0_5) {address = 1088 : i32, sym_name = "fifo_13_buff_1"} : memref<1x32xbf16> 
    %fifo_13_prod_lock_0 = aie.lock(%tile_0_5, 2) {init = 2 : i32, sym_name = "fifo_13_prod_lock_0"}
    %fifo_13_cons_lock_0 = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "fifo_13_cons_lock_0"}
    %fifo_12_buff_0 = aie.buffer(%tile_0_4) {address = 1024 : i32, sym_name = "fifo_12_buff_0"} : memref<1x32xbf16> 
    %fifo_12_buff_1 = aie.buffer(%tile_0_4) {address = 1088 : i32, sym_name = "fifo_12_buff_1"} : memref<1x32xbf16> 
    %fifo_12_prod_lock_0 = aie.lock(%tile_0_4, 2) {init = 2 : i32, sym_name = "fifo_12_prod_lock_0"}
    %fifo_12_cons_lock_0 = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "fifo_12_cons_lock_0"}
    %fifo_11_buff_0 = aie.buffer(%tile_0_3) {address = 1024 : i32, sym_name = "fifo_11_buff_0"} : memref<1x32xbf16> 
    %fifo_11_buff_1 = aie.buffer(%tile_0_3) {address = 1088 : i32, sym_name = "fifo_11_buff_1"} : memref<1x32xbf16> 
    %fifo_11_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "fifo_11_prod_lock_0"}
    %fifo_11_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "fifo_11_cons_lock_0"}
    %fifo_10_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "fifo_10_buff_0"} : memref<1x32xbf16> 
    %fifo_10_buff_1 = aie.buffer(%tile_0_2) {address = 1088 : i32, sym_name = "fifo_10_buff_1"} : memref<1x32xbf16> 
    %fifo_10_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "fifo_10_prod_lock_0"}
    %fifo_10_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "fifo_10_cons_lock_0"}
    %fifo_9_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, sym_name = "fifo_9_cons_buff_0"} : memref<1x2x1x32xbf16> 
    %fifo_9_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 128 : i32, sym_name = "fifo_9_cons_buff_1"} : memref<1x2x1x32xbf16> 
    %fifo_9_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 2 : i32, sym_name = "fifo_9_cons_prod_lock_0"}
    %fifo_9_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "fifo_9_cons_cons_lock_0"}
    %fifo_9_cons_prod_lock_1 = aie.lock(%mem_tile_1_1, 2) {init = 2 : i32, sym_name = "fifo_9_cons_prod_lock_1"}
    %fifo_9_cons_cons_lock_1 = aie.lock(%mem_tile_1_1, 3) {init = 0 : i32, sym_name = "fifo_9_cons_cons_lock_1"}
    %fifo_9_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 1 : i32, sym_name = "fifo_9_prod_lock_0"}
    %fifo_9_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "fifo_9_cons_lock_0"}
    %fifo_8_cons_buff_0 = aie.buffer(%tile_1_5) {address = 1152 : i32, sym_name = "fifo_8_cons_buff_0"} : memref<1x32xbf16> 
    %fifo_8_cons_buff_1 = aie.buffer(%tile_1_5) {address = 1216 : i32, sym_name = "fifo_8_cons_buff_1"} : memref<1x32xbf16> 
    %fifo_8_cons_prod_lock_0 = aie.lock(%tile_1_5, 0) {init = 2 : i32, sym_name = "fifo_8_cons_prod_lock_0"}
    %fifo_8_cons_cons_lock_0 = aie.lock(%tile_1_5, 1) {init = 0 : i32, sym_name = "fifo_8_cons_cons_lock_0"}
    %fifo_7_cons_buff_0 = aie.buffer(%tile_1_4) {address = 1152 : i32, sym_name = "fifo_7_cons_buff_0"} : memref<1x32xbf16> 
    %fifo_7_cons_buff_1 = aie.buffer(%tile_1_4) {address = 1216 : i32, sym_name = "fifo_7_cons_buff_1"} : memref<1x32xbf16> 
    %fifo_7_cons_prod_lock_0 = aie.lock(%tile_1_4, 0) {init = 2 : i32, sym_name = "fifo_7_cons_prod_lock_0"}
    %fifo_7_cons_cons_lock_0 = aie.lock(%tile_1_4, 1) {init = 0 : i32, sym_name = "fifo_7_cons_cons_lock_0"}
    %fifo_6_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "fifo_6_cons_buff_0"} : memref<1x6x1x32xbf16> 
    %fifo_6_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 384 : i32, sym_name = "fifo_6_cons_buff_1"} : memref<1x6x1x32xbf16> 
    %fifo_6_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "fifo_6_cons_prod_lock_0"}
    %fifo_6_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "fifo_6_cons_cons_lock_0"}
    %fifo_6_cons_prod_lock_1 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "fifo_6_cons_prod_lock_1"}
    %fifo_6_cons_cons_lock_1 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "fifo_6_cons_cons_lock_1"}
    %fifo_6_cons_prod_lock_2 = aie.lock(%mem_tile_0_1, 4) {init = 2 : i32, sym_name = "fifo_6_cons_prod_lock_2"}
    %fifo_6_cons_cons_lock_2 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "fifo_6_cons_cons_lock_2"}
    %fifo_6_cons_prod_lock_3 = aie.lock(%mem_tile_0_1, 6) {init = 2 : i32, sym_name = "fifo_6_cons_prod_lock_3"}
    %fifo_6_cons_cons_lock_3 = aie.lock(%mem_tile_0_1, 7) {init = 0 : i32, sym_name = "fifo_6_cons_cons_lock_3"}
    %fifo_6_cons_prod_lock_4 = aie.lock(%mem_tile_0_1, 8) {init = 2 : i32, sym_name = "fifo_6_cons_prod_lock_4"}
    %fifo_6_cons_cons_lock_4 = aie.lock(%mem_tile_0_1, 9) {init = 0 : i32, sym_name = "fifo_6_cons_cons_lock_4"}
    %fifo_6_cons_prod_lock_5 = aie.lock(%mem_tile_0_1, 10) {init = 2 : i32, sym_name = "fifo_6_cons_prod_lock_5"}
    %fifo_6_cons_cons_lock_5 = aie.lock(%mem_tile_0_1, 11) {init = 0 : i32, sym_name = "fifo_6_cons_cons_lock_5"}
    %fifo_6_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 1 : i32, sym_name = "fifo_6_prod_lock_0"}
    %fifo_6_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "fifo_6_cons_lock_0"}
    %fifo_5_cons_buff_0 = aie.buffer(%tile_1_3) {address = 1152 : i32, sym_name = "fifo_5_cons_buff_0"} : memref<1x32xbf16> 
    %fifo_5_cons_buff_1 = aie.buffer(%tile_1_3) {address = 1216 : i32, sym_name = "fifo_5_cons_buff_1"} : memref<1x32xbf16> 
    %fifo_5_cons_prod_lock_0 = aie.lock(%tile_1_3, 0) {init = 2 : i32, sym_name = "fifo_5_cons_prod_lock_0"}
    %fifo_5_cons_cons_lock_0 = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "fifo_5_cons_cons_lock_0"}
    %fifo_4_cons_buff_0 = aie.buffer(%tile_1_2) {address = 1152 : i32, sym_name = "fifo_4_cons_buff_0"} : memref<1x32xbf16> 
    %fifo_4_cons_buff_1 = aie.buffer(%tile_1_2) {address = 1216 : i32, sym_name = "fifo_4_cons_buff_1"} : memref<1x32xbf16> 
    %fifo_4_cons_prod_lock_0 = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "fifo_4_cons_prod_lock_0"}
    %fifo_4_cons_cons_lock_0 = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "fifo_4_cons_cons_lock_0"}
    %fifo_3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 1152 : i32, sym_name = "fifo_3_cons_buff_0"} : memref<1x32xbf16> 
    %fifo_3_cons_buff_1 = aie.buffer(%tile_0_5) {address = 1216 : i32, sym_name = "fifo_3_cons_buff_1"} : memref<1x32xbf16> 
    %fifo_3_cons_prod_lock_0 = aie.lock(%tile_0_5, 0) {init = 2 : i32, sym_name = "fifo_3_cons_prod_lock_0"}
    %fifo_3_cons_cons_lock_0 = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "fifo_3_cons_cons_lock_0"}
    %fifo_2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 1152 : i32, sym_name = "fifo_2_cons_buff_0"} : memref<1x32xbf16> 
    %fifo_2_cons_buff_1 = aie.buffer(%tile_0_4) {address = 1216 : i32, sym_name = "fifo_2_cons_buff_1"} : memref<1x32xbf16> 
    %fifo_2_cons_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "fifo_2_cons_prod_lock_0"}
    %fifo_2_cons_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "fifo_2_cons_cons_lock_0"}
    %fifo_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 1152 : i32, sym_name = "fifo_1_cons_buff_0"} : memref<1x32xbf16> 
    %fifo_1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 1216 : i32, sym_name = "fifo_1_cons_buff_1"} : memref<1x32xbf16> 
    %fifo_1_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "fifo_1_cons_prod_lock_0"}
    %fifo_1_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "fifo_1_cons_cons_lock_0"}
    %fifo_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1152 : i32, sym_name = "fifo_0_cons_buff_0"} : memref<1x32xbf16> 
    %fifo_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 1216 : i32, sym_name = "fifo_0_cons_buff_1"} : memref<1x32xbf16> 
    %fifo_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "fifo_0_cons_prod_lock_0"}
    %fifo_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "fifo_0_cons_cons_lock_0"}
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<DMA : 0, North : 1>
      aie.connect<DMA : 1, North : 0>
      aie.connect<DMA : 2, North : 2>
      aie.connect<DMA : 3, North : 5>
      aie.connect<DMA : 4, North : 4>
      aie.connect<DMA : 5, North : 3>
      aie.connect<South : 5, DMA : 0>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 0, North : 5>
      aie.connect<South : 2, North : 0>
      aie.connect<South : 5, North : 4>
      aie.connect<South : 4, East : 0>
      aie.connect<South : 3, East : 1>
      aie.connect<DMA : 0, East : 3>
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<South : 5, DMA : 0>
      aie.connect<South : 0, North : 1>
      aie.connect<South : 4, North : 2>
      aie.connect<DMA : 0, East : 0>
      aie.connect<North : 1, East : 2>
    }
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 2, North : 4>
      aie.connect<DMA : 0, South : 1>
    }
    %switchbox_0_5 = aie.switchbox(%tile_0_5) {
      aie.connect<South : 4, DMA : 0>
      aie.connect<DMA : 0, East : 3>
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<West : 0, DMA : 0>
      aie.connect<West : 1, North : 5>
      aie.connect<South : 1, North : 1>
      aie.connect<South : 5, North : 3>
      aie.connect<West : 3, East : 2>
      aie.connect<DMA : 0, South : 3>
      aie.connect<North : 3, South : 1>
    }
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<South : 5, DMA : 0>
      aie.connect<South : 1, North : 3>
      aie.connect<South : 3, North : 2>
      aie.connect<West : 0, East : 2>
      aie.connect<West : 2, East : 0>
      aie.connect<DMA : 0, South : 3>
      aie.connect<North : 1, East : 1>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 5>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<DMA : 0, North : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<South : 1, DMA : 0>
      aie.connect<North : 3, South : 3>
      aie.connect<North : 1, South : 1>
    }
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<South : 2, North : 3>
      aie.connect<DMA : 0, South : 1>
    }
    %switchbox_1_5 = aie.switchbox(%tile_1_5) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<West : 3, East : 1>
      aie.connect<DMA : 0, East : 2>
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<North : 3, East : 0>
      aie.connect<North : 1, East : 3>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_2_1 = aie.switchbox(%mem_tile_2_1) {
      aie.connect<North : 0, DMA : 0>
      aie.connect<North : 1, DMA : 1>
      aie.connect<North : 3, DMA : 2>
      aie.connect<North : 2, DMA : 3>
      aie.connect<South : 5, DMA : 4>
      aie.connect<South : 0, DMA : 5>
      aie.connect<DMA : 0, South : 2>
    }
    %tile_2_2 = aie.tile(2, 2)
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      aie.connect<West : 2, South : 0>
      aie.connect<North : 0, South : 1>
      aie.connect<North : 2, South : 3>
      aie.connect<North : 1, South : 2>
    }
    %tile_2_3 = aie.tile(2, 3)
    %switchbox_2_3 = aie.switchbox(%tile_2_3) {
      aie.connect<West : 2, South : 0>
      aie.connect<West : 0, South : 2>
      aie.connect<North : 2, South : 1>
      aie.connect<West : 1, East : 2>
      aie.connect<North : 1, East : 0>
    }
    %tile_2_4 = aie.tile(2, 4)
    %switchbox_2_4 = aie.switchbox(%tile_2_4) {
      aie.connect<North : 3, South : 2>
      aie.connect<North : 0, South : 1>
    }
    %tile_2_5 = aie.tile(2, 5)
    %switchbox_2_5 = aie.switchbox(%tile_2_5) {
      aie.connect<West : 1, South : 3>
      aie.connect<West : 2, South : 0>
    }
    %switchbox_2_0 = aie.switchbox(%shim_noc_tile_2_0) {
      aie.connect<West : 0, North : 5>
      aie.connect<West : 3, North : 0>
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_2_0 = aie.shim_mux(%shim_noc_tile_2_0) {
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_3_1 = aie.switchbox(%mem_tile_3_1) {
      aie.connect<North : 0, DMA : 0>
      aie.connect<North : 3, DMA : 1>
      aie.connect<DMA : 0, South : 2>
    }
    %tile_3_2 = aie.tile(3, 2)
    %switchbox_3_2 = aie.switchbox(%tile_3_2) {
      aie.connect<North : 0, South : 0>
      aie.connect<North : 1, South : 3>
    }
    %tile_3_3 = aie.tile(3, 3)
    %switchbox_3_3 = aie.switchbox(%tile_3_3) {
      aie.connect<West : 2, South : 0>
      aie.connect<West : 0, South : 1>
    }
    %switchbox_3_0 = aie.switchbox(%shim_noc_tile_3_0) {
      aie.connect<North : 2, South : 2>
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(South : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %0)
      }
    }
    %shim_mux_3_0 = aie.shim_mux(%shim_noc_tile_3_0) {
      aie.connect<North : 2, DMA : 0>
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview = memref.subview %fifo_0_cons_buff_0[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_10_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_0 = memref.subview %fifo_10_buff_0[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview, %subview_0 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_10_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview_1 = memref.subview %fifo_0_cons_buff_1[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_10_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_2 = memref.subview %fifo_10_buff_1[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview_1, %subview_2 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_10_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%fifo_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview_3 = memref.subview %fifo_0_cons_buff_0[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_10_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_4 = memref.subview %fifo_10_buff_0[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview_3, %subview_4 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_10_cons_lock_0, Release, 1)
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview = memref.subview %fifo_1_cons_buff_0[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_11_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_0 = memref.subview %fifo_11_buff_0[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview, %subview_0 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_11_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview_1 = memref.subview %fifo_1_cons_buff_1[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_11_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_2 = memref.subview %fifo_11_buff_1[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview_1, %subview_2 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_11_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%fifo_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview_3 = memref.subview %fifo_1_cons_buff_0[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_11_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_4 = memref.subview %fifo_11_buff_0[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview_3, %subview_4 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_11_cons_lock_0, Release, 1)
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview = memref.subview %fifo_2_cons_buff_0[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_12_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_0 = memref.subview %fifo_12_buff_0[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview, %subview_0 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_12_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview_1 = memref.subview %fifo_2_cons_buff_1[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_12_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_2 = memref.subview %fifo_12_buff_1[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview_1, %subview_2 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_12_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview_3 = memref.subview %fifo_2_cons_buff_0[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_12_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_4 = memref.subview %fifo_12_buff_0[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview_3, %subview_4 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_12_cons_lock_0, Release, 1)
      aie.end
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview = memref.subview %fifo_3_cons_buff_0[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_13_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_0 = memref.subview %fifo_13_buff_0[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview, %subview_0 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_13_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview_1 = memref.subview %fifo_3_cons_buff_1[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_13_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_2 = memref.subview %fifo_13_buff_1[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview_1, %subview_2 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_13_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%fifo_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview_3 = memref.subview %fifo_3_cons_buff_0[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_13_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_4 = memref.subview %fifo_13_buff_0[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview_3, %subview_4 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_13_cons_lock_0, Release, 1)
      aie.end
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview = memref.subview %fifo_4_cons_buff_0[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_14_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_0 = memref.subview %fifo_14_buff_0[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview, %subview_0 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_14_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview_1 = memref.subview %fifo_4_cons_buff_1[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_14_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_2 = memref.subview %fifo_14_buff_1[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview_1, %subview_2 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_14_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview_3 = memref.subview %fifo_4_cons_buff_0[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_14_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_4 = memref.subview %fifo_14_buff_0[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview_3, %subview_4 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_14_cons_lock_0, Release, 1)
      aie.end
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview = memref.subview %fifo_5_cons_buff_0[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_15_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_0 = memref.subview %fifo_15_buff_0[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview, %subview_0 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_15_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview_1 = memref.subview %fifo_5_cons_buff_1[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_15_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_2 = memref.subview %fifo_15_buff_1[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview_1, %subview_2 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_15_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%fifo_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview_3 = memref.subview %fifo_5_cons_buff_0[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_15_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_4 = memref.subview %fifo_15_buff_0[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview_3, %subview_4 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_15_cons_lock_0, Release, 1)
      aie.end
    }
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview = memref.subview %fifo_7_cons_buff_0[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_17_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_0 = memref.subview %fifo_17_buff_0[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview, %subview_0 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_17_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview_1 = memref.subview %fifo_7_cons_buff_1[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_17_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_2 = memref.subview %fifo_17_buff_1[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview_1, %subview_2 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_17_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%fifo_7_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview_3 = memref.subview %fifo_7_cons_buff_0[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_17_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_4 = memref.subview %fifo_17_buff_0[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview_3, %subview_4 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_7_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_17_cons_lock_0, Release, 1)
      aie.end
    }
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_8_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview = memref.subview %fifo_8_cons_buff_0[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_18_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_0 = memref.subview %fifo_18_buff_0[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview, %subview_0 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_8_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_18_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_8_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview_1 = memref.subview %fifo_8_cons_buff_1[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_18_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_2 = memref.subview %fifo_18_buff_1[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview_1, %subview_2 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_8_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_18_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%fifo_8_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %subview_3 = memref.subview %fifo_8_cons_buff_0[0, 0] [1, 32] [1, 1] {from = "local_A"} : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_18_prod_lock_0, AcquireGreaterEqual, 1)
      %subview_4 = memref.subview %fifo_18_buff_0[0, 0] [1, 32] [1, 1] : memref<1x32xbf16> to memref<32xbf16, strided<[1]>>
      memref.copy %subview_3, %subview_4 {to = "local_C"} : memref<32xbf16, strided<[1]>> to memref<32xbf16, strided<[1]>>
      aie.use_lock(%fifo_8_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_18_cons_lock_0, Release, 1)
      aie.end
    }
    aiex.runtime_sequence(%arg0: memref<256xbf16>, %arg1: memref<256xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 6, 1, 32][0, 32, 32, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_6} : memref<256xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 6, 0, 0][1, 2, 1, 32][0, 32, 32, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_9} : memref<256xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 6, 1, 32][0, 32, 256, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_16} : memref<256xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 6, 0, 0][1, 2, 1, 32][0, 32, 256, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_19} : memref<256xbf16>
      aiex.npu.dma_wait {symbol = @fifo_16}
      aiex.npu.dma_wait {symbol = @fifo_19}
      aie.end
    }
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<1x6x1x32xbf16>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<1x6x1x32xbf16>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_6_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<1x6x1x32xbf16>, 32, 32) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_6_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<1x6x1x32xbf16>, 32, 32) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%fifo_6_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<1x6x1x32xbf16>, 64, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%fifo_6_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<1x6x1x32xbf16>, 64, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 3, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%fifo_6_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<1x6x1x32xbf16>, 96, 32) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%fifo_6_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<1x6x1x32xbf16>, 96, 32) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(MM2S, 4, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%fifo_6_cons_cons_lock_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<1x6x1x32xbf16>, 128, 32) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_4, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%fifo_6_cons_cons_lock_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<1x6x1x32xbf16>, 128, 32) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_4, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 5, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%fifo_6_cons_cons_lock_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<1x6x1x32xbf16>, 160, 32) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_5, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%fifo_6_cons_cons_lock_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<1x6x1x32xbf16>, 160, 32) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_5, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(S2MM, 0, ^bb19, ^bb31)
    ^bb19:  // 2 preds: ^bb18, ^bb30
      aie.use_lock(%fifo_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<1x6x1x32xbf16>, 0, 32) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%fifo_6_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<1x6x1x32xbf16>, 32, 32) {bd_id = 7 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      aie.use_lock(%fifo_6_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<1x6x1x32xbf16>, 64, 32) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb22
    ^bb22:  // pred: ^bb21
      aie.use_lock(%fifo_6_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<1x6x1x32xbf16>, 96, 32) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%fifo_6_cons_prod_lock_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<1x6x1x32xbf16>, 128, 32) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_4, Release, 1)
      aie.next_bd ^bb24
    ^bb24:  // pred: ^bb23
      aie.use_lock(%fifo_6_cons_prod_lock_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<1x6x1x32xbf16>, 160, 32) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_5, Release, 1)
      aie.next_bd ^bb25
    ^bb25:  // pred: ^bb24
      aie.use_lock(%fifo_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<1x6x1x32xbf16>, 0, 32) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%fifo_6_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<1x6x1x32xbf16>, 32, 32) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%fifo_6_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<1x6x1x32xbf16>, 64, 32) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%fifo_6_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<1x6x1x32xbf16>, 96, 32) {bd_id = 15 : i32, next_bd_id = 16 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%fifo_6_cons_prod_lock_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<1x6x1x32xbf16>, 128, 32) {bd_id = 16 : i32, next_bd_id = 17 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_4, Release, 1)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      aie.use_lock(%fifo_6_cons_prod_lock_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<1x6x1x32xbf16>, 160, 32) {bd_id = 17 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_5, Release, 1)
      aie.next_bd ^bb19
    ^bb31:  // pred: ^bb18
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_0_cons_buff_0 : memref<1x32xbf16>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_0_cons_buff_1 : memref<1x32xbf16>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_10_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_10_buff_0 : memref<1x32xbf16>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_10_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_10_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_10_buff_1 : memref<1x32xbf16>, 0, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_10_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_0 : memref<1x32xbf16>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_1 : memref<1x32xbf16>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_11_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_11_buff_0 : memref<1x32xbf16>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_11_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_11_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_11_buff_1 : memref<1x32xbf16>, 0, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_11_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_2_cons_buff_0 : memref<1x32xbf16>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_2_cons_buff_1 : memref<1x32xbf16>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_12_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_12_buff_0 : memref<1x32xbf16>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_12_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_12_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_12_buff_1 : memref<1x32xbf16>, 0, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_12_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_3_cons_buff_0 : memref<1x32xbf16>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_3_cons_buff_1 : memref<1x32xbf16>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_13_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_13_buff_0 : memref<1x32xbf16>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_13_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_13_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_13_buff_1 : memref<1x32xbf16>, 0, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_13_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_cons_buff_0 : memref<1x32xbf16>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_cons_buff_1 : memref<1x32xbf16>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_14_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_14_buff_0 : memref<1x32xbf16>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_14_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_14_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_14_buff_1 : memref<1x32xbf16>, 0, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_14_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_5_cons_buff_0 : memref<1x32xbf16>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_5_cons_buff_1 : memref<1x32xbf16>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_15_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_15_buff_0 : memref<1x32xbf16>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_15_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_15_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_15_buff_1 : memref<1x32xbf16>, 0, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_15_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @fifo_6(MM2S, 0, 0)
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_9_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_9_cons_buff_0 : memref<1x2x1x32xbf16>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_9_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_9_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_9_cons_buff_1 : memref<1x2x1x32xbf16>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_9_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_9_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_9_cons_buff_0 : memref<1x2x1x32xbf16>, 32, 32) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%fifo_9_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_9_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_9_cons_buff_1 : memref<1x2x1x32xbf16>, 32, 32) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%fifo_9_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb7, ^bb11)
    ^bb7:  // 2 preds: ^bb6, ^bb10
      aie.use_lock(%fifo_9_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_9_cons_buff_0 : memref<1x2x1x32xbf16>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_9_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%fifo_9_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_9_cons_buff_0 : memref<1x2x1x32xbf16>, 32, 32) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%fifo_9_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb9
    ^bb9:  // pred: ^bb8
      aie.use_lock(%fifo_9_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_9_cons_buff_1 : memref<1x2x1x32xbf16>, 0, 32) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%fifo_9_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%fifo_9_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_9_cons_buff_1 : memref<1x2x1x32xbf16>, 32, 32) {bd_id = 5 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_9_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb7
    ^bb11:  // pred: ^bb6
      aie.end
    }
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_7_cons_buff_0 : memref<1x32xbf16>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_7_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_7_cons_buff_1 : memref<1x32xbf16>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_7_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_17_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_17_buff_0 : memref<1x32xbf16>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_17_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_17_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_17_buff_1 : memref<1x32xbf16>, 0, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_17_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_8_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_8_cons_buff_0 : memref<1x32xbf16>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_8_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_8_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_8_cons_buff_1 : memref<1x32xbf16>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_8_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_18_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_18_buff_0 : memref<1x32xbf16>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_18_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_18_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_18_buff_1 : memref<1x32xbf16>, 0, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_18_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @fifo_9(MM2S, 0, 1)
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_16_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_0 : memref<1x6x1x32xbf16>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_16_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_16_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_1 : memref<1x6x1x32xbf16>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_16_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_16_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_0 : memref<1x6x1x32xbf16>, 32, 32) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%fifo_16_cons_lock_1, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_16_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_1 : memref<1x6x1x32xbf16>, 32, 32) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%fifo_16_cons_lock_1, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 2, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%fifo_16_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_0 : memref<1x6x1x32xbf16>, 64, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_16_cons_lock_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%fifo_16_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_1 : memref<1x6x1x32xbf16>, 64, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_16_cons_lock_2, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 3, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%fifo_16_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_0 : memref<1x6x1x32xbf16>, 96, 32) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%fifo_16_cons_lock_3, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%fifo_16_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_1 : memref<1x6x1x32xbf16>, 96, 32) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%fifo_16_cons_lock_3, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 4, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%fifo_16_prod_lock_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_0 : memref<1x6x1x32xbf16>, 128, 32) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%fifo_16_cons_lock_4, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%fifo_16_prod_lock_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_1 : memref<1x6x1x32xbf16>, 128, 32) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%fifo_16_cons_lock_4, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 5, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%fifo_16_prod_lock_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_0 : memref<1x6x1x32xbf16>, 160, 32) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%fifo_16_cons_lock_5, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%fifo_16_prod_lock_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_1 : memref<1x6x1x32xbf16>, 160, 32) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%fifo_16_cons_lock_5, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(MM2S, 0, ^bb19, ^bb31)
    ^bb19:  // 2 preds: ^bb18, ^bb30
      aie.use_lock(%fifo_16_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_0 : memref<1x6x1x32xbf16>, 0, 32) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%fifo_16_prod_lock_0, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%fifo_16_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_0 : memref<1x6x1x32xbf16>, 32, 32) {bd_id = 7 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%fifo_16_prod_lock_1, Release, 1)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      aie.use_lock(%fifo_16_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_0 : memref<1x6x1x32xbf16>, 64, 32) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%fifo_16_prod_lock_2, Release, 1)
      aie.next_bd ^bb22
    ^bb22:  // pred: ^bb21
      aie.use_lock(%fifo_16_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_0 : memref<1x6x1x32xbf16>, 96, 32) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%fifo_16_prod_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%fifo_16_cons_lock_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_0 : memref<1x6x1x32xbf16>, 128, 32) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%fifo_16_prod_lock_4, Release, 1)
      aie.next_bd ^bb24
    ^bb24:  // pred: ^bb23
      aie.use_lock(%fifo_16_cons_lock_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_0 : memref<1x6x1x32xbf16>, 160, 32) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%fifo_16_prod_lock_5, Release, 1)
      aie.next_bd ^bb25
    ^bb25:  // pred: ^bb24
      aie.use_lock(%fifo_16_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_1 : memref<1x6x1x32xbf16>, 0, 32) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%fifo_16_prod_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%fifo_16_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_1 : memref<1x6x1x32xbf16>, 32, 32) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%fifo_16_prod_lock_1, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%fifo_16_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_1 : memref<1x6x1x32xbf16>, 64, 32) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%fifo_16_prod_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%fifo_16_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_1 : memref<1x6x1x32xbf16>, 96, 32) {bd_id = 15 : i32, next_bd_id = 16 : i32}
      aie.use_lock(%fifo_16_prod_lock_3, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%fifo_16_cons_lock_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_1 : memref<1x6x1x32xbf16>, 128, 32) {bd_id = 16 : i32, next_bd_id = 17 : i32}
      aie.use_lock(%fifo_16_prod_lock_4, Release, 1)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      aie.use_lock(%fifo_16_cons_lock_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_16_buff_1 : memref<1x6x1x32xbf16>, 160, 32) {bd_id = 17 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%fifo_16_prod_lock_5, Release, 1)
      aie.next_bd ^bb19
    ^bb31:  // pred: ^bb18
      aie.end
    }
    aie.shim_dma_allocation @fifo_16(S2MM, 0, 2)
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_19_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_19_buff_0 : memref<1x2x1x32xbf16>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_19_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_19_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_19_buff_1 : memref<1x2x1x32xbf16>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_19_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_19_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_19_buff_0 : memref<1x2x1x32xbf16>, 32, 32) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%fifo_19_cons_lock_1, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_19_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_19_buff_1 : memref<1x2x1x32xbf16>, 32, 32) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%fifo_19_cons_lock_1, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb11)
    ^bb7:  // 2 preds: ^bb6, ^bb10
      aie.use_lock(%fifo_19_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_19_buff_0 : memref<1x2x1x32xbf16>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_19_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%fifo_19_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_19_buff_0 : memref<1x2x1x32xbf16>, 32, 32) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%fifo_19_prod_lock_1, Release, 1)
      aie.next_bd ^bb9
    ^bb9:  // pred: ^bb8
      aie.use_lock(%fifo_19_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_19_buff_1 : memref<1x2x1x32xbf16>, 0, 32) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%fifo_19_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%fifo_19_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_19_buff_1 : memref<1x2x1x32xbf16>, 32, 32) {bd_id = 5 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_19_prod_lock_1, Release, 1)
      aie.next_bd ^bb7
    ^bb11:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @fifo_19(S2MM, 0, 3)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_2_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_2_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_3_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_3_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
    aie.wire(%tile_0_3 : DMA, %switchbox_0_3 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
    aie.wire(%tile_0_4 : Core, %switchbox_0_4 : Core)
    aie.wire(%tile_0_4 : DMA, %switchbox_0_4 : DMA)
    aie.wire(%switchbox_0_3 : North, %switchbox_0_4 : South)
    aie.wire(%tile_0_5 : Core, %switchbox_0_5 : Core)
    aie.wire(%tile_0_5 : DMA, %switchbox_0_5 : DMA)
    aie.wire(%switchbox_0_4 : North, %switchbox_0_5 : South)
    aie.wire(%switchbox_0_0 : East, %switchbox_1_0 : West)
    aie.wire(%shim_mux_1_0 : North, %switchbox_1_0 : South)
    aie.wire(%shim_noc_tile_1_0 : DMA, %shim_mux_1_0 : DMA)
    aie.wire(%switchbox_0_1 : East, %switchbox_1_1 : West)
    aie.wire(%mem_tile_1_1 : Core, %switchbox_1_1 : Core)
    aie.wire(%mem_tile_1_1 : DMA, %switchbox_1_1 : DMA)
    aie.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
    aie.wire(%switchbox_0_2 : East, %switchbox_1_2 : West)
    aie.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
    aie.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
    aie.wire(%switchbox_1_1 : North, %switchbox_1_2 : South)
    aie.wire(%switchbox_0_3 : East, %switchbox_1_3 : West)
    aie.wire(%tile_1_3 : Core, %switchbox_1_3 : Core)
    aie.wire(%tile_1_3 : DMA, %switchbox_1_3 : DMA)
    aie.wire(%switchbox_1_2 : North, %switchbox_1_3 : South)
    aie.wire(%switchbox_0_4 : East, %switchbox_1_4 : West)
    aie.wire(%tile_1_4 : Core, %switchbox_1_4 : Core)
    aie.wire(%tile_1_4 : DMA, %switchbox_1_4 : DMA)
    aie.wire(%switchbox_1_3 : North, %switchbox_1_4 : South)
    aie.wire(%switchbox_0_5 : East, %switchbox_1_5 : West)
    aie.wire(%tile_1_5 : Core, %switchbox_1_5 : Core)
    aie.wire(%tile_1_5 : DMA, %switchbox_1_5 : DMA)
    aie.wire(%switchbox_1_4 : North, %switchbox_1_5 : South)
    aie.wire(%switchbox_1_0 : East, %switchbox_2_0 : West)
    aie.wire(%shim_mux_2_0 : North, %switchbox_2_0 : South)
    aie.wire(%shim_noc_tile_2_0 : DMA, %shim_mux_2_0 : DMA)
    aie.wire(%switchbox_1_1 : East, %switchbox_2_1 : West)
    aie.wire(%mem_tile_2_1 : Core, %switchbox_2_1 : Core)
    aie.wire(%mem_tile_2_1 : DMA, %switchbox_2_1 : DMA)
    aie.wire(%switchbox_2_0 : North, %switchbox_2_1 : South)
    aie.wire(%switchbox_1_2 : East, %switchbox_2_2 : West)
    aie.wire(%tile_2_2 : Core, %switchbox_2_2 : Core)
    aie.wire(%tile_2_2 : DMA, %switchbox_2_2 : DMA)
    aie.wire(%switchbox_2_1 : North, %switchbox_2_2 : South)
    aie.wire(%switchbox_1_3 : East, %switchbox_2_3 : West)
    aie.wire(%tile_2_3 : Core, %switchbox_2_3 : Core)
    aie.wire(%tile_2_3 : DMA, %switchbox_2_3 : DMA)
    aie.wire(%switchbox_2_2 : North, %switchbox_2_3 : South)
    aie.wire(%switchbox_1_4 : East, %switchbox_2_4 : West)
    aie.wire(%tile_2_4 : Core, %switchbox_2_4 : Core)
    aie.wire(%tile_2_4 : DMA, %switchbox_2_4 : DMA)
    aie.wire(%switchbox_2_3 : North, %switchbox_2_4 : South)
    aie.wire(%switchbox_1_5 : East, %switchbox_2_5 : West)
    aie.wire(%tile_2_5 : Core, %switchbox_2_5 : Core)
    aie.wire(%tile_2_5 : DMA, %switchbox_2_5 : DMA)
    aie.wire(%switchbox_2_4 : North, %switchbox_2_5 : South)
    aie.wire(%switchbox_2_0 : East, %switchbox_3_0 : West)
    aie.wire(%shim_mux_3_0 : North, %switchbox_3_0 : South)
    aie.wire(%shim_noc_tile_3_0 : DMA, %shim_mux_3_0 : DMA)
    aie.wire(%switchbox_2_1 : East, %switchbox_3_1 : West)
    aie.wire(%mem_tile_3_1 : Core, %switchbox_3_1 : Core)
    aie.wire(%mem_tile_3_1 : DMA, %switchbox_3_1 : DMA)
    aie.wire(%switchbox_3_0 : North, %switchbox_3_1 : South)
    aie.wire(%switchbox_2_2 : East, %switchbox_3_2 : West)
    aie.wire(%tile_3_2 : Core, %switchbox_3_2 : Core)
    aie.wire(%tile_3_2 : DMA, %switchbox_3_2 : DMA)
    aie.wire(%switchbox_3_1 : North, %switchbox_3_2 : South)
    aie.wire(%switchbox_2_3 : East, %switchbox_3_3 : West)
    aie.wire(%tile_3_3 : Core, %switchbox_3_3 : Core)
    aie.wire(%tile_3_3 : DMA, %switchbox_3_3 : DMA)
    aie.wire(%switchbox_3_2 : North, %switchbox_3_3 : South)
  }
}

