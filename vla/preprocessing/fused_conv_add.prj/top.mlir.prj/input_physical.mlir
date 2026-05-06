module {
  aie.device(npu1_4col) {
    memref.global "public" @fifo_11_cons : memref<1x4x32x32xbf16>
    memref.global "public" @fifo_11 : memref<1x4x32x32xbf16>
    memref.global "public" @fifo_10_cons : memref<32x32xbf16>
    memref.global "public" @fifo_10 : memref<32x32xbf16>
    memref.global "public" @fifo_9_cons : memref<32x32xbf16>
    memref.global "public" @fifo_9 : memref<32x32xbf16>
    memref.global "public" @fifo_8_cons : memref<32x32xbf16>
    memref.global "public" @fifo_8 : memref<32x32xbf16>
    memref.global "public" @fifo_7_cons : memref<32x32xbf16>
    memref.global "public" @fifo_7 : memref<32x32xbf16>
    memref.global "public" @fifo_6_cons : memref<1x4x192x32xbf16>
    memref.global "public" @fifo_6 : memref<1x4x192x32xbf16>
    memref.global "public" @fifo_5_cons : memref<192x32xbf16>
    memref.global "public" @fifo_5 : memref<192x32xbf16>
    memref.global "public" @fifo_4_cons : memref<192x32xbf16>
    memref.global "public" @fifo_4 : memref<192x32xbf16>
    memref.global "public" @fifo_3_cons : memref<192x32xbf16>
    memref.global "public" @fifo_3 : memref<192x32xbf16>
    memref.global "public" @fifo_2_cons : memref<192x32xbf16>
    memref.global "public" @fifo_2 : memref<192x32xbf16>
    memref.global "public" @fifo_1_cons : memref<1x1x32x192xbf16>
    memref.global "public" @fifo_1 : memref<1x1x32x192xbf16>
    memref.global "public" @fifo_0_0_cons : memref<32x192xbf16>
    memref.global "public" @fifo_0_1_cons : memref<32x192xbf16>
    memref.global "public" @fifo_0_2_cons : memref<32x192xbf16>
    memref.global "public" @fifo_0_3_cons : memref<32x192xbf16>
    memref.global "public" @fifo_0 : memref<32x192xbf16>
    func.func private @fill_zeros_bf16_32_32_vector(memref<32x32xbf16>)
    func.func private @matmul_scalar_bf16_bf16(memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>)
    func.func private @add_bf16_vector(memref<32x32xbf16>, memref<32x32xbf16>, memref<32x32xbf16>)
    func.func private @matmul_bf16_bf16(memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>)
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_2_1 = aie.tile(2, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %fifo_11_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 1 : i32, sym_name = "fifo_11_cons_prod_lock_0"}
    %fifo_11_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "fifo_11_cons_cons_lock_0"}
    %fifo_11_buff_0 = aie.buffer(%mem_tile_2_1) {address = 0 : i32, sym_name = "fifo_11_buff_0"} : memref<1x4x32x32xbf16> 
    %fifo_11_buff_1 = aie.buffer(%mem_tile_2_1) {address = 8192 : i32, sym_name = "fifo_11_buff_1"} : memref<1x4x32x32xbf16> 
    %fifo_11_prod_lock_0 = aie.lock(%mem_tile_2_1, 0) {init = 2 : i32, sym_name = "fifo_11_prod_lock_0"}
    %fifo_11_cons_lock_0 = aie.lock(%mem_tile_2_1, 1) {init = 0 : i32, sym_name = "fifo_11_cons_lock_0"}
    %fifo_11_prod_lock_1 = aie.lock(%mem_tile_2_1, 2) {init = 2 : i32, sym_name = "fifo_11_prod_lock_1"}
    %fifo_11_cons_lock_1 = aie.lock(%mem_tile_2_1, 3) {init = 0 : i32, sym_name = "fifo_11_cons_lock_1"}
    %fifo_11_prod_lock_2 = aie.lock(%mem_tile_2_1, 4) {init = 2 : i32, sym_name = "fifo_11_prod_lock_2"}
    %fifo_11_cons_lock_2 = aie.lock(%mem_tile_2_1, 5) {init = 0 : i32, sym_name = "fifo_11_cons_lock_2"}
    %fifo_11_prod_lock_3 = aie.lock(%mem_tile_2_1, 6) {init = 2 : i32, sym_name = "fifo_11_prod_lock_3"}
    %fifo_11_cons_lock_3 = aie.lock(%mem_tile_2_1, 7) {init = 0 : i32, sym_name = "fifo_11_cons_lock_3"}
    %fifo_10_buff_0 = aie.buffer(%tile_0_5) {address = 50176 : i32, sym_name = "fifo_10_buff_0"} : memref<32x32xbf16> 
    %fifo_10_buff_1 = aie.buffer(%tile_0_5) {address = 52224 : i32, sym_name = "fifo_10_buff_1"} : memref<32x32xbf16> 
    %fifo_10_prod_lock_0 = aie.lock(%tile_0_5, 4) {init = 2 : i32, sym_name = "fifo_10_prod_lock_0"}
    %fifo_10_cons_lock_0 = aie.lock(%tile_0_5, 5) {init = 0 : i32, sym_name = "fifo_10_cons_lock_0"}
    %fifo_9_buff_0 = aie.buffer(%tile_0_4) {address = 50176 : i32, sym_name = "fifo_9_buff_0"} : memref<32x32xbf16> 
    %fifo_9_buff_1 = aie.buffer(%tile_0_4) {address = 52224 : i32, sym_name = "fifo_9_buff_1"} : memref<32x32xbf16> 
    %fifo_9_prod_lock_0 = aie.lock(%tile_0_4, 4) {init = 2 : i32, sym_name = "fifo_9_prod_lock_0"}
    %fifo_9_cons_lock_0 = aie.lock(%tile_0_4, 5) {init = 0 : i32, sym_name = "fifo_9_cons_lock_0"}
    %fifo_8_buff_0 = aie.buffer(%tile_0_3) {address = 50176 : i32, sym_name = "fifo_8_buff_0"} : memref<32x32xbf16> 
    %fifo_8_buff_1 = aie.buffer(%tile_0_3) {address = 52224 : i32, sym_name = "fifo_8_buff_1"} : memref<32x32xbf16> 
    %fifo_8_prod_lock_0 = aie.lock(%tile_0_3, 4) {init = 2 : i32, sym_name = "fifo_8_prod_lock_0"}
    %fifo_8_cons_lock_0 = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "fifo_8_cons_lock_0"}
    %fifo_7_buff_0 = aie.buffer(%tile_0_2) {address = 50176 : i32, sym_name = "fifo_7_buff_0"} : memref<32x32xbf16> 
    %fifo_7_buff_1 = aie.buffer(%tile_0_2) {address = 52224 : i32, sym_name = "fifo_7_buff_1"} : memref<32x32xbf16> 
    %fifo_7_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "fifo_7_prod_lock_0"}
    %fifo_7_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "fifo_7_cons_lock_0"}
    %fifo_6_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, sym_name = "fifo_6_cons_buff_0"} : memref<1x4x192x32xbf16> 
    %fifo_6_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 49152 : i32, sym_name = "fifo_6_cons_buff_1"} : memref<1x4x192x32xbf16> 
    %fifo_6_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 2 : i32, sym_name = "fifo_6_cons_prod_lock_0"}
    %fifo_6_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "fifo_6_cons_cons_lock_0"}
    %fifo_6_cons_prod_lock_1 = aie.lock(%mem_tile_1_1, 2) {init = 2 : i32, sym_name = "fifo_6_cons_prod_lock_1"}
    %fifo_6_cons_cons_lock_1 = aie.lock(%mem_tile_1_1, 3) {init = 0 : i32, sym_name = "fifo_6_cons_cons_lock_1"}
    %fifo_6_cons_prod_lock_2 = aie.lock(%mem_tile_1_1, 4) {init = 2 : i32, sym_name = "fifo_6_cons_prod_lock_2"}
    %fifo_6_cons_cons_lock_2 = aie.lock(%mem_tile_1_1, 5) {init = 0 : i32, sym_name = "fifo_6_cons_cons_lock_2"}
    %fifo_6_cons_prod_lock_3 = aie.lock(%mem_tile_1_1, 6) {init = 2 : i32, sym_name = "fifo_6_cons_prod_lock_3"}
    %fifo_6_cons_cons_lock_3 = aie.lock(%mem_tile_1_1, 7) {init = 0 : i32, sym_name = "fifo_6_cons_cons_lock_3"}
    %fifo_6_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 1 : i32, sym_name = "fifo_6_prod_lock_0"}
    %fifo_6_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "fifo_6_cons_lock_0"}
    %fifo_5_cons_buff_0 = aie.buffer(%tile_0_5) {address = 1024 : i32, sym_name = "fifo_5_cons_buff_0"} : memref<192x32xbf16> 
    %fifo_5_cons_buff_1 = aie.buffer(%tile_0_5) {address = 13312 : i32, sym_name = "fifo_5_cons_buff_1"} : memref<192x32xbf16> 
    %fifo_5_cons_prod_lock_0 = aie.lock(%tile_0_5, 2) {init = 2 : i32, sym_name = "fifo_5_cons_prod_lock_0"}
    %fifo_5_cons_cons_lock_0 = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "fifo_5_cons_cons_lock_0"}
    %fifo_4_cons_buff_0 = aie.buffer(%tile_0_4) {address = 1024 : i32, sym_name = "fifo_4_cons_buff_0"} : memref<192x32xbf16> 
    %fifo_4_cons_buff_1 = aie.buffer(%tile_0_4) {address = 13312 : i32, sym_name = "fifo_4_cons_buff_1"} : memref<192x32xbf16> 
    %fifo_4_cons_prod_lock_0 = aie.lock(%tile_0_4, 2) {init = 2 : i32, sym_name = "fifo_4_cons_prod_lock_0"}
    %fifo_4_cons_cons_lock_0 = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "fifo_4_cons_cons_lock_0"}
    %fifo_3_cons_buff_0 = aie.buffer(%tile_0_3) {address = 1024 : i32, sym_name = "fifo_3_cons_buff_0"} : memref<192x32xbf16> 
    %fifo_3_cons_buff_1 = aie.buffer(%tile_0_3) {address = 13312 : i32, sym_name = "fifo_3_cons_buff_1"} : memref<192x32xbf16> 
    %fifo_3_cons_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "fifo_3_cons_prod_lock_0"}
    %fifo_3_cons_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "fifo_3_cons_cons_lock_0"}
    %fifo_2_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "fifo_2_cons_buff_0"} : memref<192x32xbf16> 
    %fifo_2_cons_buff_1 = aie.buffer(%tile_0_2) {address = 13312 : i32, sym_name = "fifo_2_cons_buff_1"} : memref<192x32xbf16> 
    %fifo_2_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "fifo_2_cons_prod_lock_0"}
    %fifo_2_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "fifo_2_cons_cons_lock_0"}
    %fifo_1_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "fifo_1_cons_buff_0"} : memref<1x1x32x192xbf16> 
    %fifo_1_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 12288 : i32, sym_name = "fifo_1_cons_buff_1"} : memref<1x1x32x192xbf16> 
    %fifo_1_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "fifo_1_cons_prod_lock_0"}
    %fifo_1_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "fifo_1_cons_cons_lock_0"}
    %fifo_1_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 1 : i32, sym_name = "fifo_1_prod_lock_0"}
    %fifo_1_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "fifo_1_cons_lock_0"}
    %fifo_0_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 25600 : i32, sym_name = "fifo_0_0_cons_buff_0"} : memref<32x192xbf16> 
    %fifo_0_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 37888 : i32, sym_name = "fifo_0_0_cons_buff_1"} : memref<32x192xbf16> 
    %fifo_0_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "fifo_0_0_cons_prod_lock_0"}
    %fifo_0_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "fifo_0_0_cons_cons_lock_0"}
    %fifo_0_1_cons_buff_0 = aie.buffer(%tile_0_5) {address = 25600 : i32, sym_name = "fifo_0_1_cons_buff_0"} : memref<32x192xbf16> 
    %fifo_0_1_cons_buff_1 = aie.buffer(%tile_0_5) {address = 37888 : i32, sym_name = "fifo_0_1_cons_buff_1"} : memref<32x192xbf16> 
    %fifo_0_1_cons_prod_lock_0 = aie.lock(%tile_0_5, 0) {init = 2 : i32, sym_name = "fifo_0_1_cons_prod_lock_0"}
    %fifo_0_1_cons_cons_lock_0 = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "fifo_0_1_cons_cons_lock_0"}
    %fifo_0_2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 25600 : i32, sym_name = "fifo_0_2_cons_buff_0"} : memref<32x192xbf16> 
    %fifo_0_2_cons_buff_1 = aie.buffer(%tile_0_4) {address = 37888 : i32, sym_name = "fifo_0_2_cons_buff_1"} : memref<32x192xbf16> 
    %fifo_0_2_cons_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "fifo_0_2_cons_prod_lock_0"}
    %fifo_0_2_cons_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "fifo_0_2_cons_cons_lock_0"}
    %fifo_0_3_cons_buff_0 = aie.buffer(%tile_0_3) {address = 25600 : i32, sym_name = "fifo_0_3_cons_buff_0"} : memref<32x192xbf16> 
    %fifo_0_3_cons_buff_1 = aie.buffer(%tile_0_3) {address = 37888 : i32, sym_name = "fifo_0_3_cons_buff_1"} : memref<32x192xbf16> 
    %fifo_0_3_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "fifo_0_3_cons_prod_lock_0"}
    %fifo_0_3_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "fifo_0_3_cons_cons_lock_0"}
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<DMA : 0, North : 1>
      aie.connect<South : 3, DMA : 0>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, North : 1>
      aie.connect<South : 1, DMA : 0>
      aie.connect<East : 3, DMA : 1>
      aie.connect<DMA : 0, East : 0>
      aie.connect<North : 1, East : 1>
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 1, North : 0>
      aie.connect<East : 2, DMA : 1>
      aie.connect<East : 3, North : 1>
      aie.connect<DMA : 0, East : 0>
      aie.connect<North : 0, South : 1>
    }
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 0, North : 0>
      aie.connect<South : 1, DMA : 1>
      aie.connect<DMA : 0, East : 0>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_5 = aie.switchbox(%tile_0_5) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<East : 0, DMA : 1>
      aie.connect<DMA : 0, South : 0>
    }
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_4, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_5, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 3>
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
      aie.connect<DMA : 2, North : 0>
      aie.connect<DMA : 3, North : 3>
      aie.connect<South : 1, DMA : 0>
    }
    %tile_1_2 = aie.tile(1, 2)
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<South : 1, West : 3>
      aie.connect<South : 5, North : 1>
      aie.connect<South : 0, North : 2>
      aie.connect<South : 3, North : 0>
      aie.connect<West : 0, East : 1>
      aie.connect<West : 1, East : 2>
    }
    %tile_1_3 = aie.tile(1, 3)
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<South : 1, West : 2>
      aie.connect<South : 2, West : 3>
      aie.connect<South : 0, North : 2>
      aie.connect<West : 0, East : 1>
    }
    %tile_1_4 = aie.tile(1, 4)
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      aie.connect<South : 2, North : 5>
      aie.connect<West : 0, East : 1>
    }
    %tile_1_5 = aie.tile(1, 5)
    %switchbox_1_5 = aie.switchbox(%tile_1_5) {
      aie.connect<South : 5, West : 0>
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<South : 3, North : 1>
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
      aie.connect<North : 3, DMA : 0>
      aie.connect<North : 0, DMA : 1>
      aie.connect<North : 2, DMA : 2>
      aie.connect<North : 1, DMA : 3>
      aie.connect<DMA : 0, South : 2>
    }
    %tile_2_2 = aie.tile(2, 2)
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      aie.connect<West : 1, South : 3>
      aie.connect<North : 1, South : 0>
      aie.connect<North : 3, South : 2>
      aie.connect<West : 2, South : 1>
    }
    %tile_2_3 = aie.tile(2, 3)
    %switchbox_2_3 = aie.switchbox(%tile_2_3) {
      aie.connect<West : 1, South : 1>
      aie.connect<North : 3, South : 3>
    }
    %tile_2_4 = aie.tile(2, 4)
    %switchbox_2_4 = aie.switchbox(%tile_2_4) {
      aie.connect<West : 1, South : 3>
    }
    %switchbox_2_0 = aie.switchbox(%shim_noc_tile_2_0) {
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
      aie.use_lock(%fifo_7_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @fill_zeros_bf16_32_32_vector(%fifo_7_buff_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_0_cons_buff_0, %fifo_2_cons_buff_0, %fifo_7_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_0_cons_buff_1, %fifo_2_cons_buff_1, %fifo_7_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_0_cons_buff_0, %fifo_2_cons_buff_0, %fifo_7_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_0_cons_buff_1, %fifo_2_cons_buff_1, %fifo_7_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_7_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_7_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @fill_zeros_bf16_32_32_vector(%fifo_7_buff_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_0_cons_buff_0, %fifo_2_cons_buff_0, %fifo_7_buff_1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_0_cons_buff_1, %fifo_2_cons_buff_1, %fifo_7_buff_1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_0_cons_buff_0, %fifo_2_cons_buff_0, %fifo_7_buff_1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_0_cons_buff_1, %fifo_2_cons_buff_1, %fifo_7_buff_1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_7_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%fifo_7_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @fill_zeros_bf16_32_32_vector(%fifo_7_buff_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_0_cons_buff_0, %fifo_2_cons_buff_0, %fifo_7_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_0_cons_buff_1, %fifo_2_cons_buff_1, %fifo_7_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_0_cons_buff_0, %fifo_2_cons_buff_0, %fifo_7_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_0_cons_buff_1, %fifo_2_cons_buff_1, %fifo_7_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_7_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "external0.o"}
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
      aie.use_lock(%fifo_8_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @fill_zeros_bf16_32_32_vector(%fifo_8_buff_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_3_cons_buff_0, %fifo_3_cons_buff_0, %fifo_8_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_3_cons_buff_1, %fifo_3_cons_buff_1, %fifo_8_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_3_cons_buff_0, %fifo_3_cons_buff_0, %fifo_8_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_3_cons_buff_1, %fifo_3_cons_buff_1, %fifo_8_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_8_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_8_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @fill_zeros_bf16_32_32_vector(%fifo_8_buff_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_3_cons_buff_0, %fifo_3_cons_buff_0, %fifo_8_buff_1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_3_cons_buff_1, %fifo_3_cons_buff_1, %fifo_8_buff_1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_3_cons_buff_0, %fifo_3_cons_buff_0, %fifo_8_buff_1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_3_cons_buff_1, %fifo_3_cons_buff_1, %fifo_8_buff_1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_8_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%fifo_8_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @fill_zeros_bf16_32_32_vector(%fifo_8_buff_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_3_cons_buff_0, %fifo_3_cons_buff_0, %fifo_8_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_3_cons_buff_1, %fifo_3_cons_buff_1, %fifo_8_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_3_cons_buff_0, %fifo_3_cons_buff_0, %fifo_8_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_3_cons_buff_1, %fifo_3_cons_buff_1, %fifo_8_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_3_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_8_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "external0.o"}
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
      aie.use_lock(%fifo_9_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @fill_zeros_bf16_32_32_vector(%fifo_9_buff_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_2_cons_buff_0, %fifo_4_cons_buff_0, %fifo_9_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_2_cons_buff_1, %fifo_4_cons_buff_1, %fifo_9_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_2_cons_buff_0, %fifo_4_cons_buff_0, %fifo_9_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_2_cons_buff_1, %fifo_4_cons_buff_1, %fifo_9_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_9_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_9_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @fill_zeros_bf16_32_32_vector(%fifo_9_buff_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_2_cons_buff_0, %fifo_4_cons_buff_0, %fifo_9_buff_1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_2_cons_buff_1, %fifo_4_cons_buff_1, %fifo_9_buff_1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_2_cons_buff_0, %fifo_4_cons_buff_0, %fifo_9_buff_1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_2_cons_buff_1, %fifo_4_cons_buff_1, %fifo_9_buff_1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_9_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%fifo_9_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @fill_zeros_bf16_32_32_vector(%fifo_9_buff_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_2_cons_buff_0, %fifo_4_cons_buff_0, %fifo_9_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_2_cons_buff_1, %fifo_4_cons_buff_1, %fifo_9_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_2_cons_buff_0, %fifo_4_cons_buff_0, %fifo_9_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_4_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_2_cons_buff_1, %fifo_4_cons_buff_1, %fifo_9_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_2_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_4_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_9_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "external0.o"}
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
      aie.use_lock(%fifo_10_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @fill_zeros_bf16_32_32_vector(%fifo_10_buff_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_1_cons_buff_0, %fifo_5_cons_buff_0, %fifo_10_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_1_cons_buff_1, %fifo_5_cons_buff_1, %fifo_10_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_1_cons_buff_0, %fifo_5_cons_buff_0, %fifo_10_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_1_cons_buff_1, %fifo_5_cons_buff_1, %fifo_10_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_10_cons_lock_0, Release, 1)
      aie.use_lock(%fifo_10_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @fill_zeros_bf16_32_32_vector(%fifo_10_buff_1) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_1_cons_buff_0, %fifo_5_cons_buff_0, %fifo_10_buff_1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_1_cons_buff_1, %fifo_5_cons_buff_1, %fifo_10_buff_1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_1_cons_buff_0, %fifo_5_cons_buff_0, %fifo_10_buff_1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_1_cons_buff_1, %fifo_5_cons_buff_1, %fifo_10_buff_1) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_10_cons_lock_0, Release, 1)
      %2 = arith.addi %0, %c2 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.use_lock(%fifo_10_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @fill_zeros_bf16_32_32_vector(%fifo_10_buff_0) {lib = "fill_zeros_bf16_32_32_vector"} : (memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_1_cons_buff_0, %fifo_5_cons_buff_0, %fifo_10_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_1_cons_buff_1, %fifo_5_cons_buff_1, %fifo_10_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_1_cons_buff_0, %fifo_5_cons_buff_0, %fifo_10_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%fifo_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_5_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%fifo_0_1_cons_buff_1, %fifo_5_cons_buff_1, %fifo_10_buff_0) : (memref<32x192xbf16>, memref<192x32xbf16>, memref<32x32xbf16>) -> ()
      aie.use_lock(%fifo_0_1_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_5_cons_prod_lock_0, Release, 1)
      aie.use_lock(%fifo_10_cons_lock_0, Release, 1)
      aie.end
    } {link_with = "external0.o"}
    aiex.runtime_sequence(%arg0: memref<24576xbf16>, %arg1: memref<4096xbf16>, %arg2: memref<98304xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 4, 32, 192][24576, 192, 768, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_1} : memref<24576xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][4, 4, 192, 32][24576, 32, 128, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_6} : memref<98304xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 4, 32, 32][4096, 32, 128, 1]) {id = 0 : i64, issue_token = true, metadata = @fifo_11} : memref<4096xbf16>
      aiex.npu.dma_wait {symbol = @fifo_11}
      aie.end
    }
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_0 : memref<1x1x32x192xbf16>, 0, 6144, [<size = 8, stride = 768>, <size = 24, stride = 8>, <size = 4, stride = 192>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_1 : memref<1x1x32x192xbf16>, 0, 6144, [<size = 8, stride = 768>, <size = 24, stride = 8>, <size = 4, stride = 192>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_0 : memref<1x1x32x192xbf16>, 0, 6144) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_1_cons_buff_1 : memref<1x1x32x192xbf16>, 0, 6144) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_0_0_cons_buff_0 : memref<32x192xbf16>, 0, 6144) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_0_0_cons_buff_1 : memref<32x192xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_2_cons_buff_0 : memref<192x32xbf16>, 0, 6144) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_2_cons_buff_1 : memref<192x32xbf16>, 0, 6144) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%fifo_7_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_7_buff_0 : memref<32x32xbf16>, 0, 1024) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%fifo_7_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%fifo_7_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_7_buff_1 : memref<32x32xbf16>, 0, 1024) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%fifo_7_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_0_1_cons_buff_0 : memref<32x192xbf16>, 0, 6144) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_0_1_cons_buff_1 : memref<32x192xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_5_cons_buff_0 : memref<192x32xbf16>, 0, 6144) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_5_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_5_cons_buff_1 : memref<192x32xbf16>, 0, 6144) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_5_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%fifo_10_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_10_buff_0 : memref<32x32xbf16>, 0, 1024) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%fifo_10_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%fifo_10_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_10_buff_1 : memref<32x32xbf16>, 0, 1024) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%fifo_10_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_0_2_cons_buff_0 : memref<32x192xbf16>, 0, 6144) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_0_2_cons_buff_1 : memref<32x192xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_cons_buff_0 : memref<192x32xbf16>, 0, 6144) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_4_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_4_cons_buff_1 : memref<192x32xbf16>, 0, 6144) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_4_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%fifo_9_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_9_buff_0 : memref<32x32xbf16>, 0, 1024) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%fifo_9_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%fifo_9_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_9_buff_1 : memref<32x32xbf16>, 0, 1024) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%fifo_9_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_0_3_cons_buff_0 : memref<32x192xbf16>, 0, 6144) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_0_3_cons_buff_1 : memref<32x192xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_3_cons_buff_0 : memref<192x32xbf16>, 0, 6144) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_3_cons_buff_1 : memref<192x32xbf16>, 0, 6144) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%fifo_8_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_8_buff_0 : memref<32x32xbf16>, 0, 1024) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%fifo_8_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%fifo_8_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_8_buff_1 : memref<32x32xbf16>, 0, 1024) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%fifo_8_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @fifo_1(MM2S, 0, 0)
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<1x4x192x32xbf16>, 0, 6144, [<size = 24, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_6_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<1x4x192x32xbf16>, 0, 6144, [<size = 24, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_6_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<1x4x192x32xbf16>, 6144, 6144, [<size = 24, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_6_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<1x4x192x32xbf16>, 6144, 6144, [<size = 24, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 2, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%fifo_6_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<1x4x192x32xbf16>, 12288, 6144, [<size = 24, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>]) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%fifo_6_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<1x4x192x32xbf16>, 12288, 6144, [<size = 24, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>]) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 3, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%fifo_6_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<1x4x192x32xbf16>, 18432, 6144, [<size = 24, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>]) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%fifo_6_cons_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<1x4x192x32xbf16>, 18432, 6144, [<size = 24, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>]) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%fifo_6_cons_prod_lock_3, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 0, ^bb13, ^bb21)
    ^bb13:  // 2 preds: ^bb12, ^bb20
      aie.use_lock(%fifo_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<1x4x192x32xbf16>, 0, 6144) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%fifo_6_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<1x4x192x32xbf16>, 6144, 6144) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb15
    ^bb15:  // pred: ^bb14
      aie.use_lock(%fifo_6_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<1x4x192x32xbf16>, 12288, 6144) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb16
    ^bb16:  // pred: ^bb15
      aie.use_lock(%fifo_6_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_0 : memref<1x4x192x32xbf16>, 18432, 6144) {bd_id = 7 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%fifo_6_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<1x4x192x32xbf16>, 0, 6144) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb18
    ^bb18:  // pred: ^bb17
      aie.use_lock(%fifo_6_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<1x4x192x32xbf16>, 6144, 6144) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb19
    ^bb19:  // pred: ^bb18
      aie.use_lock(%fifo_6_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<1x4x192x32xbf16>, 12288, 6144) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%fifo_6_cons_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_6_cons_buff_1 : memref<1x4x192x32xbf16>, 18432, 6144) {bd_id = 11 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%fifo_6_cons_cons_lock_3, Release, 1)
      aie.next_bd ^bb13
    ^bb21:  // pred: ^bb12
      aie.end
    }
    aie.shim_dma_allocation @fifo_6(MM2S, 0, 1)
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%fifo_11_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_11_buff_0 : memref<1x4x32x32xbf16>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%fifo_11_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%fifo_11_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_11_buff_1 : memref<1x4x32x32xbf16>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%fifo_11_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%fifo_11_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_11_buff_0 : memref<1x4x32x32xbf16>, 1024, 1024) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%fifo_11_cons_lock_1, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%fifo_11_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_11_buff_1 : memref<1x4x32x32xbf16>, 1024, 1024) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%fifo_11_cons_lock_1, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 2, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%fifo_11_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_11_buff_0 : memref<1x4x32x32xbf16>, 2048, 1024) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%fifo_11_cons_lock_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%fifo_11_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_11_buff_1 : memref<1x4x32x32xbf16>, 2048, 1024) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%fifo_11_cons_lock_2, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 3, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%fifo_11_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_11_buff_0 : memref<1x4x32x32xbf16>, 3072, 1024) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%fifo_11_cons_lock_3, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%fifo_11_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_11_buff_1 : memref<1x4x32x32xbf16>, 3072, 1024) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%fifo_11_cons_lock_3, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(MM2S, 0, ^bb13, ^bb21)
    ^bb13:  // 2 preds: ^bb12, ^bb20
      aie.use_lock(%fifo_11_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_11_buff_0 : memref<1x4x32x32xbf16>, 0, 1024, [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>]) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%fifo_11_prod_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%fifo_11_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_11_buff_0 : memref<1x4x32x32xbf16>, 1024, 1024, [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>]) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%fifo_11_prod_lock_1, Release, 1)
      aie.next_bd ^bb15
    ^bb15:  // pred: ^bb14
      aie.use_lock(%fifo_11_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_11_buff_0 : memref<1x4x32x32xbf16>, 2048, 1024, [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>]) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%fifo_11_prod_lock_2, Release, 1)
      aie.next_bd ^bb16
    ^bb16:  // pred: ^bb15
      aie.use_lock(%fifo_11_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_11_buff_0 : memref<1x4x32x32xbf16>, 3072, 1024, [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>]) {bd_id = 7 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%fifo_11_prod_lock_3, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%fifo_11_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_11_buff_1 : memref<1x4x32x32xbf16>, 0, 1024, [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%fifo_11_prod_lock_0, Release, 1)
      aie.next_bd ^bb18
    ^bb18:  // pred: ^bb17
      aie.use_lock(%fifo_11_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_11_buff_1 : memref<1x4x32x32xbf16>, 1024, 1024, [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%fifo_11_prod_lock_1, Release, 1)
      aie.next_bd ^bb19
    ^bb19:  // pred: ^bb18
      aie.use_lock(%fifo_11_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_11_buff_1 : memref<1x4x32x32xbf16>, 2048, 1024, [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%fifo_11_prod_lock_2, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%fifo_11_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%fifo_11_buff_1 : memref<1x4x32x32xbf16>, 3072, 1024, [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%fifo_11_prod_lock_3, Release, 1)
      aie.next_bd ^bb13
    ^bb21:  // pred: ^bb12
      aie.end
    }
    aie.shim_dma_allocation @fifo_11(S2MM, 0, 2)
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
  }
}

