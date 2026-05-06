module attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @fifo_0_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_2_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_2_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_3_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_3_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_4_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<1 x array<4 x array<1 x array<768 x bf16>>>>
  llvm.mlir.global external @fifo_4_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<1 x array<4 x array<1 x array<768 x bf16>>>>
  llvm.mlir.global external @fifo_5_buff_1() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_5_buff_0() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_6_buff_1() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_6_buff_0() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_7_buff_1() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_7_buff_0() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_8_buff_1() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_8_buff_0() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_9_buff_1() {addr_space = 0 : i32} : !llvm.array<1 x array<4 x array<1 x array<768 x bf16>>>>
  llvm.mlir.global external @fifo_9_buff_0() {addr_space = 0 : i32} : !llvm.array<1 x array<4 x array<1 x array<768 x bf16>>>>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.mlir.global external @fifo_9_cons() {addr_space = 0 : i32} : !llvm.array<1 x array<4 x array<1 x array<768 x bf16>>>>
  llvm.mlir.global external @fifo_9() {addr_space = 0 : i32} : !llvm.array<1 x array<4 x array<1 x array<768 x bf16>>>>
  llvm.mlir.global external @fifo_8_cons() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_8() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_7_cons() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_7() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_6_cons() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_6() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_5_cons() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_5() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_4_cons() {addr_space = 0 : i32} : !llvm.array<1 x array<4 x array<1 x array<768 x bf16>>>>
  llvm.mlir.global external @fifo_4() {addr_space = 0 : i32} : !llvm.array<1 x array<4 x array<1 x array<768 x bf16>>>>
  llvm.mlir.global external @fifo_3_cons() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_3() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_2_cons() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_2() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_1_cons() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_1() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_0_cons() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.mlir.global external @fifo_0() {addr_space = 0 : i32} : !llvm.array<1 x array<768 x bf16>>
  llvm.func @core_0_5() {
    %0 = llvm.mlir.addressof @fifo_8_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @fifo_3_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @fifo_8_buff_0 : !llvm.ptr
    %3 = llvm.mlir.constant(32 : index) : i64
    %4 = llvm.mlir.constant(true) : i1
    %5 = llvm.mlir.addressof @fifo_3_cons_buff_0 : !llvm.ptr
    %6 = llvm.mlir.zero : !llvm.ptr
    %7 = llvm.mlir.constant(768 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(51 : i32) : i32
    %10 = llvm.mlir.constant(48 : i32) : i32
    %11 = llvm.mlir.constant(50 : i32) : i32
    %12 = llvm.mlir.constant(49 : i32) : i32
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.mlir.constant(-1 : i32) : i32
    %15 = llvm.mlir.constant(2 : index) : i64
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.mlir.constant(9223372036854775806 : index) : i64
    llvm.br ^bb1(%16 : i64)
  ^bb1(%18: i64):  // 2 preds: ^bb0, ^bb2
    %19 = llvm.icmp "slt" %18, %17 : i64
    llvm.cond_br %19, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%12, %14) : (i32, i32) -> ()
    %20 = llvm.getelementptr %5[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%20, %3 : !llvm.ptr, i64)] : i1
    llvm.call @llvm.aie2.acquire(%11, %14) : (i32, i32) -> ()
    %21 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%21, %3 : !llvm.ptr, i64)] : i1
    %22 = llvm.mul %7, %8 : i64
    %23 = llvm.getelementptr %6[1] : (!llvm.ptr) -> !llvm.ptr, bf16
    %24 = llvm.ptrtoint %23 : !llvm.ptr to i64
    %25 = llvm.mul %22, %24 : i64
    "llvm.intr.memcpy"(%21, %20, %25) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @llvm.aie2.release(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %14) : (i32, i32) -> ()
    %26 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%26, %3 : !llvm.ptr, i64)] : i1
    llvm.call @llvm.aie2.acquire(%11, %14) : (i32, i32) -> ()
    %27 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%27, %3 : !llvm.ptr, i64)] : i1
    "llvm.intr.memcpy"(%27, %26, %25) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @llvm.aie2.release(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %13) : (i32, i32) -> ()
    %28 = llvm.add %18, %15 : i64
    llvm.br ^bb1(%28 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%12, %14) : (i32, i32) -> ()
    %29 = llvm.getelementptr %5[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%29, %3 : !llvm.ptr, i64)] : i1
    llvm.call @llvm.aie2.acquire(%11, %14) : (i32, i32) -> ()
    %30 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%30, %3 : !llvm.ptr, i64)] : i1
    %31 = llvm.mul %7, %8 : i64
    %32 = llvm.getelementptr %6[1] : (!llvm.ptr) -> !llvm.ptr, bf16
    %33 = llvm.ptrtoint %32 : !llvm.ptr to i64
    %34 = llvm.mul %31, %33 : i64
    "llvm.intr.memcpy"(%30, %29, %34) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @llvm.aie2.release(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %13) : (i32, i32) -> ()
    llvm.return
  }
  llvm.func @core_0_4() {
    %0 = llvm.mlir.addressof @fifo_7_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @fifo_2_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @fifo_7_buff_0 : !llvm.ptr
    %3 = llvm.mlir.constant(32 : index) : i64
    %4 = llvm.mlir.constant(true) : i1
    %5 = llvm.mlir.addressof @fifo_2_cons_buff_0 : !llvm.ptr
    %6 = llvm.mlir.zero : !llvm.ptr
    %7 = llvm.mlir.constant(768 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(51 : i32) : i32
    %10 = llvm.mlir.constant(48 : i32) : i32
    %11 = llvm.mlir.constant(50 : i32) : i32
    %12 = llvm.mlir.constant(49 : i32) : i32
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.mlir.constant(-1 : i32) : i32
    %15 = llvm.mlir.constant(2 : index) : i64
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.mlir.constant(9223372036854775806 : index) : i64
    llvm.br ^bb1(%16 : i64)
  ^bb1(%18: i64):  // 2 preds: ^bb0, ^bb2
    %19 = llvm.icmp "slt" %18, %17 : i64
    llvm.cond_br %19, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%12, %14) : (i32, i32) -> ()
    %20 = llvm.getelementptr %5[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%20, %3 : !llvm.ptr, i64)] : i1
    llvm.call @llvm.aie2.acquire(%11, %14) : (i32, i32) -> ()
    %21 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%21, %3 : !llvm.ptr, i64)] : i1
    %22 = llvm.mul %7, %8 : i64
    %23 = llvm.getelementptr %6[1] : (!llvm.ptr) -> !llvm.ptr, bf16
    %24 = llvm.ptrtoint %23 : !llvm.ptr to i64
    %25 = llvm.mul %22, %24 : i64
    "llvm.intr.memcpy"(%21, %20, %25) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @llvm.aie2.release(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %14) : (i32, i32) -> ()
    %26 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%26, %3 : !llvm.ptr, i64)] : i1
    llvm.call @llvm.aie2.acquire(%11, %14) : (i32, i32) -> ()
    %27 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%27, %3 : !llvm.ptr, i64)] : i1
    "llvm.intr.memcpy"(%27, %26, %25) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @llvm.aie2.release(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %13) : (i32, i32) -> ()
    %28 = llvm.add %18, %15 : i64
    llvm.br ^bb1(%28 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%12, %14) : (i32, i32) -> ()
    %29 = llvm.getelementptr %5[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%29, %3 : !llvm.ptr, i64)] : i1
    llvm.call @llvm.aie2.acquire(%11, %14) : (i32, i32) -> ()
    %30 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%30, %3 : !llvm.ptr, i64)] : i1
    %31 = llvm.mul %7, %8 : i64
    %32 = llvm.getelementptr %6[1] : (!llvm.ptr) -> !llvm.ptr, bf16
    %33 = llvm.ptrtoint %32 : !llvm.ptr to i64
    %34 = llvm.mul %31, %33 : i64
    "llvm.intr.memcpy"(%30, %29, %34) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @llvm.aie2.release(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %13) : (i32, i32) -> ()
    llvm.return
  }
  llvm.func @core_0_3() {
    %0 = llvm.mlir.addressof @fifo_6_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @fifo_1_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @fifo_6_buff_0 : !llvm.ptr
    %3 = llvm.mlir.constant(32 : index) : i64
    %4 = llvm.mlir.constant(true) : i1
    %5 = llvm.mlir.addressof @fifo_1_cons_buff_0 : !llvm.ptr
    %6 = llvm.mlir.zero : !llvm.ptr
    %7 = llvm.mlir.constant(768 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(51 : i32) : i32
    %10 = llvm.mlir.constant(48 : i32) : i32
    %11 = llvm.mlir.constant(50 : i32) : i32
    %12 = llvm.mlir.constant(49 : i32) : i32
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.mlir.constant(-1 : i32) : i32
    %15 = llvm.mlir.constant(2 : index) : i64
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.mlir.constant(9223372036854775806 : index) : i64
    llvm.br ^bb1(%16 : i64)
  ^bb1(%18: i64):  // 2 preds: ^bb0, ^bb2
    %19 = llvm.icmp "slt" %18, %17 : i64
    llvm.cond_br %19, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%12, %14) : (i32, i32) -> ()
    %20 = llvm.getelementptr %5[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%20, %3 : !llvm.ptr, i64)] : i1
    llvm.call @llvm.aie2.acquire(%11, %14) : (i32, i32) -> ()
    %21 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%21, %3 : !llvm.ptr, i64)] : i1
    %22 = llvm.mul %7, %8 : i64
    %23 = llvm.getelementptr %6[1] : (!llvm.ptr) -> !llvm.ptr, bf16
    %24 = llvm.ptrtoint %23 : !llvm.ptr to i64
    %25 = llvm.mul %22, %24 : i64
    "llvm.intr.memcpy"(%21, %20, %25) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @llvm.aie2.release(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %14) : (i32, i32) -> ()
    %26 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%26, %3 : !llvm.ptr, i64)] : i1
    llvm.call @llvm.aie2.acquire(%11, %14) : (i32, i32) -> ()
    %27 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%27, %3 : !llvm.ptr, i64)] : i1
    "llvm.intr.memcpy"(%27, %26, %25) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @llvm.aie2.release(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %13) : (i32, i32) -> ()
    %28 = llvm.add %18, %15 : i64
    llvm.br ^bb1(%28 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%12, %14) : (i32, i32) -> ()
    %29 = llvm.getelementptr %5[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%29, %3 : !llvm.ptr, i64)] : i1
    llvm.call @llvm.aie2.acquire(%11, %14) : (i32, i32) -> ()
    %30 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%30, %3 : !llvm.ptr, i64)] : i1
    %31 = llvm.mul %7, %8 : i64
    %32 = llvm.getelementptr %6[1] : (!llvm.ptr) -> !llvm.ptr, bf16
    %33 = llvm.ptrtoint %32 : !llvm.ptr to i64
    %34 = llvm.mul %31, %33 : i64
    "llvm.intr.memcpy"(%30, %29, %34) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @llvm.aie2.release(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %13) : (i32, i32) -> ()
    llvm.return
  }
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @fifo_5_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @fifo_0_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @fifo_5_buff_0 : !llvm.ptr
    %3 = llvm.mlir.constant(32 : index) : i64
    %4 = llvm.mlir.constant(true) : i1
    %5 = llvm.mlir.addressof @fifo_0_cons_buff_0 : !llvm.ptr
    %6 = llvm.mlir.zero : !llvm.ptr
    %7 = llvm.mlir.constant(768 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(51 : i32) : i32
    %10 = llvm.mlir.constant(48 : i32) : i32
    %11 = llvm.mlir.constant(50 : i32) : i32
    %12 = llvm.mlir.constant(49 : i32) : i32
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.mlir.constant(-1 : i32) : i32
    %15 = llvm.mlir.constant(0 : index) : i64
    %16 = llvm.mlir.constant(9223372036854775806 : index) : i64
    %17 = llvm.mlir.constant(2 : index) : i64
    llvm.br ^bb1(%15 : i64)
  ^bb1(%18: i64):  // 2 preds: ^bb0, ^bb2
    %19 = llvm.icmp "slt" %18, %16 : i64
    llvm.cond_br %19, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%12, %14) : (i32, i32) -> ()
    %20 = llvm.getelementptr %5[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%20, %3 : !llvm.ptr, i64)] : i1
    llvm.call @llvm.aie2.acquire(%11, %14) : (i32, i32) -> ()
    %21 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%21, %3 : !llvm.ptr, i64)] : i1
    %22 = llvm.mul %7, %8 : i64
    %23 = llvm.getelementptr %6[1] : (!llvm.ptr) -> !llvm.ptr, bf16
    %24 = llvm.ptrtoint %23 : !llvm.ptr to i64
    %25 = llvm.mul %22, %24 : i64
    "llvm.intr.memcpy"(%21, %20, %25) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @llvm.aie2.release(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %14) : (i32, i32) -> ()
    %26 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%26, %3 : !llvm.ptr, i64)] : i1
    llvm.call @llvm.aie2.acquire(%11, %14) : (i32, i32) -> ()
    %27 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%27, %3 : !llvm.ptr, i64)] : i1
    "llvm.intr.memcpy"(%27, %26, %25) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @llvm.aie2.release(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %13) : (i32, i32) -> ()
    %28 = llvm.add %18, %17 : i64
    llvm.br ^bb1(%28 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%12, %14) : (i32, i32) -> ()
    %29 = llvm.getelementptr %5[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%29, %3 : !llvm.ptr, i64)] : i1
    llvm.call @llvm.aie2.acquire(%11, %14) : (i32, i32) -> ()
    %30 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x bf16>>
    llvm.intr.assume %4 ["align"(%30, %3 : !llvm.ptr, i64)] : i1
    %31 = llvm.mul %7, %8 : i64
    %32 = llvm.getelementptr %6[1] : (!llvm.ptr) -> !llvm.ptr, bf16
    %33 = llvm.ptrtoint %32 : !llvm.ptr to i64
    %34 = llvm.mul %31, %33 : i64
    "llvm.intr.memcpy"(%30, %29, %34) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @llvm.aie2.release(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %13) : (i32, i32) -> ()
    llvm.return
  }
}

