module attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @fifo_0_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<48 x array<128 x bf16>>
  llvm.mlir.global external @fifo_0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<48 x array<128 x bf16>>
  llvm.mlir.global external @fifo_1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<1 x array<1 x array<48 x array<128 x bf16>>>>
  llvm.mlir.global external @fifo_1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<1 x array<1 x array<48 x array<128 x bf16>>>>
  llvm.mlir.global external @fifo_2_buff_1() {addr_space = 0 : i32} : !llvm.array<8 x array<768 x bf16>>
  llvm.mlir.global external @fifo_2_buff_0() {addr_space = 0 : i32} : !llvm.array<8 x array<768 x bf16>>
  llvm.mlir.global external @fifo_2_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<8 x array<768 x bf16>>
  llvm.mlir.global external @fifo_2_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<8 x array<768 x bf16>>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.mlir.global external @fifo_3_cons() {addr_space = 0 : i32} : !llvm.array<1 x array<1 x array<8 x array<768 x bf16>>>>
  llvm.mlir.global external @fifo_3() {addr_space = 0 : i32} : !llvm.array<1 x array<1 x array<8 x array<768 x bf16>>>>
  llvm.mlir.global external @fifo_2_cons() {addr_space = 0 : i32} : !llvm.array<8 x array<768 x bf16>>
  llvm.mlir.global external @fifo_2() {addr_space = 0 : i32} : !llvm.array<8 x array<768 x bf16>>
  llvm.mlir.global external @fifo_1_cons() {addr_space = 0 : i32} : !llvm.array<1 x array<1 x array<48 x array<128 x bf16>>>>
  llvm.mlir.global external @fifo_1() {addr_space = 0 : i32} : !llvm.array<1 x array<1 x array<48 x array<128 x bf16>>>>
  llvm.mlir.global external @fifo_0_cons() {addr_space = 0 : i32} : !llvm.array<48 x array<128 x bf16>>
  llvm.mlir.global external @fifo_0() {addr_space = 0 : i32} : !llvm.array<48 x array<128 x bf16>>
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @fifo_2_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @fifo_0_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @fifo_2_buff_0 : !llvm.ptr
    %3 = llvm.mlir.constant(768 : index) : i64
    %4 = llvm.mlir.constant(32 : index) : i64
    %5 = llvm.mlir.constant(true) : i1
    %6 = llvm.mlir.addressof @fifo_0_cons_buff_0 : !llvm.ptr
    %7 = llvm.mlir.constant(128 : index) : i64
    %8 = llvm.mlir.constant(51 : i32) : i32
    %9 = llvm.mlir.constant(48 : i32) : i32
    %10 = llvm.mlir.constant(50 : i32) : i32
    %11 = llvm.mlir.constant(49 : i32) : i32
    %12 = llvm.mlir.constant(1 : i32) : i32
    %13 = llvm.mlir.constant(256 : index) : i64
    %14 = llvm.mlir.constant(16 : index) : i64
    %15 = llvm.mlir.constant(3 : index) : i64
    %16 = llvm.mlir.constant(8 : index) : i64
    %17 = llvm.mlir.constant(-1 : i32) : i32
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.mlir.constant(9223372036854775806 : index) : i64
    %21 = llvm.mlir.constant(2 : index) : i64
    llvm.br ^bb1(%18 : i64)
  ^bb1(%22: i64):  // 2 preds: ^bb0, ^bb20
    %23 = llvm.icmp "slt" %22, %20 : i64
    llvm.cond_br %23, ^bb2, ^bb21
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%11, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%10, %17) : (i32, i32) -> ()
    llvm.br ^bb3(%18 : i64)
  ^bb3(%24: i64):  // 2 preds: ^bb2, ^bb10
    %25 = llvm.icmp "slt" %24, %16 : i64
    llvm.cond_br %25, ^bb4(%18 : i64), ^bb11
  ^bb4(%26: i64):  // 2 preds: ^bb3, ^bb9
    %27 = llvm.icmp "slt" %26, %15 : i64
    llvm.cond_br %27, ^bb5(%18 : i64), ^bb10
  ^bb5(%28: i64):  // 2 preds: ^bb4, ^bb8
    %29 = llvm.icmp "slt" %28, %14 : i64
    llvm.cond_br %29, ^bb6(%18 : i64), ^bb9
  ^bb6(%30: i64):  // 2 preds: ^bb5, ^bb7
    %31 = llvm.icmp "slt" %30, %14 : i64
    llvm.cond_br %31, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %32 = llvm.mul %26, %14 overflow<nsw> : i64
    %33 = llvm.add %32, %28 : i64
    %34 = llvm.mul %24, %14 overflow<nsw> : i64
    %35 = llvm.add %34, %30 : i64
    %36 = llvm.getelementptr %6[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<48 x array<128 x bf16>>
    llvm.intr.assume %5 ["align"(%36, %4 : !llvm.ptr, i64)] : i1
    %37 = llvm.mul %33, %7 : i64
    %38 = llvm.add %37, %35 : i64
    %39 = llvm.getelementptr %36[%38] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %40 = llvm.load %39 : !llvm.ptr -> bf16
    %41 = llvm.mul %26, %13 overflow<nsw> : i64
    %42 = llvm.mul %28, %14 overflow<nsw> : i64
    %43 = llvm.add %41, %42 : i64
    %44 = llvm.add %43, %30 : i64
    %45 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x array<768 x bf16>>
    llvm.intr.assume %5 ["align"(%45, %4 : !llvm.ptr, i64)] : i1
    %46 = llvm.mul %24, %3 : i64
    %47 = llvm.add %46, %44 : i64
    %48 = llvm.getelementptr %45[%47] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    llvm.store %40, %48 : bf16, !llvm.ptr
    %49 = llvm.add %30, %19 : i64
    llvm.br ^bb6(%49 : i64)
  ^bb8:  // pred: ^bb6
    %50 = llvm.add %28, %19 : i64
    llvm.br ^bb5(%50 : i64)
  ^bb9:  // pred: ^bb5
    %51 = llvm.add %26, %19 : i64
    llvm.br ^bb4(%51 : i64)
  ^bb10:  // pred: ^bb4
    %52 = llvm.add %24, %19 : i64
    llvm.br ^bb3(%52 : i64)
  ^bb11:  // pred: ^bb3
    llvm.call @llvm.aie2.release(%9, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%8, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%10, %17) : (i32, i32) -> ()
    llvm.br ^bb12(%18 : i64)
  ^bb12(%53: i64):  // 2 preds: ^bb11, ^bb19
    %54 = llvm.icmp "slt" %53, %16 : i64
    llvm.cond_br %54, ^bb13(%18 : i64), ^bb20
  ^bb13(%55: i64):  // 2 preds: ^bb12, ^bb18
    %56 = llvm.icmp "slt" %55, %15 : i64
    llvm.cond_br %56, ^bb14(%18 : i64), ^bb19
  ^bb14(%57: i64):  // 2 preds: ^bb13, ^bb17
    %58 = llvm.icmp "slt" %57, %14 : i64
    llvm.cond_br %58, ^bb15(%18 : i64), ^bb18
  ^bb15(%59: i64):  // 2 preds: ^bb14, ^bb16
    %60 = llvm.icmp "slt" %59, %14 : i64
    llvm.cond_br %60, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %61 = llvm.mul %55, %14 overflow<nsw> : i64
    %62 = llvm.add %61, %57 : i64
    %63 = llvm.mul %53, %14 overflow<nsw> : i64
    %64 = llvm.add %63, %59 : i64
    %65 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<48 x array<128 x bf16>>
    llvm.intr.assume %5 ["align"(%65, %4 : !llvm.ptr, i64)] : i1
    %66 = llvm.mul %62, %7 : i64
    %67 = llvm.add %66, %64 : i64
    %68 = llvm.getelementptr %65[%67] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %69 = llvm.load %68 : !llvm.ptr -> bf16
    %70 = llvm.mul %55, %13 overflow<nsw> : i64
    %71 = llvm.mul %57, %14 overflow<nsw> : i64
    %72 = llvm.add %70, %71 : i64
    %73 = llvm.add %72, %59 : i64
    %74 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x array<768 x bf16>>
    llvm.intr.assume %5 ["align"(%74, %4 : !llvm.ptr, i64)] : i1
    %75 = llvm.mul %53, %3 : i64
    %76 = llvm.add %75, %73 : i64
    %77 = llvm.getelementptr %74[%76] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    llvm.store %69, %77 : bf16, !llvm.ptr
    %78 = llvm.add %59, %19 : i64
    llvm.br ^bb15(%78 : i64)
  ^bb17:  // pred: ^bb15
    %79 = llvm.add %57, %19 : i64
    llvm.br ^bb14(%79 : i64)
  ^bb18:  // pred: ^bb14
    %80 = llvm.add %55, %19 : i64
    llvm.br ^bb13(%80 : i64)
  ^bb19:  // pred: ^bb13
    %81 = llvm.add %53, %19 : i64
    llvm.br ^bb12(%81 : i64)
  ^bb20:  // pred: ^bb12
    llvm.call @llvm.aie2.release(%9, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%8, %12) : (i32, i32) -> ()
    %82 = llvm.add %22, %21 : i64
    llvm.br ^bb1(%82 : i64)
  ^bb21:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%11, %17) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%10, %17) : (i32, i32) -> ()
    llvm.br ^bb22(%18 : i64)
  ^bb22(%83: i64):  // 2 preds: ^bb21, ^bb29
    %84 = llvm.icmp "slt" %83, %16 : i64
    llvm.cond_br %84, ^bb23(%18 : i64), ^bb30
  ^bb23(%85: i64):  // 2 preds: ^bb22, ^bb28
    %86 = llvm.icmp "slt" %85, %15 : i64
    llvm.cond_br %86, ^bb24(%18 : i64), ^bb29
  ^bb24(%87: i64):  // 2 preds: ^bb23, ^bb27
    %88 = llvm.icmp "slt" %87, %14 : i64
    llvm.cond_br %88, ^bb25(%18 : i64), ^bb28
  ^bb25(%89: i64):  // 2 preds: ^bb24, ^bb26
    %90 = llvm.icmp "slt" %89, %14 : i64
    llvm.cond_br %90, ^bb26, ^bb27
  ^bb26:  // pred: ^bb25
    %91 = llvm.mul %85, %14 overflow<nsw> : i64
    %92 = llvm.add %91, %87 : i64
    %93 = llvm.mul %83, %14 overflow<nsw> : i64
    %94 = llvm.add %93, %89 : i64
    %95 = llvm.getelementptr %6[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<48 x array<128 x bf16>>
    llvm.intr.assume %5 ["align"(%95, %4 : !llvm.ptr, i64)] : i1
    %96 = llvm.mul %92, %7 : i64
    %97 = llvm.add %96, %94 : i64
    %98 = llvm.getelementptr %95[%97] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %99 = llvm.load %98 : !llvm.ptr -> bf16
    %100 = llvm.mul %85, %13 overflow<nsw> : i64
    %101 = llvm.mul %87, %14 overflow<nsw> : i64
    %102 = llvm.add %100, %101 : i64
    %103 = llvm.add %102, %89 : i64
    %104 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x array<768 x bf16>>
    llvm.intr.assume %5 ["align"(%104, %4 : !llvm.ptr, i64)] : i1
    %105 = llvm.mul %83, %3 : i64
    %106 = llvm.add %105, %103 : i64
    %107 = llvm.getelementptr %104[%106] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    llvm.store %99, %107 : bf16, !llvm.ptr
    %108 = llvm.add %89, %19 : i64
    llvm.br ^bb25(%108 : i64)
  ^bb27:  // pred: ^bb25
    %109 = llvm.add %87, %19 : i64
    llvm.br ^bb24(%109 : i64)
  ^bb28:  // pred: ^bb24
    %110 = llvm.add %85, %19 : i64
    llvm.br ^bb23(%110 : i64)
  ^bb29:  // pred: ^bb23
    %111 = llvm.add %83, %19 : i64
    llvm.br ^bb22(%111 : i64)
  ^bb30:  // pred: ^bb22
    llvm.call @llvm.aie2.release(%9, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%8, %12) : (i32, i32) -> ()
    llvm.return
  }
}

