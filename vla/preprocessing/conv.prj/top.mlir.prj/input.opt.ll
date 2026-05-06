; ModuleID = '/home/dl2239/vla-to-npu/vla/preprocessing/conv.prj/top.mlir.prj/input.llpeanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2"

@fifo_0_cons_buff_1 = external global [64 x [64 x bfloat]]
@fifo_0_cons_buff_0 = external global [64 x [64 x bfloat]]
@fifo_1_cons_buff_1 = external global [64 x [64 x bfloat]]
@fifo_1_cons_buff_0 = external global [64 x [64 x bfloat]]
@fifo_2_cons_buff_1 = external global [64 x [64 x bfloat]]
@fifo_2_cons_buff_0 = external global [64 x [64 x bfloat]]
@fifo_3_cons_buff_1 = external global [64 x [64 x bfloat]]
@fifo_3_cons_buff_0 = external global [64 x [64 x bfloat]]
@fifo_5_cons_buff_1 = external global [64 x [64 x bfloat]]
@fifo_5_cons_buff_0 = external global [64 x [64 x bfloat]]
@fifo_6_cons_buff_1 = external global [64 x [64 x bfloat]]
@fifo_6_cons_buff_0 = external global [64 x [64 x bfloat]]
@fifo_7_cons_buff_1 = external global [64 x [64 x bfloat]]
@fifo_7_cons_buff_0 = external global [64 x [64 x bfloat]]
@fifo_8_cons_buff_1 = external global [64 x [64 x bfloat]]
@fifo_8_cons_buff_0 = external global [64 x [64 x bfloat]]
@fifo_10_cons_buff_1 = external global [64 x [64 x bfloat]]
@fifo_10_cons_buff_0 = external global [64 x [64 x bfloat]]
@fifo_11_cons_buff_1 = external global [64 x [64 x bfloat]]
@fifo_11_cons_buff_0 = external global [64 x [64 x bfloat]]
@fifo_12_cons_buff_1 = external global [64 x [64 x bfloat]]
@fifo_12_cons_buff_0 = external global [64 x [64 x bfloat]]
@fifo_13_cons_buff_1 = external global [64 x [64 x bfloat]]
@fifo_13_cons_buff_0 = external global [64 x [64 x bfloat]]
@fifo_15_cons_buff_1 = external global [64 x [64 x bfloat]]
@fifo_15_cons_buff_0 = external global [64 x [64 x bfloat]]
@fifo_16_cons_buff_1 = external global [64 x [64 x bfloat]]
@fifo_16_cons_buff_0 = external global [64 x [64 x bfloat]]
@fifo_17_cons_buff_1 = external global [64 x [64 x bfloat]]
@fifo_17_cons_buff_0 = external global [64 x [64 x bfloat]]
@fifo_18_cons_buff_1 = external global [64 x [64 x bfloat]]
@fifo_18_cons_buff_0 = external global [64 x [64 x bfloat]]
@fifo_20_15_cons_buff_1 = external global [16 x [16 x bfloat]]
@fifo_20_15_cons_buff_0 = external global [16 x [16 x bfloat]]
@fifo_20_14_cons_buff_1 = external global [16 x [16 x bfloat]]
@fifo_20_14_cons_buff_0 = external global [16 x [16 x bfloat]]
@fifo_20_13_cons_buff_1 = external global [16 x [16 x bfloat]]
@fifo_20_13_cons_buff_0 = external global [16 x [16 x bfloat]]
@fifo_20_12_cons_buff_1 = external global [16 x [16 x bfloat]]
@fifo_20_12_cons_buff_0 = external global [16 x [16 x bfloat]]
@fifo_20_11_cons_buff_1 = external global [16 x [16 x bfloat]]
@fifo_20_11_cons_buff_0 = external global [16 x [16 x bfloat]]
@fifo_20_10_cons_buff_1 = external global [16 x [16 x bfloat]]
@fifo_20_10_cons_buff_0 = external global [16 x [16 x bfloat]]
@fifo_20_9_cons_buff_1 = external global [16 x [16 x bfloat]]
@fifo_20_9_cons_buff_0 = external global [16 x [16 x bfloat]]
@fifo_20_8_cons_buff_1 = external global [16 x [16 x bfloat]]
@fifo_20_8_cons_buff_0 = external global [16 x [16 x bfloat]]
@fifo_20_7_cons_buff_1 = external global [16 x [16 x bfloat]]
@fifo_20_7_cons_buff_0 = external global [16 x [16 x bfloat]]
@fifo_20_6_cons_buff_1 = external global [16 x [16 x bfloat]]
@fifo_20_6_cons_buff_0 = external global [16 x [16 x bfloat]]
@fifo_20_5_cons_buff_1 = external global [16 x [16 x bfloat]]
@fifo_20_5_cons_buff_0 = external global [16 x [16 x bfloat]]
@fifo_20_4_cons_buff_1 = external global [16 x [16 x bfloat]]
@fifo_20_4_cons_buff_0 = external global [16 x [16 x bfloat]]
@fifo_20_3_cons_buff_1 = external global [16 x [16 x bfloat]]
@fifo_20_3_cons_buff_0 = external global [16 x [16 x bfloat]]
@fifo_20_2_cons_buff_1 = external global [16 x [16 x bfloat]]
@fifo_20_2_cons_buff_0 = external global [16 x [16 x bfloat]]
@fifo_20_1_cons_buff_1 = external global [16 x [16 x bfloat]]
@fifo_20_1_cons_buff_0 = external global [16 x [16 x bfloat]]
@fifo_20_0_cons_buff_1 = external global [16 x [16 x bfloat]]
@fifo_20_0_cons_buff_0 = external global [16 x [16 x bfloat]]
@fifo_22_buff_1 = external global [4 x [4 x bfloat]]
@fifo_22_buff_0 = external global [4 x [4 x bfloat]]
@fifo_23_buff_1 = external global [4 x [4 x bfloat]]
@fifo_23_buff_0 = external global [4 x [4 x bfloat]]
@fifo_24_buff_1 = external global [4 x [4 x bfloat]]
@fifo_24_buff_0 = external global [4 x [4 x bfloat]]
@fifo_25_buff_1 = external global [4 x [4 x bfloat]]
@fifo_25_buff_0 = external global [4 x [4 x bfloat]]
@fifo_27_buff_1 = external global [4 x [4 x bfloat]]
@fifo_27_buff_0 = external global [4 x [4 x bfloat]]
@fifo_28_buff_1 = external global [4 x [4 x bfloat]]
@fifo_28_buff_0 = external global [4 x [4 x bfloat]]
@fifo_29_buff_1 = external global [4 x [4 x bfloat]]
@fifo_29_buff_0 = external global [4 x [4 x bfloat]]
@fifo_30_buff_1 = external global [4 x [4 x bfloat]]
@fifo_30_buff_0 = external global [4 x [4 x bfloat]]
@fifo_32_buff_1 = external global [4 x [4 x bfloat]]
@fifo_32_buff_0 = external global [4 x [4 x bfloat]]
@fifo_33_buff_1 = external global [4 x [4 x bfloat]]
@fifo_33_buff_0 = external global [4 x [4 x bfloat]]
@fifo_34_buff_1 = external global [4 x [4 x bfloat]]
@fifo_34_buff_0 = external global [4 x [4 x bfloat]]
@fifo_35_buff_1 = external global [4 x [4 x bfloat]]
@fifo_35_buff_0 = external global [4 x [4 x bfloat]]
@fifo_37_buff_1 = external global [4 x [4 x bfloat]]
@fifo_37_buff_0 = external global [4 x [4 x bfloat]]
@fifo_38_buff_1 = external global [4 x [4 x bfloat]]
@fifo_38_buff_0 = external global [4 x [4 x bfloat]]
@fifo_39_buff_1 = external global [4 x [4 x bfloat]]
@fifo_39_buff_0 = external global [4 x [4 x bfloat]]
@fifo_40_buff_1 = external global [4 x [4 x bfloat]]
@fifo_40_buff_0 = external global [4 x [4 x bfloat]]

; Function Attrs: nounwind
declare void @llvm.aie2.acquire(i32, i32) #0

; Function Attrs: nounwind
declare void @llvm.aie2.release(i32, i32) #0

declare void @conv(ptr, ptr, ptr) local_unnamed_addr

define void @core_3_5() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_40_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_4_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_18_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_18_cons_buff_0, ptr nonnull @fifo_20_4_cons_buff_0, ptr nonnull @fifo_40_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_40_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_4_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_18_cons_buff_1, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_18_cons_buff_1, ptr nonnull @fifo_20_4_cons_buff_1, ptr nonnull @fifo_40_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_40_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_4_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_18_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_18_cons_buff_0, ptr nonnull @fifo_20_4_cons_buff_0, ptr nonnull @fifo_40_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  ret void
}

define void @core_3_4() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_39_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_13_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_17_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_17_cons_buff_0, ptr nonnull @fifo_20_13_cons_buff_0, ptr nonnull @fifo_39_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_39_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_13_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_17_cons_buff_1, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_17_cons_buff_1, ptr nonnull @fifo_20_13_cons_buff_1, ptr nonnull @fifo_39_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_39_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_13_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_17_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_17_cons_buff_0, ptr nonnull @fifo_20_13_cons_buff_0, ptr nonnull @fifo_39_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  ret void
}

define void @core_3_3() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_38_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_14_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_16_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_16_cons_buff_0, ptr nonnull @fifo_20_14_cons_buff_0, ptr nonnull @fifo_38_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_38_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_14_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_16_cons_buff_1, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_16_cons_buff_1, ptr nonnull @fifo_20_14_cons_buff_1, ptr nonnull @fifo_38_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_38_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_14_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_16_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_16_cons_buff_0, ptr nonnull @fifo_20_14_cons_buff_0, ptr nonnull @fifo_38_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  ret void
}

define void @core_3_2() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_37_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_11_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_15_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_15_cons_buff_0, ptr nonnull @fifo_20_11_cons_buff_0, ptr nonnull @fifo_37_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_37_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_11_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_15_cons_buff_1, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_15_cons_buff_1, ptr nonnull @fifo_20_11_cons_buff_1, ptr nonnull @fifo_37_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_37_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_11_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_15_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_15_cons_buff_0, ptr nonnull @fifo_20_11_cons_buff_0, ptr nonnull @fifo_37_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  ret void
}

define void @core_2_5() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_35_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_3_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_13_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_13_cons_buff_0, ptr nonnull @fifo_20_3_cons_buff_0, ptr nonnull @fifo_35_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_35_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_3_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_13_cons_buff_1, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_13_cons_buff_1, ptr nonnull @fifo_20_3_cons_buff_1, ptr nonnull @fifo_35_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_35_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_3_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_13_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_13_cons_buff_0, ptr nonnull @fifo_20_3_cons_buff_0, ptr nonnull @fifo_35_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  ret void
}

define void @core_2_4() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_34_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_2_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_12_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_12_cons_buff_0, ptr nonnull @fifo_20_2_cons_buff_0, ptr nonnull @fifo_34_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_34_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_2_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_12_cons_buff_1, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_12_cons_buff_1, ptr nonnull @fifo_20_2_cons_buff_1, ptr nonnull @fifo_34_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_34_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_2_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_12_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_12_cons_buff_0, ptr nonnull @fifo_20_2_cons_buff_0, ptr nonnull @fifo_34_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  ret void
}

define void @core_2_3() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_33_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_8_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_11_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_11_cons_buff_0, ptr nonnull @fifo_20_8_cons_buff_0, ptr nonnull @fifo_33_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_33_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_8_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_11_cons_buff_1, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_11_cons_buff_1, ptr nonnull @fifo_20_8_cons_buff_1, ptr nonnull @fifo_33_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_33_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_8_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_11_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_11_cons_buff_0, ptr nonnull @fifo_20_8_cons_buff_0, ptr nonnull @fifo_33_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  ret void
}

define void @core_2_2() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_32_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_5_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_10_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_10_cons_buff_0, ptr nonnull @fifo_20_5_cons_buff_0, ptr nonnull @fifo_32_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_32_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_5_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_10_cons_buff_1, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_10_cons_buff_1, ptr nonnull @fifo_20_5_cons_buff_1, ptr nonnull @fifo_32_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_32_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_5_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_10_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_10_cons_buff_0, ptr nonnull @fifo_20_5_cons_buff_0, ptr nonnull @fifo_32_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  ret void
}

define void @core_1_5() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_30_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_1_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_8_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_8_cons_buff_0, ptr nonnull @fifo_20_1_cons_buff_0, ptr nonnull @fifo_30_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_30_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_1_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_8_cons_buff_1, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_8_cons_buff_1, ptr nonnull @fifo_20_1_cons_buff_1, ptr nonnull @fifo_30_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_30_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_1_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_8_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_8_cons_buff_0, ptr nonnull @fifo_20_1_cons_buff_0, ptr nonnull @fifo_30_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  ret void
}

define void @core_1_4() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_29_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_9_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_7_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_7_cons_buff_0, ptr nonnull @fifo_20_9_cons_buff_0, ptr nonnull @fifo_29_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_29_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_9_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_7_cons_buff_1, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_7_cons_buff_1, ptr nonnull @fifo_20_9_cons_buff_1, ptr nonnull @fifo_29_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_29_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_9_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_7_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_7_cons_buff_0, ptr nonnull @fifo_20_9_cons_buff_0, ptr nonnull @fifo_29_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  ret void
}

define void @core_1_3() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_28_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_10_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_6_cons_buff_0, ptr nonnull @fifo_20_10_cons_buff_0, ptr nonnull @fifo_28_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_28_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_10_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_cons_buff_1, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_6_cons_buff_1, ptr nonnull @fifo_20_10_cons_buff_1, ptr nonnull @fifo_28_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_28_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_10_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_6_cons_buff_0, ptr nonnull @fifo_20_10_cons_buff_0, ptr nonnull @fifo_28_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  ret void
}

define void @core_1_2() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_27_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_12_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_5_cons_buff_0, ptr nonnull @fifo_20_12_cons_buff_0, ptr nonnull @fifo_27_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_27_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_12_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_cons_buff_1, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_5_cons_buff_1, ptr nonnull @fifo_20_12_cons_buff_1, ptr nonnull @fifo_27_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_27_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_12_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_5_cons_buff_0, ptr nonnull @fifo_20_12_cons_buff_0, ptr nonnull @fifo_27_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  ret void
}

define void @core_0_5() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_25_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_0_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_3_cons_buff_0, ptr nonnull @fifo_20_0_cons_buff_0, ptr nonnull @fifo_25_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_25_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_0_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_1, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_3_cons_buff_1, ptr nonnull @fifo_20_0_cons_buff_1, ptr nonnull @fifo_25_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_25_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_0_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_3_cons_buff_0, ptr nonnull @fifo_20_0_cons_buff_0, ptr nonnull @fifo_25_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  ret void
}

define void @core_0_4() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_24_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_6_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_2_cons_buff_0, ptr nonnull @fifo_20_6_cons_buff_0, ptr nonnull @fifo_24_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_24_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_6_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_1, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_2_cons_buff_1, ptr nonnull @fifo_20_6_cons_buff_1, ptr nonnull @fifo_24_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_24_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_6_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_2_cons_buff_0, ptr nonnull @fifo_20_6_cons_buff_0, ptr nonnull @fifo_24_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  ret void
}

define void @core_0_3() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_23_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_7_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_1_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_1_cons_buff_0, ptr nonnull @fifo_20_7_cons_buff_0, ptr nonnull @fifo_23_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_23_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_7_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_1_cons_buff_1, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_1_cons_buff_1, ptr nonnull @fifo_20_7_cons_buff_1, ptr nonnull @fifo_23_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_23_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_7_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_1_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_1_cons_buff_0, ptr nonnull @fifo_20_7_cons_buff_0, ptr nonnull @fifo_23_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  ret void
}

define void @core_0_2() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_22_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_15_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_0_cons_buff_0, ptr nonnull @fifo_20_15_cons_buff_0, ptr nonnull @fifo_22_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_22_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_15_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_cons_buff_1, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_0_cons_buff_1, ptr nonnull @fifo_20_15_cons_buff_1, ptr nonnull @fifo_22_buff_1)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_22_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_20_15_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_cons_buff_0, i64 32) ]
  tail call void @conv(ptr nonnull @fifo_0_cons_buff_0, ptr nonnull @fifo_20_15_cons_buff_0, ptr nonnull @fifo_22_buff_0)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #1

attributes #0 = { nounwind }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
