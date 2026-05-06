; ModuleID = '/home/dl2239/vla-to-npu/vla/connector/copy.prj/top.mlir.prj/input.llpeanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2"

@fifo_0_cons_buff_1 = external global [1 x [768 x bfloat]]
@fifo_0_cons_buff_0 = external global [1 x [768 x bfloat]]
@fifo_1_cons_buff_1 = external global [1 x [768 x bfloat]]
@fifo_1_cons_buff_0 = external global [1 x [768 x bfloat]]
@fifo_2_cons_buff_1 = external global [1 x [768 x bfloat]]
@fifo_2_cons_buff_0 = external global [1 x [768 x bfloat]]
@fifo_3_cons_buff_1 = external global [1 x [768 x bfloat]]
@fifo_3_cons_buff_0 = external global [1 x [768 x bfloat]]
@fifo_5_buff_1 = external global [1 x [768 x bfloat]]
@fifo_5_buff_0 = external global [1 x [768 x bfloat]]
@fifo_6_buff_1 = external global [1 x [768 x bfloat]]
@fifo_6_buff_0 = external global [1 x [768 x bfloat]]
@fifo_7_buff_1 = external global [1 x [768 x bfloat]]
@fifo_7_buff_0 = external global [1 x [768 x bfloat]]
@fifo_8_buff_1 = external global [1 x [768 x bfloat]]
@fifo_8_buff_0 = external global [1 x [768 x bfloat]]

; Function Attrs: nounwind
declare void @llvm.aie2.acquire(i32, i32) #0

; Function Attrs: nounwind
declare void @llvm.aie2.release(i32, i32) #0

; Function Attrs: nounwind
define void @core_0_5() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_0, i64 32) ]
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_8_buff_0, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 32 dereferenceable(1536) @fifo_8_buff_0, ptr noundef nonnull align 32 dereferenceable(1536) @fifo_3_cons_buff_0, i64 1536, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_1, i64 32) ]
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_8_buff_1, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 32 dereferenceable(1536) @fifo_8_buff_1, ptr noundef nonnull align 32 dereferenceable(1536) @fifo_3_cons_buff_1, i64 1536, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_0, i64 32) ]
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_8_buff_0, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 32 dereferenceable(1536) @fifo_8_buff_0, ptr noundef nonnull align 32 dereferenceable(1536) @fifo_3_cons_buff_0, i64 1536, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

; Function Attrs: nounwind
define void @core_0_4() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_0, i64 32) ]
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_7_buff_0, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 32 dereferenceable(1536) @fifo_7_buff_0, ptr noundef nonnull align 32 dereferenceable(1536) @fifo_2_cons_buff_0, i64 1536, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_1, i64 32) ]
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_7_buff_1, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 32 dereferenceable(1536) @fifo_7_buff_1, ptr noundef nonnull align 32 dereferenceable(1536) @fifo_2_cons_buff_1, i64 1536, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_0, i64 32) ]
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_7_buff_0, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 32 dereferenceable(1536) @fifo_7_buff_0, ptr noundef nonnull align 32 dereferenceable(1536) @fifo_2_cons_buff_0, i64 1536, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

; Function Attrs: nounwind
define void @core_0_3() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_1_cons_buff_0, i64 32) ]
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_0, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 32 dereferenceable(1536) @fifo_6_buff_0, ptr noundef nonnull align 32 dereferenceable(1536) @fifo_1_cons_buff_0, i64 1536, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_1_cons_buff_1, i64 32) ]
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_1, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 32 dereferenceable(1536) @fifo_6_buff_1, ptr noundef nonnull align 32 dereferenceable(1536) @fifo_1_cons_buff_1, i64 1536, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_1_cons_buff_0, i64 32) ]
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_0, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 32 dereferenceable(1536) @fifo_6_buff_0, ptr noundef nonnull align 32 dereferenceable(1536) @fifo_1_cons_buff_0, i64 1536, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

; Function Attrs: nounwind
define void @core_0_2() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_cons_buff_0, i64 32) ]
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_0, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 32 dereferenceable(1536) @fifo_5_buff_0, ptr noundef nonnull align 32 dereferenceable(1536) @fifo_0_cons_buff_0, i64 1536, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_cons_buff_1, i64 32) ]
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_1, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 32 dereferenceable(1536) @fifo_5_buff_1, ptr noundef nonnull align 32 dereferenceable(1536) @fifo_0_cons_buff_1, i64 1536, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_cons_buff_0, i64 32) ]
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_0, i64 32) ]
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 32 dereferenceable(1536) @fifo_5_buff_0, ptr noundef nonnull align 32 dereferenceable(1536) @fifo_0_cons_buff_0, i64 1536, i1 false)
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #2

attributes #0 = { nounwind }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
