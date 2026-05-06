; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@fifo_0_cons_buff_1 = external global [1 x [768 x bfloat]]
@fifo_0_cons_buff_0 = external global [1 x [768 x bfloat]]
@fifo_1_cons_buff_1 = external global [1 x [768 x bfloat]]
@fifo_1_cons_buff_0 = external global [1 x [768 x bfloat]]
@fifo_2_cons_buff_1 = external global [1 x [768 x bfloat]]
@fifo_2_cons_buff_0 = external global [1 x [768 x bfloat]]
@fifo_3_cons_buff_1 = external global [1 x [768 x bfloat]]
@fifo_3_cons_buff_0 = external global [1 x [768 x bfloat]]
@fifo_4_cons_buff_1 = external global [1 x [4 x [1 x [768 x bfloat]]]]
@fifo_4_cons_buff_0 = external global [1 x [4 x [1 x [768 x bfloat]]]]
@fifo_5_buff_1 = external global [1 x [768 x bfloat]]
@fifo_5_buff_0 = external global [1 x [768 x bfloat]]
@fifo_6_buff_1 = external global [1 x [768 x bfloat]]
@fifo_6_buff_0 = external global [1 x [768 x bfloat]]
@fifo_7_buff_1 = external global [1 x [768 x bfloat]]
@fifo_7_buff_0 = external global [1 x [768 x bfloat]]
@fifo_8_buff_1 = external global [1 x [768 x bfloat]]
@fifo_8_buff_0 = external global [1 x [768 x bfloat]]
@fifo_9_buff_1 = external global [1 x [4 x [1 x [768 x bfloat]]]]
@fifo_9_buff_0 = external global [1 x [4 x [1 x [768 x bfloat]]]]
@fifo_9_cons = external global [1 x [4 x [1 x [768 x bfloat]]]]
@fifo_9 = external global [1 x [4 x [1 x [768 x bfloat]]]]
@fifo_8_cons = external global [1 x [768 x bfloat]]
@fifo_8 = external global [1 x [768 x bfloat]]
@fifo_7_cons = external global [1 x [768 x bfloat]]
@fifo_7 = external global [1 x [768 x bfloat]]
@fifo_6_cons = external global [1 x [768 x bfloat]]
@fifo_6 = external global [1 x [768 x bfloat]]
@fifo_5_cons = external global [1 x [768 x bfloat]]
@fifo_5 = external global [1 x [768 x bfloat]]
@fifo_4_cons = external global [1 x [4 x [1 x [768 x bfloat]]]]
@fifo_4 = external global [1 x [4 x [1 x [768 x bfloat]]]]
@fifo_3_cons = external global [1 x [768 x bfloat]]
@fifo_3 = external global [1 x [768 x bfloat]]
@fifo_2_cons = external global [1 x [768 x bfloat]]
@fifo_2 = external global [1 x [768 x bfloat]]
@fifo_1_cons = external global [1 x [768 x bfloat]]
@fifo_1 = external global [1 x [768 x bfloat]]
@fifo_0_cons = external global [1 x [768 x bfloat]]
@fifo_0 = external global [1 x [768 x bfloat]]

declare void @debug_i32(i32)

declare void @llvm.aie2.put.ms(i32, i32)

declare { i32, i32 } @llvm.aie2.get.ss()

declare void @llvm.aie2.mcd.write.vec(<16 x i32>, i32)

declare <16 x i32> @llvm.aie2.scd.read.vec(i32)

declare void @llvm.aie2.acquire(i32, i32)

declare void @llvm.aie2.release(i32, i32)

define void @core_0_5() {
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %5, %4 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775806
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_0, i64 32) ]
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_8_buff_0, i64 32) ]
  call void @llvm.memcpy.p0.p0.i64(ptr @fifo_8_buff_0, ptr @fifo_3_cons_buff_0, i64 1536, i1 false)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_1, i64 32) ]
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_8_buff_1, i64 32) ]
  call void @llvm.memcpy.p0.p0.i64(ptr @fifo_8_buff_1, ptr @fifo_3_cons_buff_1, i64 1536, i1 false)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  %5 = add i64 %2, 2
  br label %1

6:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_0, i64 32) ]
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_8_buff_0, i64 32) ]
  call void @llvm.memcpy.p0.p0.i64(ptr @fifo_8_buff_0, ptr @fifo_3_cons_buff_0, i64 1536, i1 false)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

define void @core_0_4() {
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %5, %4 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775806
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_0, i64 32) ]
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_7_buff_0, i64 32) ]
  call void @llvm.memcpy.p0.p0.i64(ptr @fifo_7_buff_0, ptr @fifo_2_cons_buff_0, i64 1536, i1 false)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_1, i64 32) ]
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_7_buff_1, i64 32) ]
  call void @llvm.memcpy.p0.p0.i64(ptr @fifo_7_buff_1, ptr @fifo_2_cons_buff_1, i64 1536, i1 false)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  %5 = add i64 %2, 2
  br label %1

6:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_0, i64 32) ]
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_7_buff_0, i64 32) ]
  call void @llvm.memcpy.p0.p0.i64(ptr @fifo_7_buff_0, ptr @fifo_2_cons_buff_0, i64 1536, i1 false)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

define void @core_0_3() {
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %5, %4 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775806
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_1_cons_buff_0, i64 32) ]
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_0, i64 32) ]
  call void @llvm.memcpy.p0.p0.i64(ptr @fifo_6_buff_0, ptr @fifo_1_cons_buff_0, i64 1536, i1 false)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_1_cons_buff_1, i64 32) ]
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_1, i64 32) ]
  call void @llvm.memcpy.p0.p0.i64(ptr @fifo_6_buff_1, ptr @fifo_1_cons_buff_1, i64 1536, i1 false)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  %5 = add i64 %2, 2
  br label %1

6:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_1_cons_buff_0, i64 32) ]
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_0, i64 32) ]
  call void @llvm.memcpy.p0.p0.i64(ptr @fifo_6_buff_0, ptr @fifo_1_cons_buff_0, i64 1536, i1 false)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

define void @core_0_2() {
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %5, %4 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775806
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_cons_buff_0, i64 32) ]
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_0, i64 32) ]
  call void @llvm.memcpy.p0.p0.i64(ptr @fifo_5_buff_0, ptr @fifo_0_cons_buff_0, i64 1536, i1 false)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_cons_buff_1, i64 32) ]
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_1, i64 32) ]
  call void @llvm.memcpy.p0.p0.i64(ptr @fifo_5_buff_1, ptr @fifo_0_cons_buff_1, i64 1536, i1 false)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  %5 = add i64 %2, 2
  br label %1

6:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_cons_buff_0, i64 32) ]
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_0, i64 32) ]
  call void @llvm.memcpy.p0.p0.i64(ptr @fifo_5_buff_0, ptr @fifo_0_cons_buff_0, i64 1536, i1 false)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
