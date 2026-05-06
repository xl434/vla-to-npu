; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@fifo_0_1_cons_buff_1 = external global [32 x [192 x bfloat]]
@fifo_0_1_cons_buff_0 = external global [32 x [192 x bfloat]]
@fifo_0_0_cons_buff_1 = external global [32 x [192 x bfloat]]
@fifo_0_0_cons_buff_0 = external global [32 x [192 x bfloat]]
@fifo_1_cons_buff_1 = external global [1 x [1 x [32 x [192 x bfloat]]]]
@fifo_1_cons_buff_0 = external global [1 x [1 x [32 x [192 x bfloat]]]]
@fifo_2_cons_buff_1 = external global [192 x [32 x bfloat]]
@fifo_2_cons_buff_0 = external global [192 x [32 x bfloat]]
@fifo_3_cons_buff_1 = external global [192 x [32 x bfloat]]
@fifo_3_cons_buff_0 = external global [192 x [32 x bfloat]]
@fifo_4_cons_buff_1 = external global [1 x [2 x [192 x [32 x bfloat]]]]
@fifo_4_cons_buff_0 = external global [1 x [2 x [192 x [32 x bfloat]]]]
@fifo_5_buff_1 = external global [32 x [32 x bfloat]]
@fifo_5_buff_0 = external global [32 x [32 x bfloat]]
@fifo_6_buff_1 = external global [32 x [32 x bfloat]]
@fifo_6_buff_0 = external global [32 x [32 x bfloat]]
@fifo_7_buff_1 = external global [1 x [2 x [32 x [32 x bfloat]]]]
@fifo_7_buff_0 = external global [1 x [2 x [32 x [32 x bfloat]]]]
@fifo_7_cons = external global [1 x [2 x [32 x [32 x bfloat]]]]
@fifo_7 = external global [1 x [2 x [32 x [32 x bfloat]]]]
@fifo_6_cons = external global [32 x [32 x bfloat]]
@fifo_6 = external global [32 x [32 x bfloat]]
@fifo_5_cons = external global [32 x [32 x bfloat]]
@fifo_5 = external global [32 x [32 x bfloat]]
@fifo_4_cons = external global [1 x [2 x [192 x [32 x bfloat]]]]
@fifo_4 = external global [1 x [2 x [192 x [32 x bfloat]]]]
@fifo_3_cons = external global [192 x [32 x bfloat]]
@fifo_3 = external global [192 x [32 x bfloat]]
@fifo_2_cons = external global [192 x [32 x bfloat]]
@fifo_2 = external global [192 x [32 x bfloat]]
@fifo_1_cons = external global [1 x [1 x [32 x [192 x bfloat]]]]
@fifo_1 = external global [1 x [1 x [32 x [192 x bfloat]]]]
@fifo_0_0_cons = external global [32 x [192 x bfloat]]
@fifo_0_1_cons = external global [32 x [192 x bfloat]]
@fifo_0 = external global [32 x [192 x bfloat]]

declare void @debug_i32(i32)

declare void @llvm.aie2.put.ms(i32, i32)

declare { i32, i32 } @llvm.aie2.get.ss()

declare void @llvm.aie2.mcd.write.vec(<16 x i32>, i32)

declare <16 x i32> @llvm.aie2.scd.read.vec(i32)

declare void @llvm.aie2.acquire(i32, i32)

declare void @llvm.aie2.release(i32, i32)

declare void @fill_zeros_bf16_32_32_vector(ptr)

declare void @matmul_scalar_bf16_bf16(ptr, ptr, ptr)

declare void @add_bf16_vector(ptr, ptr, ptr)

declare void @matmul_bf16_bf16(ptr, ptr, ptr)

define void @core_0_3() {
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %5, %4 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775806
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_0, i64 32) ]
  call void @fill_zeros_bf16_32_32_vector(ptr @fifo_6_buff_0)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_0_cons_buff_0, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_0_cons_buff_0, ptr @fifo_3_cons_buff_0, ptr @fifo_6_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_0_cons_buff_1, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_0_cons_buff_1, ptr @fifo_3_cons_buff_1, ptr @fifo_6_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_0_cons_buff_0, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_0_cons_buff_0, ptr @fifo_3_cons_buff_0, ptr @fifo_6_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_0_cons_buff_1, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_0_cons_buff_1, ptr @fifo_3_cons_buff_1, ptr @fifo_6_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_1, i64 32) ]
  call void @fill_zeros_bf16_32_32_vector(ptr @fifo_6_buff_1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_0_cons_buff_0, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_0_cons_buff_0, ptr @fifo_3_cons_buff_0, ptr @fifo_6_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_0_cons_buff_1, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_0_cons_buff_1, ptr @fifo_3_cons_buff_1, ptr @fifo_6_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_0_cons_buff_0, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_0_cons_buff_0, ptr @fifo_3_cons_buff_0, ptr @fifo_6_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_0_cons_buff_1, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_0_cons_buff_1, ptr @fifo_3_cons_buff_1, ptr @fifo_6_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %5 = add i64 %2, 2
  br label %1

6:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_0, i64 32) ]
  call void @fill_zeros_bf16_32_32_vector(ptr @fifo_6_buff_0)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_0_cons_buff_0, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_0_cons_buff_0, ptr @fifo_3_cons_buff_0, ptr @fifo_6_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_0_cons_buff_1, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_0_cons_buff_1, ptr @fifo_3_cons_buff_1, ptr @fifo_6_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_0_cons_buff_0, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_0_cons_buff_0, ptr @fifo_3_cons_buff_0, ptr @fifo_6_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_6_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_3_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_0_cons_buff_1, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_0_cons_buff_1, ptr @fifo_3_cons_buff_1, ptr @fifo_6_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  ret void
}

define void @core_0_2() {
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %5, %4 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775806
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_0, i64 32) ]
  call void @fill_zeros_bf16_32_32_vector(ptr @fifo_5_buff_0)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_1_cons_buff_0, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_1_cons_buff_0, ptr @fifo_2_cons_buff_0, ptr @fifo_5_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_1_cons_buff_1, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_1_cons_buff_1, ptr @fifo_2_cons_buff_1, ptr @fifo_5_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_1_cons_buff_0, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_1_cons_buff_0, ptr @fifo_2_cons_buff_0, ptr @fifo_5_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_1_cons_buff_1, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_1_cons_buff_1, ptr @fifo_2_cons_buff_1, ptr @fifo_5_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_1, i64 32) ]
  call void @fill_zeros_bf16_32_32_vector(ptr @fifo_5_buff_1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_1_cons_buff_0, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_1_cons_buff_0, ptr @fifo_2_cons_buff_0, ptr @fifo_5_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_1_cons_buff_1, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_1_cons_buff_1, ptr @fifo_2_cons_buff_1, ptr @fifo_5_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_1_cons_buff_0, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_1_cons_buff_0, ptr @fifo_2_cons_buff_0, ptr @fifo_5_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_1_cons_buff_1, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_1_cons_buff_1, ptr @fifo_2_cons_buff_1, ptr @fifo_5_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %5 = add i64 %2, 2
  br label %1

6:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_0, i64 32) ]
  call void @fill_zeros_bf16_32_32_vector(ptr @fifo_5_buff_0)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_1_cons_buff_0, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_1_cons_buff_0, ptr @fifo_2_cons_buff_0, ptr @fifo_5_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_1_cons_buff_1, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_1_cons_buff_1, ptr @fifo_2_cons_buff_1, ptr @fifo_5_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_1_cons_buff_0, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_1_cons_buff_0, ptr @fifo_2_cons_buff_0, ptr @fifo_5_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_5_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_1_cons_buff_1, i64 32) ]
  call void @matmul_bf16_bf16(ptr @fifo_0_1_cons_buff_1, ptr @fifo_2_cons_buff_1, ptr @fifo_5_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
