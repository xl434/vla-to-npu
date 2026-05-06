; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@fifo_0_cons_buff_1 = external global [48 x [128 x bfloat]]
@fifo_0_cons_buff_0 = external global [48 x [128 x bfloat]]
@fifo_1_cons_buff_1 = external global [1 x [1 x [48 x [128 x bfloat]]]]
@fifo_1_cons_buff_0 = external global [1 x [1 x [48 x [128 x bfloat]]]]
@fifo_2_buff_1 = external global [8 x [768 x bfloat]]
@fifo_2_buff_0 = external global [8 x [768 x bfloat]]
@fifo_2_cons_buff_1 = external global [8 x [768 x bfloat]]
@fifo_2_cons_buff_0 = external global [8 x [768 x bfloat]]
@fifo_3_cons = external global [1 x [1 x [8 x [768 x bfloat]]]]
@fifo_3 = external global [1 x [1 x [8 x [768 x bfloat]]]]
@fifo_2_cons = external global [8 x [768 x bfloat]]
@fifo_2 = external global [8 x [768 x bfloat]]
@fifo_1_cons = external global [1 x [1 x [48 x [128 x bfloat]]]]
@fifo_1 = external global [1 x [1 x [48 x [128 x bfloat]]]]
@fifo_0_cons = external global [48 x [128 x bfloat]]
@fifo_0 = external global [48 x [128 x bfloat]]

declare void @debug_i32(i32)

declare void @llvm.aie2.put.ms(i32, i32)

declare { i32, i32 } @llvm.aie2.get.ss()

declare void @llvm.aie2.mcd.write.vec(<16 x i32>, i32)

declare <16 x i32> @llvm.aie2.scd.read.vec(i32)

declare void @llvm.aie2.acquire(i32, i32)

declare void @llvm.aie2.release(i32, i32)

define void @core_0_2() {
  br label %1

1:                                                ; preds = %76, %0
  %2 = phi i64 [ %77, %76 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775806
  br i1 %3, label %4, label %78

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  br label %5

5:                                                ; preds = %38, %4
  %6 = phi i64 [ %39, %38 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 8
  br i1 %7, label %8, label %40

8:                                                ; preds = %36, %5
  %9 = phi i64 [ %37, %36 ], [ 0, %5 ]
  %10 = icmp slt i64 %9, 3
  br i1 %10, label %11, label %38

11:                                               ; preds = %34, %8
  %12 = phi i64 [ %35, %34 ], [ 0, %8 ]
  %13 = icmp slt i64 %12, 16
  br i1 %13, label %14, label %36

14:                                               ; preds = %17, %11
  %15 = phi i64 [ %33, %17 ], [ 0, %11 ]
  %16 = icmp slt i64 %15, 16
  br i1 %16, label %17, label %34

17:                                               ; preds = %14
  %18 = mul nsw i64 %9, 16
  %19 = add i64 %18, %12
  %20 = mul nsw i64 %6, 16
  %21 = add i64 %20, %15
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_cons_buff_0, i64 32) ]
  %22 = mul i64 %19, 128
  %23 = add i64 %22, %21
  %24 = getelementptr bfloat, ptr @fifo_0_cons_buff_0, i64 %23
  %25 = load bfloat, ptr %24, align 2
  %26 = mul nsw i64 %9, 256
  %27 = mul nsw i64 %12, 16
  %28 = add i64 %26, %27
  %29 = add i64 %28, %15
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_buff_0, i64 32) ]
  %30 = mul i64 %6, 768
  %31 = add i64 %30, %29
  %32 = getelementptr bfloat, ptr @fifo_2_buff_0, i64 %31
  store bfloat %25, ptr %32, align 2
  %33 = add i64 %15, 1
  br label %14

34:                                               ; preds = %14
  %35 = add i64 %12, 1
  br label %11

36:                                               ; preds = %11
  %37 = add i64 %9, 1
  br label %8

38:                                               ; preds = %8
  %39 = add i64 %6, 1
  br label %5

40:                                               ; preds = %5
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  br label %41

41:                                               ; preds = %74, %40
  %42 = phi i64 [ %75, %74 ], [ 0, %40 ]
  %43 = icmp slt i64 %42, 8
  br i1 %43, label %44, label %76

44:                                               ; preds = %72, %41
  %45 = phi i64 [ %73, %72 ], [ 0, %41 ]
  %46 = icmp slt i64 %45, 3
  br i1 %46, label %47, label %74

47:                                               ; preds = %70, %44
  %48 = phi i64 [ %71, %70 ], [ 0, %44 ]
  %49 = icmp slt i64 %48, 16
  br i1 %49, label %50, label %72

50:                                               ; preds = %53, %47
  %51 = phi i64 [ %69, %53 ], [ 0, %47 ]
  %52 = icmp slt i64 %51, 16
  br i1 %52, label %53, label %70

53:                                               ; preds = %50
  %54 = mul nsw i64 %45, 16
  %55 = add i64 %54, %48
  %56 = mul nsw i64 %42, 16
  %57 = add i64 %56, %51
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_cons_buff_1, i64 32) ]
  %58 = mul i64 %55, 128
  %59 = add i64 %58, %57
  %60 = getelementptr bfloat, ptr @fifo_0_cons_buff_1, i64 %59
  %61 = load bfloat, ptr %60, align 2
  %62 = mul nsw i64 %45, 256
  %63 = mul nsw i64 %48, 16
  %64 = add i64 %62, %63
  %65 = add i64 %64, %51
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_buff_1, i64 32) ]
  %66 = mul i64 %42, 768
  %67 = add i64 %66, %65
  %68 = getelementptr bfloat, ptr @fifo_2_buff_1, i64 %67
  store bfloat %61, ptr %68, align 2
  %69 = add i64 %51, 1
  br label %50

70:                                               ; preds = %50
  %71 = add i64 %48, 1
  br label %47

72:                                               ; preds = %47
  %73 = add i64 %45, 1
  br label %44

74:                                               ; preds = %44
  %75 = add i64 %42, 1
  br label %41

76:                                               ; preds = %41
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  %77 = add i64 %2, 2
  br label %1

78:                                               ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  br label %79

79:                                               ; preds = %112, %78
  %80 = phi i64 [ %113, %112 ], [ 0, %78 ]
  %81 = icmp slt i64 %80, 8
  br i1 %81, label %82, label %114

82:                                               ; preds = %110, %79
  %83 = phi i64 [ %111, %110 ], [ 0, %79 ]
  %84 = icmp slt i64 %83, 3
  br i1 %84, label %85, label %112

85:                                               ; preds = %108, %82
  %86 = phi i64 [ %109, %108 ], [ 0, %82 ]
  %87 = icmp slt i64 %86, 16
  br i1 %87, label %88, label %110

88:                                               ; preds = %91, %85
  %89 = phi i64 [ %107, %91 ], [ 0, %85 ]
  %90 = icmp slt i64 %89, 16
  br i1 %90, label %91, label %108

91:                                               ; preds = %88
  %92 = mul nsw i64 %83, 16
  %93 = add i64 %92, %86
  %94 = mul nsw i64 %80, 16
  %95 = add i64 %94, %89
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_cons_buff_0, i64 32) ]
  %96 = mul i64 %93, 128
  %97 = add i64 %96, %95
  %98 = getelementptr bfloat, ptr @fifo_0_cons_buff_0, i64 %97
  %99 = load bfloat, ptr %98, align 2
  %100 = mul nsw i64 %83, 256
  %101 = mul nsw i64 %86, 16
  %102 = add i64 %100, %101
  %103 = add i64 %102, %89
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_buff_0, i64 32) ]
  %104 = mul i64 %80, 768
  %105 = add i64 %104, %103
  %106 = getelementptr bfloat, ptr @fifo_2_buff_0, i64 %105
  store bfloat %99, ptr %106, align 2
  %107 = add i64 %89, 1
  br label %88

108:                                              ; preds = %88
  %109 = add i64 %86, 1
  br label %85

110:                                              ; preds = %85
  %111 = add i64 %83, 1
  br label %82

112:                                              ; preds = %82
  %113 = add i64 %80, 1
  br label %79

114:                                              ; preds = %79
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
