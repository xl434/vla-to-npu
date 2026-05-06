; ModuleID = '/home/dl2239/vla-to-npu/vla/preprocessing/im2col.prj/top.mlir.prj/input.llpeanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2"

@fifo_0_cons_buff_1 = external global [48 x [128 x bfloat]]
@fifo_0_cons_buff_0 = external global [48 x [128 x bfloat]]
@fifo_2_buff_1 = external global [8 x [768 x bfloat]]
@fifo_2_buff_0 = external global [8 x [768 x bfloat]]

; Function Attrs: nounwind
declare void @llvm.aie2.acquire(i32, i32) #0

; Function Attrs: nounwind
declare void @llvm.aie2.release(i32, i32) #0

; Function Attrs: nounwind
define void @core_0_2() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %0, %204
  %2 = phi i64 [ 0, %0 ], [ %205, %204 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_buff_0, i64 32) ]
  br label %.preheader11

.preheader11:                                     ; preds = %1, %100
  %3 = phi i64 [ 0, %1 ], [ %101, %100 ]
  %4 = mul nuw nsw i64 %3, 1536
  %5 = shl nuw nsw i64 %3, 5
  br label %.preheader7

.preheader7:                                      ; preds = %.preheader7, %.preheader11
  %6 = phi i64 [ 0, %.preheader11 ], [ %34, %.preheader7 ]
  %7 = shl nuw nsw i64 %6, 5
  %8 = add nuw nsw i64 %4, %7
  %9 = trunc i64 %8 to i20
  %scevgep = getelementptr i8, ptr @fifo_2_buff_0, i20 %9
  %10 = shl nuw nsw i64 %6, 8
  %11 = add nuw nsw i64 %5, %10
  %12 = trunc i64 %11 to i20
  %scevgep12 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %12
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep, ptr noundef nonnull align 32 dereferenceable(32) %scevgep12, i20 32, i1 false)
  %13 = or disjoint i64 %6, 1
  %14 = shl nuw nsw i64 %13, 5
  %15 = add nuw nsw i64 %4, %14
  %16 = trunc i64 %15 to i20
  %scevgep.117 = getelementptr i8, ptr @fifo_2_buff_0, i20 %16
  %17 = shl nuw nsw i64 %13, 8
  %18 = add nuw nsw i64 %5, %17
  %19 = trunc i64 %18 to i20
  %scevgep12.118 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %19
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep.117, ptr noundef nonnull align 32 dereferenceable(32) %scevgep12.118, i20 32, i1 false)
  %20 = or disjoint i64 %6, 2
  %21 = shl nuw nsw i64 %20, 5
  %22 = add nuw nsw i64 %4, %21
  %23 = trunc i64 %22 to i20
  %scevgep.220 = getelementptr i8, ptr @fifo_2_buff_0, i20 %23
  %24 = shl nuw nsw i64 %20, 8
  %25 = add nuw nsw i64 %5, %24
  %26 = trunc i64 %25 to i20
  %scevgep12.221 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %26
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep.220, ptr noundef nonnull align 32 dereferenceable(32) %scevgep12.221, i20 32, i1 false)
  %27 = or disjoint i64 %6, 3
  %28 = shl nuw nsw i64 %27, 5
  %29 = add nuw nsw i64 %4, %28
  %30 = trunc i64 %29 to i20
  %scevgep.3 = getelementptr i8, ptr @fifo_2_buff_0, i20 %30
  %31 = shl nuw nsw i64 %27, 8
  %32 = add nuw nsw i64 %5, %31
  %33 = trunc i64 %32 to i20
  %scevgep12.3 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %33
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep.3, ptr noundef nonnull align 32 dereferenceable(32) %scevgep12.3, i20 32, i1 false)
  %34 = add nuw nsw i64 %6, 4
  %35 = icmp ult i64 %27, 15
  br i1 %35, label %.preheader7, label %.preheader9.1

.preheader9.1:                                    ; preds = %.preheader7
  %36 = add nuw nsw i64 %4, 512
  %37 = add nuw nsw i64 %5, 4096
  br label %.preheader7.1

.preheader7.1:                                    ; preds = %.preheader7.1, %.preheader9.1
  %38 = phi i64 [ 0, %.preheader9.1 ], [ %66, %.preheader7.1 ]
  %39 = shl nuw nsw i64 %38, 5
  %40 = add nuw nsw i64 %36, %39
  %41 = trunc i64 %40 to i20
  %scevgep.1 = getelementptr i8, ptr @fifo_2_buff_0, i20 %41
  %42 = shl nuw nsw i64 %38, 8
  %43 = add nuw nsw i64 %37, %42
  %44 = trunc i64 %43 to i20
  %scevgep12.1 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %44
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep.1, ptr noundef nonnull align 32 dereferenceable(32) %scevgep12.1, i20 32, i1 false)
  %45 = or disjoint i64 %38, 1
  %46 = shl nuw nsw i64 %45, 5
  %47 = add nuw nsw i64 %36, %46
  %48 = trunc i64 %47 to i20
  %scevgep.1.1 = getelementptr i8, ptr @fifo_2_buff_0, i20 %48
  %49 = shl nuw nsw i64 %45, 8
  %50 = add nuw nsw i64 %37, %49
  %51 = trunc i64 %50 to i20
  %scevgep12.1.1 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %51
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep.1.1, ptr noundef nonnull align 32 dereferenceable(32) %scevgep12.1.1, i20 32, i1 false)
  %52 = or disjoint i64 %38, 2
  %53 = shl nuw nsw i64 %52, 5
  %54 = add nuw nsw i64 %36, %53
  %55 = trunc i64 %54 to i20
  %scevgep.1.2 = getelementptr i8, ptr @fifo_2_buff_0, i20 %55
  %56 = shl nuw nsw i64 %52, 8
  %57 = add nuw nsw i64 %37, %56
  %58 = trunc i64 %57 to i20
  %scevgep12.1.2 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %58
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep.1.2, ptr noundef nonnull align 32 dereferenceable(32) %scevgep12.1.2, i20 32, i1 false)
  %59 = or disjoint i64 %38, 3
  %60 = shl nuw nsw i64 %59, 5
  %61 = add nuw nsw i64 %36, %60
  %62 = trunc i64 %61 to i20
  %scevgep.1.3 = getelementptr i8, ptr @fifo_2_buff_0, i20 %62
  %63 = shl nuw nsw i64 %59, 8
  %64 = add nuw nsw i64 %37, %63
  %65 = trunc i64 %64 to i20
  %scevgep12.1.3 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %65
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep.1.3, ptr noundef nonnull align 32 dereferenceable(32) %scevgep12.1.3, i20 32, i1 false)
  %66 = add nuw nsw i64 %38, 4
  %67 = icmp ult i64 %59, 15
  br i1 %67, label %.preheader7.1, label %.preheader9.2

.preheader9.2:                                    ; preds = %.preheader7.1
  %68 = add nuw nsw i64 %4, 1024
  %69 = add nuw nsw i64 %5, 8192
  br label %.preheader7.2

.preheader7.2:                                    ; preds = %.preheader7.2, %.preheader9.2
  %70 = phi i64 [ 0, %.preheader9.2 ], [ %98, %.preheader7.2 ]
  %71 = shl nuw nsw i64 %70, 5
  %72 = add nuw nsw i64 %68, %71
  %73 = trunc i64 %72 to i20
  %scevgep.2 = getelementptr i8, ptr @fifo_2_buff_0, i20 %73
  %74 = shl nuw nsw i64 %70, 8
  %75 = add nuw nsw i64 %69, %74
  %76 = trunc i64 %75 to i20
  %scevgep12.2 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %76
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep.2, ptr noundef nonnull align 32 dereferenceable(32) %scevgep12.2, i20 32, i1 false)
  %77 = or disjoint i64 %70, 1
  %78 = shl nuw nsw i64 %77, 5
  %79 = add nuw nsw i64 %68, %78
  %80 = trunc i64 %79 to i20
  %scevgep.2.1 = getelementptr i8, ptr @fifo_2_buff_0, i20 %80
  %81 = shl nuw nsw i64 %77, 8
  %82 = add nuw nsw i64 %69, %81
  %83 = trunc i64 %82 to i20
  %scevgep12.2.1 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %83
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep.2.1, ptr noundef nonnull align 32 dereferenceable(32) %scevgep12.2.1, i20 32, i1 false)
  %84 = or disjoint i64 %70, 2
  %85 = shl nuw nsw i64 %84, 5
  %86 = add nuw nsw i64 %68, %85
  %87 = trunc i64 %86 to i20
  %scevgep.2.2 = getelementptr i8, ptr @fifo_2_buff_0, i20 %87
  %88 = shl nuw nsw i64 %84, 8
  %89 = add nuw nsw i64 %69, %88
  %90 = trunc i64 %89 to i20
  %scevgep12.2.2 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %90
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep.2.2, ptr noundef nonnull align 32 dereferenceable(32) %scevgep12.2.2, i20 32, i1 false)
  %91 = or disjoint i64 %70, 3
  %92 = shl nuw nsw i64 %91, 5
  %93 = add nuw nsw i64 %68, %92
  %94 = trunc i64 %93 to i20
  %scevgep.2.3 = getelementptr i8, ptr @fifo_2_buff_0, i20 %94
  %95 = shl nuw nsw i64 %91, 8
  %96 = add nuw nsw i64 %69, %95
  %97 = trunc i64 %96 to i20
  %scevgep12.2.3 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %97
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep.2.3, ptr noundef nonnull align 32 dereferenceable(32) %scevgep12.2.3, i20 32, i1 false)
  %98 = add nuw nsw i64 %70, 4
  %99 = icmp ult i64 %91, 15
  br i1 %99, label %.preheader7.2, label %100

100:                                              ; preds = %.preheader7.2
  %101 = add nuw nsw i64 %3, 1
  %102 = icmp ult i64 %3, 7
  br i1 %102, label %.preheader11, label %103

103:                                              ; preds = %100
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_buff_1, i64 32) ]
  br label %.preheader10

.preheader10:                                     ; preds = %103, %201
  %104 = phi i64 [ 0, %103 ], [ %202, %201 ]
  %105 = mul nuw nsw i64 %104, 1536
  %106 = shl nuw nsw i64 %104, 5
  br label %.preheader6

.preheader6:                                      ; preds = %.preheader6, %.preheader10
  %107 = phi i64 [ 0, %.preheader10 ], [ %135, %.preheader6 ]
  %108 = shl nuw nsw i64 %107, 5
  %109 = add nuw nsw i64 %105, %108
  %110 = trunc i64 %109 to i20
  %scevgep13 = getelementptr i8, ptr @fifo_2_buff_1, i20 %110
  %111 = shl nuw nsw i64 %107, 8
  %112 = add nuw nsw i64 %106, %111
  %113 = trunc i64 %112 to i20
  %scevgep14 = getelementptr i8, ptr @fifo_0_cons_buff_1, i20 %113
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep13, ptr noundef nonnull align 32 dereferenceable(32) %scevgep14, i20 32, i1 false)
  %114 = or disjoint i64 %107, 1
  %115 = shl nuw nsw i64 %114, 5
  %116 = add nuw nsw i64 %105, %115
  %117 = trunc i64 %116 to i20
  %scevgep13.123 = getelementptr i8, ptr @fifo_2_buff_1, i20 %117
  %118 = shl nuw nsw i64 %114, 8
  %119 = add nuw nsw i64 %106, %118
  %120 = trunc i64 %119 to i20
  %scevgep14.124 = getelementptr i8, ptr @fifo_0_cons_buff_1, i20 %120
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep13.123, ptr noundef nonnull align 32 dereferenceable(32) %scevgep14.124, i20 32, i1 false)
  %121 = or disjoint i64 %107, 2
  %122 = shl nuw nsw i64 %121, 5
  %123 = add nuw nsw i64 %105, %122
  %124 = trunc i64 %123 to i20
  %scevgep13.226 = getelementptr i8, ptr @fifo_2_buff_1, i20 %124
  %125 = shl nuw nsw i64 %121, 8
  %126 = add nuw nsw i64 %106, %125
  %127 = trunc i64 %126 to i20
  %scevgep14.227 = getelementptr i8, ptr @fifo_0_cons_buff_1, i20 %127
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep13.226, ptr noundef nonnull align 32 dereferenceable(32) %scevgep14.227, i20 32, i1 false)
  %128 = or disjoint i64 %107, 3
  %129 = shl nuw nsw i64 %128, 5
  %130 = add nuw nsw i64 %105, %129
  %131 = trunc i64 %130 to i20
  %scevgep13.3 = getelementptr i8, ptr @fifo_2_buff_1, i20 %131
  %132 = shl nuw nsw i64 %128, 8
  %133 = add nuw nsw i64 %106, %132
  %134 = trunc i64 %133 to i20
  %scevgep14.3 = getelementptr i8, ptr @fifo_0_cons_buff_1, i20 %134
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep13.3, ptr noundef nonnull align 32 dereferenceable(32) %scevgep14.3, i20 32, i1 false)
  %135 = add nuw nsw i64 %107, 4
  %136 = icmp ult i64 %128, 15
  br i1 %136, label %.preheader6, label %.preheader8.1

.preheader8.1:                                    ; preds = %.preheader6
  %137 = add nuw nsw i64 %105, 512
  %138 = add nuw nsw i64 %106, 4096
  br label %.preheader6.1

.preheader6.1:                                    ; preds = %.preheader6.1, %.preheader8.1
  %139 = phi i64 [ 0, %.preheader8.1 ], [ %167, %.preheader6.1 ]
  %140 = shl nuw nsw i64 %139, 5
  %141 = add nuw nsw i64 %137, %140
  %142 = trunc i64 %141 to i20
  %scevgep13.1 = getelementptr i8, ptr @fifo_2_buff_1, i20 %142
  %143 = shl nuw nsw i64 %139, 8
  %144 = add nuw nsw i64 %138, %143
  %145 = trunc i64 %144 to i20
  %scevgep14.1 = getelementptr i8, ptr @fifo_0_cons_buff_1, i20 %145
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep13.1, ptr noundef nonnull align 32 dereferenceable(32) %scevgep14.1, i20 32, i1 false)
  %146 = or disjoint i64 %139, 1
  %147 = shl nuw nsw i64 %146, 5
  %148 = add nuw nsw i64 %137, %147
  %149 = trunc i64 %148 to i20
  %scevgep13.1.1 = getelementptr i8, ptr @fifo_2_buff_1, i20 %149
  %150 = shl nuw nsw i64 %146, 8
  %151 = add nuw nsw i64 %138, %150
  %152 = trunc i64 %151 to i20
  %scevgep14.1.1 = getelementptr i8, ptr @fifo_0_cons_buff_1, i20 %152
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep13.1.1, ptr noundef nonnull align 32 dereferenceable(32) %scevgep14.1.1, i20 32, i1 false)
  %153 = or disjoint i64 %139, 2
  %154 = shl nuw nsw i64 %153, 5
  %155 = add nuw nsw i64 %137, %154
  %156 = trunc i64 %155 to i20
  %scevgep13.1.2 = getelementptr i8, ptr @fifo_2_buff_1, i20 %156
  %157 = shl nuw nsw i64 %153, 8
  %158 = add nuw nsw i64 %138, %157
  %159 = trunc i64 %158 to i20
  %scevgep14.1.2 = getelementptr i8, ptr @fifo_0_cons_buff_1, i20 %159
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep13.1.2, ptr noundef nonnull align 32 dereferenceable(32) %scevgep14.1.2, i20 32, i1 false)
  %160 = or disjoint i64 %139, 3
  %161 = shl nuw nsw i64 %160, 5
  %162 = add nuw nsw i64 %137, %161
  %163 = trunc i64 %162 to i20
  %scevgep13.1.3 = getelementptr i8, ptr @fifo_2_buff_1, i20 %163
  %164 = shl nuw nsw i64 %160, 8
  %165 = add nuw nsw i64 %138, %164
  %166 = trunc i64 %165 to i20
  %scevgep14.1.3 = getelementptr i8, ptr @fifo_0_cons_buff_1, i20 %166
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep13.1.3, ptr noundef nonnull align 32 dereferenceable(32) %scevgep14.1.3, i20 32, i1 false)
  %167 = add nuw nsw i64 %139, 4
  %168 = icmp ult i64 %160, 15
  br i1 %168, label %.preheader6.1, label %.preheader8.2

.preheader8.2:                                    ; preds = %.preheader6.1
  %169 = add nuw nsw i64 %105, 1024
  %170 = add nuw nsw i64 %106, 8192
  br label %.preheader6.2

.preheader6.2:                                    ; preds = %.preheader6.2, %.preheader8.2
  %171 = phi i64 [ 0, %.preheader8.2 ], [ %199, %.preheader6.2 ]
  %172 = shl nuw nsw i64 %171, 5
  %173 = add nuw nsw i64 %169, %172
  %174 = trunc i64 %173 to i20
  %scevgep13.2 = getelementptr i8, ptr @fifo_2_buff_1, i20 %174
  %175 = shl nuw nsw i64 %171, 8
  %176 = add nuw nsw i64 %170, %175
  %177 = trunc i64 %176 to i20
  %scevgep14.2 = getelementptr i8, ptr @fifo_0_cons_buff_1, i20 %177
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep13.2, ptr noundef nonnull align 32 dereferenceable(32) %scevgep14.2, i20 32, i1 false)
  %178 = or disjoint i64 %171, 1
  %179 = shl nuw nsw i64 %178, 5
  %180 = add nuw nsw i64 %169, %179
  %181 = trunc i64 %180 to i20
  %scevgep13.2.1 = getelementptr i8, ptr @fifo_2_buff_1, i20 %181
  %182 = shl nuw nsw i64 %178, 8
  %183 = add nuw nsw i64 %170, %182
  %184 = trunc i64 %183 to i20
  %scevgep14.2.1 = getelementptr i8, ptr @fifo_0_cons_buff_1, i20 %184
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep13.2.1, ptr noundef nonnull align 32 dereferenceable(32) %scevgep14.2.1, i20 32, i1 false)
  %185 = or disjoint i64 %171, 2
  %186 = shl nuw nsw i64 %185, 5
  %187 = add nuw nsw i64 %169, %186
  %188 = trunc i64 %187 to i20
  %scevgep13.2.2 = getelementptr i8, ptr @fifo_2_buff_1, i20 %188
  %189 = shl nuw nsw i64 %185, 8
  %190 = add nuw nsw i64 %170, %189
  %191 = trunc i64 %190 to i20
  %scevgep14.2.2 = getelementptr i8, ptr @fifo_0_cons_buff_1, i20 %191
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep13.2.2, ptr noundef nonnull align 32 dereferenceable(32) %scevgep14.2.2, i20 32, i1 false)
  %192 = or disjoint i64 %171, 3
  %193 = shl nuw nsw i64 %192, 5
  %194 = add nuw nsw i64 %169, %193
  %195 = trunc i64 %194 to i20
  %scevgep13.2.3 = getelementptr i8, ptr @fifo_2_buff_1, i20 %195
  %196 = shl nuw nsw i64 %192, 8
  %197 = add nuw nsw i64 %170, %196
  %198 = trunc i64 %197 to i20
  %scevgep14.2.3 = getelementptr i8, ptr @fifo_0_cons_buff_1, i20 %198
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep13.2.3, ptr noundef nonnull align 32 dereferenceable(32) %scevgep14.2.3, i20 32, i1 false)
  %199 = add nuw nsw i64 %171, 4
  %200 = icmp ult i64 %192, 15
  br i1 %200, label %.preheader6.2, label %201

201:                                              ; preds = %.preheader6.2
  %202 = add nuw nsw i64 %104, 1
  %203 = icmp ult i64 %104, 7
  br i1 %203, label %.preheader10, label %204

204:                                              ; preds = %201
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  %205 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %205, 9223372036854775806
  br i1 %.not, label %206, label %1

206:                                              ; preds = %204
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_0_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @fifo_2_buff_0, i64 32) ]
  br label %.preheader5

.preheader5:                                      ; preds = %206, %304
  %207 = phi i64 [ 0, %206 ], [ %305, %304 ]
  %208 = mul nuw nsw i64 %207, 1536
  %209 = shl nuw nsw i64 %207, 5
  br label %.preheader

.preheader:                                       ; preds = %.preheader, %.preheader5
  %210 = phi i64 [ 0, %.preheader5 ], [ %238, %.preheader ]
  %211 = shl nuw nsw i64 %210, 5
  %212 = add nuw nsw i64 %208, %211
  %213 = trunc i64 %212 to i20
  %scevgep15 = getelementptr i8, ptr @fifo_2_buff_0, i20 %213
  %214 = shl nuw nsw i64 %210, 8
  %215 = add nuw nsw i64 %209, %214
  %216 = trunc i64 %215 to i20
  %scevgep16 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %216
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep15, ptr noundef nonnull align 32 dereferenceable(32) %scevgep16, i20 32, i1 false)
  %217 = or disjoint i64 %210, 1
  %218 = shl nuw nsw i64 %217, 5
  %219 = add nuw nsw i64 %208, %218
  %220 = trunc i64 %219 to i20
  %scevgep15.129 = getelementptr i8, ptr @fifo_2_buff_0, i20 %220
  %221 = shl nuw nsw i64 %217, 8
  %222 = add nuw nsw i64 %209, %221
  %223 = trunc i64 %222 to i20
  %scevgep16.130 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %223
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep15.129, ptr noundef nonnull align 32 dereferenceable(32) %scevgep16.130, i20 32, i1 false)
  %224 = or disjoint i64 %210, 2
  %225 = shl nuw nsw i64 %224, 5
  %226 = add nuw nsw i64 %208, %225
  %227 = trunc i64 %226 to i20
  %scevgep15.232 = getelementptr i8, ptr @fifo_2_buff_0, i20 %227
  %228 = shl nuw nsw i64 %224, 8
  %229 = add nuw nsw i64 %209, %228
  %230 = trunc i64 %229 to i20
  %scevgep16.233 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %230
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep15.232, ptr noundef nonnull align 32 dereferenceable(32) %scevgep16.233, i20 32, i1 false)
  %231 = or disjoint i64 %210, 3
  %232 = shl nuw nsw i64 %231, 5
  %233 = add nuw nsw i64 %208, %232
  %234 = trunc i64 %233 to i20
  %scevgep15.3 = getelementptr i8, ptr @fifo_2_buff_0, i20 %234
  %235 = shl nuw nsw i64 %231, 8
  %236 = add nuw nsw i64 %209, %235
  %237 = trunc i64 %236 to i20
  %scevgep16.3 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %237
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep15.3, ptr noundef nonnull align 32 dereferenceable(32) %scevgep16.3, i20 32, i1 false)
  %238 = add nuw nsw i64 %210, 4
  %239 = icmp ult i64 %231, 15
  br i1 %239, label %.preheader, label %.preheader4.1

.preheader4.1:                                    ; preds = %.preheader
  %240 = add nuw nsw i64 %208, 512
  %241 = add nuw nsw i64 %209, 4096
  br label %.preheader.1

.preheader.1:                                     ; preds = %.preheader.1, %.preheader4.1
  %242 = phi i64 [ 0, %.preheader4.1 ], [ %270, %.preheader.1 ]
  %243 = shl nuw nsw i64 %242, 5
  %244 = add nuw nsw i64 %240, %243
  %245 = trunc i64 %244 to i20
  %scevgep15.1 = getelementptr i8, ptr @fifo_2_buff_0, i20 %245
  %246 = shl nuw nsw i64 %242, 8
  %247 = add nuw nsw i64 %241, %246
  %248 = trunc i64 %247 to i20
  %scevgep16.1 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %248
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep15.1, ptr noundef nonnull align 32 dereferenceable(32) %scevgep16.1, i20 32, i1 false)
  %249 = or disjoint i64 %242, 1
  %250 = shl nuw nsw i64 %249, 5
  %251 = add nuw nsw i64 %240, %250
  %252 = trunc i64 %251 to i20
  %scevgep15.1.1 = getelementptr i8, ptr @fifo_2_buff_0, i20 %252
  %253 = shl nuw nsw i64 %249, 8
  %254 = add nuw nsw i64 %241, %253
  %255 = trunc i64 %254 to i20
  %scevgep16.1.1 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %255
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep15.1.1, ptr noundef nonnull align 32 dereferenceable(32) %scevgep16.1.1, i20 32, i1 false)
  %256 = or disjoint i64 %242, 2
  %257 = shl nuw nsw i64 %256, 5
  %258 = add nuw nsw i64 %240, %257
  %259 = trunc i64 %258 to i20
  %scevgep15.1.2 = getelementptr i8, ptr @fifo_2_buff_0, i20 %259
  %260 = shl nuw nsw i64 %256, 8
  %261 = add nuw nsw i64 %241, %260
  %262 = trunc i64 %261 to i20
  %scevgep16.1.2 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %262
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep15.1.2, ptr noundef nonnull align 32 dereferenceable(32) %scevgep16.1.2, i20 32, i1 false)
  %263 = or disjoint i64 %242, 3
  %264 = shl nuw nsw i64 %263, 5
  %265 = add nuw nsw i64 %240, %264
  %266 = trunc i64 %265 to i20
  %scevgep15.1.3 = getelementptr i8, ptr @fifo_2_buff_0, i20 %266
  %267 = shl nuw nsw i64 %263, 8
  %268 = add nuw nsw i64 %241, %267
  %269 = trunc i64 %268 to i20
  %scevgep16.1.3 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %269
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep15.1.3, ptr noundef nonnull align 32 dereferenceable(32) %scevgep16.1.3, i20 32, i1 false)
  %270 = add nuw nsw i64 %242, 4
  %271 = icmp ult i64 %263, 15
  br i1 %271, label %.preheader.1, label %.preheader4.2

.preheader4.2:                                    ; preds = %.preheader.1
  %272 = add nuw nsw i64 %208, 1024
  %273 = add nuw nsw i64 %209, 8192
  br label %.preheader.2

.preheader.2:                                     ; preds = %.preheader.2, %.preheader4.2
  %274 = phi i64 [ 0, %.preheader4.2 ], [ %302, %.preheader.2 ]
  %275 = shl nuw nsw i64 %274, 5
  %276 = add nuw nsw i64 %272, %275
  %277 = trunc i64 %276 to i20
  %scevgep15.2 = getelementptr i8, ptr @fifo_2_buff_0, i20 %277
  %278 = shl nuw nsw i64 %274, 8
  %279 = add nuw nsw i64 %273, %278
  %280 = trunc i64 %279 to i20
  %scevgep16.2 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %280
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep15.2, ptr noundef nonnull align 32 dereferenceable(32) %scevgep16.2, i20 32, i1 false)
  %281 = or disjoint i64 %274, 1
  %282 = shl nuw nsw i64 %281, 5
  %283 = add nuw nsw i64 %272, %282
  %284 = trunc i64 %283 to i20
  %scevgep15.2.1 = getelementptr i8, ptr @fifo_2_buff_0, i20 %284
  %285 = shl nuw nsw i64 %281, 8
  %286 = add nuw nsw i64 %273, %285
  %287 = trunc i64 %286 to i20
  %scevgep16.2.1 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %287
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep15.2.1, ptr noundef nonnull align 32 dereferenceable(32) %scevgep16.2.1, i20 32, i1 false)
  %288 = or disjoint i64 %274, 2
  %289 = shl nuw nsw i64 %288, 5
  %290 = add nuw nsw i64 %272, %289
  %291 = trunc i64 %290 to i20
  %scevgep15.2.2 = getelementptr i8, ptr @fifo_2_buff_0, i20 %291
  %292 = shl nuw nsw i64 %288, 8
  %293 = add nuw nsw i64 %273, %292
  %294 = trunc i64 %293 to i20
  %scevgep16.2.2 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %294
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep15.2.2, ptr noundef nonnull align 32 dereferenceable(32) %scevgep16.2.2, i20 32, i1 false)
  %295 = or disjoint i64 %274, 3
  %296 = shl nuw nsw i64 %295, 5
  %297 = add nuw nsw i64 %272, %296
  %298 = trunc i64 %297 to i20
  %scevgep15.2.3 = getelementptr i8, ptr @fifo_2_buff_0, i20 %298
  %299 = shl nuw nsw i64 %295, 8
  %300 = add nuw nsw i64 %273, %299
  %301 = trunc i64 %300 to i20
  %scevgep16.2.3 = getelementptr i8, ptr @fifo_0_cons_buff_0, i20 %301
  tail call void @llvm.memcpy.p0.p0.i20(ptr noundef nonnull align 32 dereferenceable(32) %scevgep15.2.3, ptr noundef nonnull align 32 dereferenceable(32) %scevgep16.2.3, i20 32, i1 false)
  %302 = add nuw nsw i64 %274, 4
  %303 = icmp ult i64 %295, 15
  br i1 %303, label %.preheader.2, label %304

304:                                              ; preds = %.preheader.2
  %305 = add nuw nsw i64 %207, 1
  %306 = icmp ult i64 %207, 7
  br i1 %306, label %.preheader5, label %307

307:                                              ; preds = %304
  tail call void @llvm.aie2.release(i32 48, i32 1)
  tail call void @llvm.aie2.release(i32 51, i32 1)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i20(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i20, i1 immarg) #2

attributes #0 = { nounwind }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
