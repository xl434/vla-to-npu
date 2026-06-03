/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define NOCPP

// Transpose matmul with scale for bfloat16
// Computes C[i][j] = scale * dot(A[i,:], B[j,:])
// A is [M, K], B is [N, K] (B rows are "transposed" K vectors)
// C is [M, N]
template <const int M, const int N, const int K, const float scale>
void transpose_matmul_with_scale_bf16_impl(bfloat16 *tensor_a,
                                           bfloat16 *tensor_b,
                                           bfloat16 *output_tensor) {
  constexpr int vec_factor = 32; // 32 bf16 per vector
  using vec_t = aie::vector<bfloat16, vec_factor>;

  bfloat16 *__restrict tensor_a_ptr = tensor_a;
  for (int outer_iter = 0; outer_iter < M; outer_iter++) {
    bfloat16 *__restrict tensor_b_ptr = tensor_b;
    for (int inner_iter = 0; inner_iter < N; ++inner_iter) {
      float sum = 0.0f;
      const int F = K / vec_factor;
      bfloat16 *__restrict tensor_a_tile_ptr = tensor_a_ptr;
      bfloat16 *__restrict tensor_b_tile_ptr = tensor_b_ptr;
      for (int i = 0; i < F; i++) {
        vec_t input_vec_a = aie::load_v<vec_factor>(tensor_a_tile_ptr);
        vec_t input_vec_b = aie::load_v<vec_factor>(tensor_b_tile_ptr);
        tensor_a_tile_ptr += vec_factor;
        tensor_b_tile_ptr += vec_factor;
        vec_t mul_vec = aie::mul(input_vec_a, input_vec_b);
        sum += aie::reduce_add(mul_vec);
      }
      output_tensor[outer_iter * N + inner_iter] =
          static_cast<bfloat16>(sum * scale);
      tensor_b_ptr += K;
    }
    tensor_a_ptr += K;
  }
}

extern "C" {

// M=32, N=32, K=64, scale=1/sqrt(64)=0.125
void transpose_matmul_with_scale_bf16(bfloat16 A_in[32][64],
                                      bfloat16 B_in[32][64],
                                      bfloat16 C_out[32][32]) {
  transpose_matmul_with_scale_bf16_impl<32, 32, 64, 0.125f>(
      &A_in[0][0], &B_in[0][0], &C_out[0][0]);
}

} // extern "C"
