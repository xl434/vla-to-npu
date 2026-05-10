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

#define EPS 1e-5f // epsilon

template <typename T_in, typename T_out, const int SEQ_LEN, const int HIDDEN>
void rms_norm_single_batch_bf16(T_in *input_tensor, T_in *weight,
                                T_out *output_tensor) {
  constexpr int vec_factor = 16;
  using vec_t = aie::vector<T_in, vec_factor>;
  event0();
  for (int iter = 0; iter < SEQ_LEN; iter++) {
    T_in *__restrict input_ptr = input_tensor;
    T_in *__restrict weight_ptr = weight;
    T_out *__restrict output_ptr = output_tensor;
    float square_sum_f = 0.0f;
    const int F = HIDDEN / vec_factor;
    // Pass 1: compute sum of squares in float32 for precision
    for (int i = 0; i < F; i++) {
      vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      aie::accum<accfloat, vec_factor> sq_acc = aie::mul(input_vec, input_vec);
      square_sum_f += aie::reduce_add(sq_acc.to_vector<float>());
    }
    // Compute 1/sqrt(mean(x^2) + eps)
    float inv_rms_f = square_sum_f / HIDDEN + EPS;
    vec_t rms_vec =
        aie::broadcast<T_in, vec_factor>((bfloat16)inv_rms_f);
    vec_t rms = aie::invsqrt(rms_vec);
    // Pass 2: normalize and apply weight
    input_ptr = input_tensor;
    for (int i = 0; i < F; i++) {
      vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      vec_t normed = aie::mul(input_vec, rms);
      vec_t weight_vec = aie::load_v<vec_factor>(weight_ptr);
      weight_ptr += vec_factor;
      vec_t result = aie::mul(normed, weight_vec);
      aie::store_v(output_ptr, result);
      output_ptr += vec_factor;
    }
    input_tensor += HIDDEN;
    output_tensor += HIDDEN;
  }
  event1();
}

extern "C" {

void rms_norm_bf16(bfloat16 A_in[4][96], bfloat16 B_in[96],
                   bfloat16 C_out[4][96]) {
  rms_norm_single_batch_bf16<bfloat16, bfloat16, 4, 96>(
      &A_in[0][0], B_in, &C_out[0][0]);
}

} // extern "C"