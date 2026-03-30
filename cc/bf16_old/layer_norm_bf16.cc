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
void layer_norm_single_batch_no_bias(T_in *input_tensor, T_in *weight,
                                     T_out *output_tensor) {
  constexpr int vec_factor = 16;
  using vec_t = aie::vector<T_in, vec_factor>;
  event0();
  for (int iter = 0; iter < SEQ_LEN; iter++) {
    T_in *__restrict input_ptr = input_tensor;
    T_in *__restrict weight_ptr = weight;
    T_out *__restrict output_ptr = output_tensor;
    float mean_f = 0.0f, variance_sum_f = 0.0f;
    const int F = HIDDEN / vec_factor;
    for (int i = 0; i < F; i++) {
      vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      aie::accum<accfloat, vec_factor> acc = aie::mul(input_vec, (bfloat16)1.0f);
      mean_f += aie::reduce_add(acc.to_vector<float>());
    }
    mean_f /= HIDDEN;
    input_ptr = input_tensor;
    bfloat16 mean = (bfloat16)mean_f;
    for (int i = 0; i < F; i++) {
      vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      vec_t diff = aie::sub(input_vec, mean);
      aie::accum<accfloat, vec_factor> diff_acc = aie::mul(diff, diff);
      variance_sum_f += aie::reduce_add(diff_acc.to_vector<float>());
    }
    float inv_std_f = variance_sum_f / HIDDEN + EPS;
    vec_t variance_vec =
        aie::broadcast<T_in, vec_factor>((bfloat16)inv_std_f);
    vec_t rms = aie::invsqrt(variance_vec);
    input_ptr = input_tensor;
    for (int i = 0; i < F; i++) {
      vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      vec_t normed = aie::mul(aie::sub(input_vec, mean), rms);
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

void layer_norm_bf16(bfloat16 A_in[4][768], bfloat16 B_in[768], bfloat16 C_out[4][768]) {
  layer_norm_single_batch_no_bias<bfloat16, bfloat16, 4, 768>(&A_in[0][0], B_in,
                                                              &C_out[0][0]);
}

} // extern "C"


// // bf16 layer norm: loads bf16, accumulates mean/var in float32, stores bf16.
// // Weight is also bf16.
// template <const int SEQ_LEN, const int HIDDEN>
// void layer_norm_bf16_impl(bfloat16 *input_tensor, bfloat16 *weight,
//                           bfloat16 *output_tensor) {
//   constexpr int vec_factor = 32; // 32 bf16 elements per vector
//   using vec_bf16 = aie::vector<bfloat16, vec_factor>;
//   constexpr int F = HIDDEN / vec_factor;

//   for (int iter = 0; iter < SEQ_LEN; iter++) {
//     bfloat16 *__restrict input_ptr = input_tensor;
//     bfloat16 *__restrict weight_ptr = weight;
//     bfloat16 *__restrict output_ptr = output_tensor;

//     // Pass 1: compute mean (accumulate in float32)
//     float mean = 0.0f;
//     for (int i = 0; i < F; i++) {
//       vec_bf16 input_vec = aie::load_v<vec_factor>(input_ptr);
//       input_ptr += vec_factor;
//       // reduce_add on bf16 vector, accumulate in float
//       mean += aie::reduce_add(input_vec);
//     }
//     mean /= HIDDEN;

//     // Pass 2: compute variance (accumulate in float32)
//     input_ptr = input_tensor;
//     float variance_sum = 0.0f;
//     for (int i = 0; i < F; i++) {
//       vec_bf16 input_vec = aie::load_v<vec_factor>(input_ptr);
//       input_ptr += vec_factor;
//       vec_bf16 diff = aie::sub(input_vec, (bfloat16)mean);
//       vec_bf16 square_vec = aie::mul(diff, diff);
//       variance_sum += aie::reduce_add(square_vec);
//     }

//     // Compute 1/sqrt(var + eps) using float, then broadcast as bf16
//     float inv_std_f = 1.0f / sqrtf(variance_sum / HIDDEN + EPS);
//     bfloat16 inv_std = (bfloat16)inv_std_f;
//     bfloat16 mean_bf16 = (bfloat16)mean;

//     // Pass 3: normalize and apply weight
//     input_ptr = input_tensor;
//     for (int i = 0; i < F; i++) {
//       vec_bf16 input_vec = aie::load_v<vec_factor>(input_ptr);
//       input_ptr += vec_factor;
//       vec_bf16 normed = aie::mul(aie::sub(input_vec, mean_bf16), inv_std);
//       vec_bf16 weight_vec = aie::load_v<vec_factor>(weight_ptr);
//       weight_ptr += vec_factor;
//       vec_bf16 result = aie::mul(normed, weight_vec);
//       aie::store_v(output_ptr, result);
//       output_ptr += vec_factor;
//     }
//     input_tensor += HIDDEN;
//     output_tensor += HIDDEN;
//   }
// }

// extern "C" {

// void layer_norm_bf16(bfloat16 A_in[4][768], bfloat16 B_in[768],
//                      bfloat16 C_out[4][768]) {
//   layer_norm_bf16_impl<4, 768>(&A_in[0][0], B_in, &C_out[0][0]);
// }

// } // extern "C"
