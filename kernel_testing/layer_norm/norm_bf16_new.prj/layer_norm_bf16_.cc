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

template <const int SEQ_LEN, const int HIDDEN>
void layer_norm_single_batch_no_bias(bfloat16 *input_tensor, bfloat16 *weight,
                                     bfloat16 *output_tensor) {
  constexpr int vec_factor = 16;
  using bvec_t = aie::vector<bfloat16, vec_factor>;
  using fvec_t = aie::vector<float, vec_factor>;

  event0();
  for (int iter = 0; iter < SEQ_LEN; iter++) {
    bfloat16 *__restrict input_ptr = input_tensor;
    bfloat16 *__restrict weight_ptr = weight;
    bfloat16 *__restrict output_ptr = output_tensor;

    float mean = 0.0f, variance_sum = 0.0f;
    const int F = HIDDEN / vec_factor;

    // Pass 1: compute mean (accumulate in float)
    for (int i = 0; i < F; i++) {
      bvec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      // bf16 -> float for accumulation
      fvec_t fv;
      for (int j = 0; j < vec_factor; j++) {
        fv[j] = (float)input_vec[j];
      }
      mean += aie::reduce_add(fv);
    }
    mean /= HIDDEN;
    fvec_t mean_vec = aie::broadcast<float, vec_factor>(mean);

    // Pass 2: compute variance (in float)
    input_ptr = input_tensor;
    for (int i = 0; i < F; i++) {
      bvec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      fvec_t fv;
      for (int j = 0; j < vec_factor; j++) {
        fv[j] = (float)input_vec[j];
      }
      fvec_t diff = aie::sub(fv, mean_vec);
      fvec_t square_vec = aie::mul(diff, diff);
      variance_sum += aie::reduce_add(square_vec);
    }
    fvec_t variance_vec =
        aie::broadcast<float, vec_factor>(variance_sum / HIDDEN + EPS);
    fvec_t rms = aie::invsqrt(variance_vec);

    // Pass 3: normalize and scale (compute in float, store as bf16)
    input_ptr = input_tensor;
    for (int i = 0; i < F; i++) {
      bvec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      // bf16 input -> float
      fvec_t fv;
      for (int j = 0; j < vec_factor; j++) {
        fv[j] = (float)input_vec[j];
      }
      fvec_t normed = aie::mul(aie::sub(fv, mean_vec), rms);
      // bf16 weight -> float
      bvec_t weight_bv = aie::load_v<vec_factor>(weight_ptr);
      weight_ptr += vec_factor;
      fvec_t weight_fv;
      for (int j = 0; j < vec_factor; j++) {
        weight_fv[j] = (float)weight_bv[j];
      }
      fvec_t result_f = aie::mul(normed, weight_fv);
      // float -> bf16
      bvec_t result;
      for (int j = 0; j < vec_factor; j++) {
        result[j] = (bfloat16)result_f[j];
      }
      aie::store_v(output_ptr, result);
      output_ptr += vec_factor;
    }
    input_tensor += HIDDEN;
    output_tensor += HIDDEN;
  }
  event1();
}

extern "C" {

void layer_norm(bfloat16 A_in[4][768], bfloat16 B_in[768], bfloat16 C_out[4][768]) {
  layer_norm_single_batch_no_bias<4, 768>(&A_in[0][0], B_in, &C_out[0][0]);
}

} // extern "C"
