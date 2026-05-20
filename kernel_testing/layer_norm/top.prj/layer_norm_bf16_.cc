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

// bf16 -> float
template <unsigned vec_factor>
static inline aie::vector<float, vec_factor> bf16_to_float(aie::vector<bfloat16, vec_factor>& x_bf16) {
  aie::vector<float, vec_factor> x;
  for (int i = 0; i < vec_factor; i++) {
      x[i] = (float)x_bf16[i];
  }
  return x;
}

// float -> bf16
template <unsigned vec_factor>
static inline aie::vector<bfloat16, vec_factor> float_to_bf16(aie::vector<float, vec_factor>& r) {
  aie::vector<bfloat16, vec_factor> outv;
    for (int i = 0; i < vec_factor; i++) {
      outv[i] = (bfloat16)r[i];
    }
  return outv;
}

template <const int SEQ_LEN, const int HIDDEN>
void layer_norm_single_batch_no_bias(bfloat16 *input_tensor, bfloat16 *weight,
                                     bfloat16 *output_tensor) {
  constexpr int vec_factor = 16;
  using bvec_t = aie::vector<bfloat16, vec_factor>;  // bf16 for load/store
  using fvec_t = aie::vector<float,    vec_factor>;  // float for all math

  event0();
  for (int iter = 0; iter < SEQ_LEN; iter++) {
    bfloat16 *__restrict input_ptr  = input_tensor;
    bfloat16 *__restrict weight_ptr = weight;
    bfloat16 *__restrict output_ptr = output_tensor;

    const int F = HIDDEN / vec_factor;

    float mean = 0.0f;
    for (int i = 0; i < F; i++) {
      bvec_t input_bf16 = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      fvec_t input_f = bf16_to_float(input_bf16);  // ← load bf16, compute in float
      mean += aie::reduce_add(input_f);
    }
    mean /= HIDDEN;
    fvec_t mean_vec = aie::broadcast<float, vec_factor>(mean);

    float variance_sum = 0.0f;
    input_ptr = input_tensor;
    for (int i = 0; i < F; i++) {
      bvec_t input_bf16 = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      fvec_t input_f = bf16_to_float(input_bf16);
      fvec_t diff       = aie::sub(input_f, mean_vec);
      fvec_t square_vec = aie::mul(diff, diff);
      variance_sum += aie::reduce_add(square_vec);
    }
    fvec_t variance_vec = aie::broadcast<float, vec_factor>(variance_sum / HIDDEN + EPS);
    fvec_t inv_std      = aie::invsqrt(variance_vec);

    input_ptr = input_tensor;
    for (int i = 0; i < F; i++) {
      bvec_t input_bf16  = aie::load_v<vec_factor>(input_ptr);
      input_ptr  += vec_factor;
      bvec_t weight_bf16 = aie::load_v<vec_factor>(weight_ptr);
      weight_ptr += vec_factor;

      fvec_t input_f  = bf16_to_float(input_bf16);
      fvec_t weight_f = bf16_to_float(weight_bf16);

      fvec_t normed = aie::mul(aie::sub(input_f, mean_vec), inv_std);
      fvec_t result = aie::mul(normed, weight_f);

      bvec_t result_bf16 = float_to_bf16(result);  // convert back before store
      aie::store_v(output_ptr, result_bf16);
      output_ptr += vec_factor;
    }

    input_tensor  += HIDDEN;
    output_tensor += HIDDEN;
  }
  event1();
}

extern "C" {

void layer_norm(bfloat16 A_in[4][768], bfloat16 B_in[768], bfloat16 C_out[4][768]) {
  layer_norm_single_batch_no_bias<4, 768>(&A_in[0][0], B_in, &C_out[0][0]);
}

} // extern "C"