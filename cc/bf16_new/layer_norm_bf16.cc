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
  using vec_t = aie::vector<float, vec_factor>;
  using fvec_t = aie::vector<bfloat16, vec_factor>;

  event0();
  for (int iter = 0; iter < SEQ_LEN; iter++) {
    bfloat16 *__restrict input_ptr = input_tensor;
    bfloat16 *__restrict weight_ptr = weight;
    bfloat16 *__restrict output_ptr = output_tensor;

    // bfloat16 -> float
    vec_t x_bf16 = aie::load_v<vec_factor>(input_ptr + inner_it);
    fvec_t x;
    for (int i = 0; i < vec_factor; i++) {
      x[i] = (float)x_bf16[i];
    }

    float mean = 0.0f, variance_sum = 0.0f;
    const int F = HIDDEN / vec_factor;
    for (int i = 0; i < F; i++) {
      fvec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      mean += aie::reduce_add(input_vec);
    }
    mean /= HIDDEN;
    fvec_t mean_vec = aie::broadcast<float, vec_factor>(mean);
    input_ptr = input_tensor;
    for (int i = 0; i < F; i++) {
      fvec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      fvec_t diff = aie::sub(input_vec, mean_vec);
      fvec_t square_vec = aie::mul(diff, diff);
      variance_sum += aie::reduce_add(square_vec);
    }
    fvec_t variance_vec =
        aie::broadcast<float, vec_factor>(variance_sum / HIDDEN + EPS);
    fvec_t rms = aie::invsqrt(variance_vec);
    input_ptr = input_tensor;
    for (int i = 0; i < F; i++) {
      fvec_t input_vec = aie::load_v<vec_factor>(input_ptr);
      input_ptr += vec_factor;
      fvec_t normed = aie::mul(aie::sub(input_vec, mean_vec), rms);
      fvec_t weight_vec = aie::load_v<vec_factor>(weight_ptr);
      weight_ptr += vec_factor;
      fvec_t result = aie::mul(normed, weight_vec);
      aie::store_v(output_ptr, result);
      output_ptr += vec_factor;
    }
    input_tensor += HIDDEN;
    output_tensor += HIDDEN;
  }
    // float -> bfloat16
    vec_t outv;
    for (int i = 0; i < vec_factor; i++) {
      outv[i] = (bfloat16)r[i];
    }
    aie::store_v(output_ptr + inner_it, outv);
    
  event1();
}

extern "C" {

void layer_norm(bfloat16 A_in[4][768], bfloat16 B_in[768], bfloat16 C_out[4][768]) {
  layer_norm_single_batch_no_bias<bfloat16, bfloat16, 4, 768>(&A_in[0][0], B_in, &C_out[0][0]);
}

} // extern "C"