/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * SiLU bf16 for per-core tile [4][256]
 * (SmolVLA action expert: FFN_HID=2048, tiled 2048/8=256 per core)
 *
 * Uses float32 internally for the Taylor series to avoid bf16 overflow
 * and precision loss. Input/output remain bfloat16.
 * Horner's method keeps intermediates bounded.
 */

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define NOCPP

void silu_bfloat16_256(bfloat16 input_x[4][256], bfloat16 output_x[4][256]) {
  event0();
  constexpr int SEQ_TILE         = 4;
  constexpr int FEATURE_DIM_TILE = 256;
  constexpr int vec_factor       = 16;  // float32: 16 per 512-bit register

  using fvec_t = aie::vector<float, vec_factor>;
  using bvec_t = aie::vector<bfloat16, vec_factor>;

  const fvec_t one   = aie::broadcast<float, vec_factor>(1.0f);
  const fvec_t fzero = aie::broadcast<float, vec_factor>(0.0f);
  const fvec_t fpos4 = aie::broadcast<float, vec_factor>(4.0f);
  const fvec_t fpos7 = aie::broadcast<float, vec_factor>(7.0f);
  const fvec_t fneg7 = aie::broadcast<float, vec_factor>(-7.0f);

  // Horner coefficients c_n = (-1)^n / n!
  const fvec_t h16 = aie::broadcast<float, vec_factor>( 1.0f / 20922789888000.0f);
  const fvec_t h15 = aie::broadcast<float, vec_factor>(-1.0f / 1307674368000.0f);
  const fvec_t h14 = aie::broadcast<float, vec_factor>( 1.0f / 87178291200.0f);
  const fvec_t h13 = aie::broadcast<float, vec_factor>(-1.0f / 6227020800.0f);
  const fvec_t h12 = aie::broadcast<float, vec_factor>( 1.0f / 479001600.0f);
  const fvec_t h11 = aie::broadcast<float, vec_factor>(-1.0f / 39916800.0f);
  const fvec_t h10 = aie::broadcast<float, vec_factor>( 1.0f / 3628800.0f);
  const fvec_t h9  = aie::broadcast<float, vec_factor>(-1.0f / 362880.0f);
  const fvec_t h8  = aie::broadcast<float, vec_factor>( 1.0f / 40320.0f);
  const fvec_t h7  = aie::broadcast<float, vec_factor>(-1.0f / 5040.0f);
  const fvec_t h6  = aie::broadcast<float, vec_factor>( 1.0f / 720.0f);
  const fvec_t h5  = aie::broadcast<float, vec_factor>(-1.0f / 120.0f);
  const fvec_t h4  = aie::broadcast<float, vec_factor>( 1.0f / 24.0f);
  const fvec_t h3  = aie::broadcast<float, vec_factor>(-1.0f / 6.0f);
  const fvec_t h2  = aie::broadcast<float, vec_factor>( 1.0f / 2.0f);
  const fvec_t h1  = aie::broadcast<float, vec_factor>(-1.0f);

  for (int s = 0; s < SEQ_TILE; ++s) {
    bfloat16 *__restrict input_ptr  = &input_x[s][0];
    bfloat16 *__restrict output_ptr = &output_x[s][0];

    for (int inner_it = 0; inner_it < FEATURE_DIM_TILE; inner_it += vec_factor) {
      // Load bf16, promote to float32
      bvec_t x_bf16 = aie::load_v<vec_factor>(input_ptr + inner_it);
      aie::accum<accfloat, vec_factor> x_acc;
      x_acc.from_vector(x_bf16, 0);
      fvec_t x = x_acc.template to_vector<float>();

      // Horner evaluation in float32: exp(-x) = 1 + x*(h1 + x*(h2 + ... + x*h16))
      #define HORNER_STEP(coeff, x, t) aie::add(coeff, aie::mul(x, t).template to_vector<float>())
      fvec_t t = h16;
      t = HORNER_STEP(h15, x, t);
      t = HORNER_STEP(h14, x, t);
      t = HORNER_STEP(h13, x, t);
      t = HORNER_STEP(h12, x, t);
      t = HORNER_STEP(h11, x, t);
      t = HORNER_STEP(h10, x, t);
      t = HORNER_STEP(h9,  x, t);
      t = HORNER_STEP(h8,  x, t);
      t = HORNER_STEP(h7,  x, t);
      t = HORNER_STEP(h6,  x, t);
      t = HORNER_STEP(h5,  x, t);
      t = HORNER_STEP(h4,  x, t);
      t = HORNER_STEP(h3,  x, t);
      t = HORNER_STEP(h2,  x, t);
      t = HORNER_STEP(h1,  x, t);
      #undef HORNER_STEP

      // exp(-x) in float32, then convert to bf16 for sigmoid/div
      fvec_t expnegx_f32 = aie::add(one, aie::mul(x, t).template to_vector<float>());
      aie::accum<accfloat, vec_factor> exp_acc;
      exp_acc.from_vector(expnegx_f32, 0);
      bvec_t expnegx = exp_acc.template to_vector<bfloat16>();

      // Sigmoid, edge cases, and SiLU in bf16 (div works for bf16)
      const bvec_t bf_one   = aie::broadcast<bfloat16, vec_factor>(1.0f);
      const bvec_t bf_zero  = aie::broadcast<bfloat16, vec_factor>(0.0f);
      const bvec_t bf_pos4  = aie::broadcast<bfloat16, vec_factor>(4.0f);
      const bvec_t bf_pos7  = aie::broadcast<bfloat16, vec_factor>(7.0f);
      const bvec_t bf_neg7  = aie::broadcast<bfloat16, vec_factor>(-7.0f);

      bvec_t sigmoid = aie::div(bf_one, aie::add(bf_one, expnegx));
      bvec_t linear  = aie::add(
          aie::mul(x_bf16, aie::broadcast<bfloat16, vec_factor>(0.0041f)).template to_vector<bfloat16>(),
          aie::broadcast<bfloat16, vec_factor>(0.9736f));
      sigmoid = aie::select(sigmoid, linear,  aie::gt(x_bf16, bf_pos4));
      sigmoid = aie::select(sigmoid, bf_one,  aie::gt(x_bf16, bf_pos7));
      sigmoid = aie::select(sigmoid, bf_zero, aie::lt(x_bf16, bf_neg7));

      bvec_t outv = aie::mul(x_bf16, sigmoid).template to_vector<bfloat16>();
      aie::store_v(output_ptr + inner_it, outv);
    }
  }
  event1();
}

extern "C" {
void silu_256_bf16(bfloat16 input_x[4][256], bfloat16 output_x[4][256]) {
    silu_bfloat16_256(input_x, output_x);
}
}
