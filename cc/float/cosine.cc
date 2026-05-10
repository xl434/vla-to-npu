/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <math.h>
#define NOCPP

constexpr float PI_F       = 3.14159265358979323846f;
constexpr float TWO_PI_F   = 6.28318530717958647692f;
constexpr float HALF_PI_F  = 1.57079632679489661923f;
constexpr float INV_TWO_PI = 1.0f / TWO_PI_F;

// SHIFT SINE BY PI/2 --------------------------------------------------------
// input restriction 0 < x < π/2 (for accuracy) 
template<typename T_in, unsigned vec_factor>
static inline aie::vector<T_in, vec_factor> reduce_halfpi(
  aie::vector<T_in, vec_factor> f,
  aie::mask<vec_factor>& neg_mask)
{
  using vec_t = aie::vector<T_in, vec_factor>;

  const vec_t inv_two_pi = aie::broadcast<T_in, vec_factor>(INV_TWO_PI);
  const vec_t two_pi     = aie::broadcast<T_in, vec_factor>(TWO_PI_F);
  const vec_t pi         = aie::broadcast<T_in, vec_factor>(PI_F);
  const vec_t half_pi    = aie::broadcast<T_in, vec_factor>(HALF_PI_F);
  const vec_t zero       = aie::broadcast<T_in, vec_factor>(0.0f);

  // cos(x) = sin(x + pi/2)
  f = aie::add(f, half_pi);

  vec_t q  = aie::mul(f, inv_two_pi);
  auto n_i = aie::to_fixed(q, 0);
  auto n_f = aie::to_float(n_i, 0);
  q        = aie::negmul(n_f, two_pi);
  vec_t r  = aie::add(f, q);

  auto m_lt0   = aie::lt(r, zero);
  r = aie::select(r, aie::add(r, two_pi), m_lt0);

  auto m_ge2pi = aie::ge(r, two_pi);
  r = aie::select(r, aie::sub(r, two_pi), m_ge2pi);

  neg_mask = aie::ge(r, pi);
  r = aie::select(r, aie::sub(r, pi), neg_mask);

  auto m_gt_hpi = aie::gt(r, half_pi);
  r = aie::select(r, aie::sub(pi, r), m_gt_hpi);

  return r;
}

float div_fac(int num) {
  float product = 1; 
  while (num > 0) {
    product *= num;
    num --;
  }
  return 1/product;
}

void cos_kernel_32x64(float input_x[32][64], float output_x[32][64]) {
  event0();

  constexpr int SEQ_TILE         = 32;
  constexpr int FEATURE_DIM_TILE = 64;
  constexpr int vec_factor       = 16;

  using vec_t  = aie::vector<float, vec_factor>;

  const vec_t cdiv3fac = aie::broadcast<float, vec_factor>(1.0f * div_fac(3));
  const vec_t cdiv5fac = aie::broadcast<float, vec_factor>(1.0f * div_fac(5));
  // const vec_t cdiv7fac = aie::broadcast<float, vec_factor>(1.0f * div_fac(7));

  for (int s = 0; s < SEQ_TILE; ++s) {
    float *__restrict input_ptr  = &input_x[s][0];
    float *__restrict output_ptr = &output_x[s][0];

    for (int inner_it = 0; inner_it < FEATURE_DIM_TILE; inner_it += vec_factor) {

      vec_t x = aie::load_v<vec_factor>(input_ptr + inner_it);

      aie::mask<vec_factor> neg_mask;
      x = reduce_halfpi<float, vec_factor>(x, neg_mask);

      // Taylor on reduced value
      vec_t x2        = aie::mul(x, x);
      vec_t x3        = aie::mul(x2, x);
      vec_t x3div3fac = aie::negmul(x3, cdiv3fac);

      vec_t x4        = aie::mul(x3, x);
      vec_t x5        = aie::mul(x4, x);
      vec_t x5div5fac = aie::mul(x5, cdiv5fac);

      // vec_t x6        = aie::mul(x5, x);
      // vec_t x7        = aie::mul(x6, x);
      // vec_t x7div7fac = aie::negmul(x7, cdiv7fac);

      vec_t outv = aie::add(x, aie::add(x3div3fac, x5div5fac));
      // outv = aie::add(outv, x7div7fac);

      outv = aie::select(outv, aie::neg(outv), neg_mask);

      aie::store_v(output_ptr + inner_it, outv);
    }
  }
  event1();
}

extern "C" {

void cos_float32(float in_mat[32][64], float out_mat[32][64]) {
    cos_kernel_32x64(in_mat, out_mat);
}

} // extern "C"