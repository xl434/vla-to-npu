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

// --------- Sine LUT over [0, 2π] with linear interpolation ---------
constexpr float PI_F       = 3.14159265358979323846f;
constexpr float TWO_PI_F   = 6.28318530717958647692f;
constexpr float HALF_PI_F  = 1.57079632679489661923f;
constexpr float INV_TWO_PI = 1.0f / TWO_PI_F;

// ----------------------------------------------------------------
// input restriction 0 < x < π/2 (for accuracy) 
template<unsigned vec_factor>
static inline aie::vector<float, vec_factor> mod_halfpi_vec(
  const aie::vector<float, vec_factor>& x,
  aie::mask<vec_factor>& neg_mask)
{
  using fvec_t = aie::vector<float, vec_factor>;

  const fvec_t inv_two_pi = aie::broadcast<float, vec_factor>(INV_TWO_PI);
  const fvec_t two_pi     = aie::broadcast<float, vec_factor>(TWO_PI_F);
  const fvec_t pi         = aie::broadcast<float, vec_factor>(PI_F);
  const fvec_t half_pi    = aie::broadcast<float, vec_factor>(HALF_PI_F);
  const fvec_t zero       = aie::broadcast<float, vec_factor>(0.0f);

  fvec_t q = aie::mul(x, inv_two_pi);
  auto n_i      = aie::to_fixed(q, 0);
  auto n_f      = aie::to_float(n_i, 0);
  q             = aie::negmul(n_f, two_pi);
  fvec_t r = aie::add(x, q);

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

// low-degree Taylor polynomial (float version)
void sin_f32(float input_x[32][64], float output_x[32][64]) {
  event0();
  constexpr int SEQ_TILE          = 32;
  constexpr int FEATURE_DIM_TILE  = 64;
  constexpr int vec_factor        = 16;
  using fvec_t = aie::vector<float, vec_factor>;

  const fvec_t cdiv3fac = aie::broadcast<float, vec_factor>(1.0f * div_fac(3));
  const fvec_t cdiv5fac = aie::broadcast<float, vec_factor>(1.0f * div_fac(5));
  // const fvec_t cdiv7fac = aie::broadcast<float, vec_factor>(1.0f * div_fac(7));

  for (int s = 0; s < SEQ_TILE; ++s) {
    float *__restrict input_ptr  = &input_x[s][0];
    float *__restrict output_ptr = &output_x[s][0];

    for (int inner_it = 0; inner_it < FEATURE_DIM_TILE; inner_it += vec_factor) {
      fvec_t x = aie::load_v<vec_factor>(input_ptr + inner_it);

      // Reduce to [0, π/2]
      // neg_mask=true where sin should be negated
      aie::mask<vec_factor> neg_mask;

      x = mod_halfpi_vec(x, neg_mask);

      // Taylor: x - x³/3! + x⁵/5! - x^7/7!
      fvec_t xtemp          = aie::mul(x, x);              // x^2
      fvec_t x3             = aie::mul(xtemp, x);          // x^3
      fvec_t x3div3fac      = aie::negmul(x3, cdiv3fac);   // -x^3/3!
      xtemp                 = aie::mul(x3, x);             // x^4
      fvec_t x5             = aie::mul(xtemp, x);          // x^5
      fvec_t x5div5fac      = aie::mul(x5, cdiv5fac);      // +x^5/5!
      // xtemp                 = aie::mul(x5, x);             // x^6
      // fvec_t x7             = aie::mul(xtemp, x);          // x^7
      // fvec_t x7div7fac      = aie::negmul(x7, cdiv7fac);   // -x^7/7!

      xtemp = aie::add(x3div3fac, x5div5fac);
      // xtemp = aie::add(xtemp, x7div7fac);

      fvec_t outv = aie::add(x, xtemp);

      // Restore sign for quadrants where sin is negative
      outv = aie::select(outv, aie::neg(outv), neg_mask);

      // Store 16 floats
      aie::store_v(output_ptr + inner_it, outv);
    }
  }
  event1();
}

extern "C" {

// float version sine
void sin_float32(float in_mat[32][64], float out_mat[32][64]) {
  sin_f32(in_mat, out_mat);
}

} // extern "C"