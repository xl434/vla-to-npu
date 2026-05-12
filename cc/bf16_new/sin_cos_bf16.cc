/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Fused bfloat16 sin+cos kernel: computes both sin and cos in a single pass,
 * loading x once and converting bf16→float once per vector.
 *
 * Matches sine_bf16.cc (7th-order Taylor) for both paths for consistency.
 * sin path: mod_halfpi_vec reduction
 * cos path: reduce_halfpi reduction (shift by π/2 → same Taylor)
 */
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#define NOCPP

constexpr float PI_F       = 3.14159265358979323846f;
constexpr float TWO_PI_F   = 6.28318530717958647692f;
constexpr float HALF_PI_F  = 1.57079632679489661923f;
constexpr float INV_TWO_PI = 1.0f / TWO_PI_F;

template <unsigned vec_factor>
aie::vector<float, vec_factor> bf16_to_float(aie::vector<bfloat16, vec_factor>& x_bf16) {
  aie::vector<float, vec_factor> x;
  for (int i = 0; i < vec_factor; i++) x[i] = (float)x_bf16[i];
  return x;
}

template <unsigned vec_factor>
aie::vector<bfloat16, vec_factor> float_to_bf16(aie::vector<float, vec_factor>& r) {
  aie::vector<bfloat16, vec_factor> out;
  for (int i = 0; i < vec_factor; i++) out[i] = (bfloat16)r[i];
  return out;
}

float div_fac(int num) {
  float product = 1;
  while (num > 0) { product *= num--; }
  return 1.0f / product;
}

// Reduce x to [0, π/2] for sin.
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
  auto n_i = aie::to_fixed(q, 0);
  auto n_f = aie::to_float(n_i, 0);
  q        = aie::negmul(n_f, two_pi);
  fvec_t r = aie::add(x, q);

  r = aie::select(r, aie::add(r, two_pi), aie::lt(r, zero));
  r = aie::select(r, aie::sub(r, two_pi), aie::ge(r, two_pi));

  neg_mask = aie::ge(r, pi);
  r = aie::select(r, aie::sub(r, pi), neg_mask);
  r = aie::select(r, aie::sub(pi, r), aie::gt(r, half_pi));

  return r;
}

// Reduce x to [0, π/2] for cos (cos(x) = sin(x + π/2)).
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

  f = aie::add(f, half_pi);

  vec_t q  = aie::mul(f, inv_two_pi);
  auto n_i = aie::to_fixed(q, 0);
  auto n_f = aie::to_float(n_i, 0);
  q        = aie::negmul(n_f, two_pi);
  vec_t r  = aie::add(f, q);

  r = aie::select(r, aie::add(r, two_pi), aie::lt(r, zero));
  r = aie::select(r, aie::sub(r, two_pi), aie::ge(r, two_pi));

  neg_mask = aie::ge(r, pi);
  r = aie::select(r, aie::sub(r, pi), neg_mask);
  r = aie::select(r, aie::sub(pi, r), aie::gt(r, half_pi));

  return r;
}

// 7th-order Taylor: x - x³/3! + x⁵/5! - x⁷/7!
template<unsigned vec_factor>
static inline aie::vector<float, vec_factor> taylor_sin7(
    const aie::vector<float, vec_factor>& x,
    const aie::vector<float, vec_factor>& c3,
    const aie::vector<float, vec_factor>& c5,
    const aie::vector<float, vec_factor>& c7)
{
  using fvec_t = aie::vector<float, vec_factor>;
  fvec_t x2 = aie::mul(x, x);
  fvec_t x3 = aie::mul(x2, x);
  fvec_t x4 = aie::mul(x3, x);
  fvec_t x5 = aie::mul(x4, x);
  fvec_t x6 = aie::mul(x5, x);
  fvec_t x7 = aie::mul(x6, x);
  fvec_t r  = aie::add(x, aie::negmul(x3, c3));
  r = aie::add(r, aie::mul(x5, c5));
  r = aie::add(r, aie::negmul(x7, c7));
  return r;
}

void sin_cos_bf16_kernel(bfloat16 in[32][64], bfloat16 sin_out[32][64], bfloat16 cos_out[32][64]) {
  event0();
  constexpr int SEQ_TILE         = 32;
  constexpr int FEATURE_DIM_TILE = 64;
  constexpr int vec_factor       = 16;
  using vec_t  = aie::vector<bfloat16, vec_factor>;
  using fvec_t = aie::vector<float,    vec_factor>;

  const fvec_t c3 = aie::broadcast<float, vec_factor>(div_fac(3));
  const fvec_t c5 = aie::broadcast<float, vec_factor>(div_fac(5));
  const fvec_t c7 = aie::broadcast<float, vec_factor>(div_fac(7));

  for (int s = 0; s < SEQ_TILE; ++s) {
    bfloat16 *__restrict in_ptr  = &in[s][0];
    bfloat16 *__restrict sin_ptr = &sin_out[s][0];
    bfloat16 *__restrict cos_ptr = &cos_out[s][0];

    for (int i = 0; i < FEATURE_DIM_TILE; i += vec_factor) {
      vec_t  f = aie::load_v<vec_factor>(in_ptr + i);
      fvec_t x = bf16_to_float(f);

      // sin path
      aie::mask<vec_factor> sin_neg;
      fvec_t xs = mod_halfpi_vec<vec_factor>(x, sin_neg);
      fvec_t sv = taylor_sin7<vec_factor>(xs, c3, c5, c7);
      sv = aie::select(sv, aie::neg(sv), sin_neg);
      vec_t sv_bf16 = float_to_bf16(sv);
      aie::store_v(sin_ptr + i, sv_bf16);

      // cos path
      aie::mask<vec_factor> cos_neg;
      fvec_t xc = reduce_halfpi<float, vec_factor>(x, cos_neg);
      fvec_t cv = taylor_sin7<vec_factor>(xc, c3, c5, c7);
      cv = aie::select(cv, aie::neg(cv), cos_neg);
      vec_t cv_bf16 = float_to_bf16(cv);
      aie::store_v(cos_ptr + i, cv_bf16);
    }
  }
  event1();
}

extern "C" {

void sin_cos_bfloat16(bfloat16 in[32][64], bfloat16 sin_out[32][64], bfloat16 cos_out[32][64]) {
  sin_cos_bf16_kernel(in, sin_out, cos_out);
}

} // extern "C"
