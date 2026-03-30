
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

#define EPS 1e-6f // epsilon

// bf16 -> float
template <unsigned vec_factor>
aie::vector<float, vec_factor> bf16_to_float(aie::vector<bfloat16, vec_factor>& x_bf16) {
  aie::vector<float, vec_factor> x;
  for (int i = 0; i < vec_factor; i++) {
      x[i] = (float)x_bf16[i];
  }
  return x;
}

// float -> bf16
template <unsigned vec_factor>
aie::vector<bfloat16, vec_factor> float_to_bf16(aie::vector<float, vec_factor>& r) {
  aie::vector<bfloat16, vec_factor> outv;
    for (int i = 0; i < vec_factor; i++) {
      outv[i] = (bfloat16)r[i];
    }
  return outv;
}

template<typename T_in, int vec_factor>
static inline aie::vector<T_in, vec_factor>
get_tanh_vec(const aie::vector<T_in, vec_factor>& x) {
  using vec_t = aie::vector<T_in, vec_factor>;
 
  const vec_t thr_pos = aie::broadcast<T_in, vec_factor>( 4.0f);
  const vec_t thr_neg = aie::broadcast<T_in, vec_factor>(-4.0f);
  const vec_t one     = aie::broadcast<T_in, vec_factor>( 1.0f);
  const vec_t neg_one = aie::broadcast<T_in, vec_factor>(-1.0f);
 
  const vec_t c135135 = aie::broadcast<T_in, vec_factor>(135135.0f);
  const vec_t c17325  = aie::broadcast<T_in, vec_factor>( 17325.0f);
  const vec_t c378    = aie::broadcast<T_in, vec_factor>(   378.0f);
  const vec_t c62370  = aie::broadcast<T_in, vec_factor>( 62370.0f);
  const vec_t c3150   = aie::broadcast<T_in, vec_factor>(  3150.0f);
  const vec_t c28     = aie::broadcast<T_in, vec_factor>(    28.0f);
 
  // powers of x
  vec_t x2 = aie::mul(x, x);
  vec_t x4 = aie::mul(x2, x2);
  vec_t x6 = aie::mul(x4, x2);
 
  // numerator  = x * (135135 + 17325·x² + 378·x⁴)
  vec_t temp = aie::mul(c17325, x2);
  vec_t num_inner = aie::add(c135135, temp);
  temp = aie::mul(c378, x4);
  num_inner = aie::add(num_inner, temp);
  vec_t num = aie::mul(x, num_inner);
 
  // denominator = 135135 + 62370·x² + 3150·x⁴ + 28·x⁶
  temp = aie::mul(c62370, x2);
  vec_t den = aie::add(c135135, temp);
  temp = aie::mul(c3150,  x4);
  den = aie::add(den, temp);
  temp = aie::mul(c28,    x6);
  den = aie::add(den, temp);
 
  vec_t y = aie::div(num, den);
 
  // Saturate to ±1 outside |x| > 4
  y = aie::select(y, one,     aie::gt(x, thr_pos));
  y = aie::select(y, neg_one, aie::lt(x, thr_neg));
 
  return y;
}

// Conversion: bf16 -> float -> bf16
void gelu_bfloat16(bfloat16 input_x[4][768], bfloat16 output_x[4][768]) {
  event0();
 
  constexpr int SEQ_TILE       = 4;
  constexpr int FEATURE_DIM    = 768;
  constexpr int vec_factor     = 16;

  using vec_t = aie::vector<bfloat16, vec_factor>;
  using fvec_t = aie::vector<float, vec_factor>; 
 
  const fvec_t C1       = aie::broadcast<float, vec_factor>(0.7978845608f);  // sqrt(2/π)
  const fvec_t C2       = aie::broadcast<float, vec_factor>(0.044715f);
  const fvec_t ONE      = aie::broadcast<float, vec_factor>(1.0f);
  const fvec_t ONE_HALF = aie::broadcast<float, vec_factor>(0.5f);
 
  for (int s = 0; s < SEQ_TILE; ++s) {
    bfloat16 *__restrict in_ptr  = &input_x[s][0];
    bfloat16 *__restrict out_ptr = &output_x[s][0];
 
    for (int i = 0; i < FEATURE_DIM; i += vec_factor) {

      // bfloat16 -> float
      vec_t x_bf16 = aie::load_v<vec_factor>(in_ptr + i);
      fvec_t x = bf16_to_float(x_bf16);

      // inner = C1 · x · (1 + C2·x²)
      fvec_t temp  = aie::mul(x,  x);
      temp = aie::mul(temp, x);
      fvec_t cubic = aie::mul(C2, temp);
      temp  = aie::add(x, cubic);
      fvec_t inner = aie::mul(C1, temp);

      // tanh(inner)
      fvec_t t = get_tanh_vec<float, vec_factor>(inner);

      // out = 0.5 · x · (1 + tanh(inner))
      fvec_t t1   = aie::add(t, ONE);
      temp = aie::mul(x, t1);
      fvec_t outf = aie::mul(ONE_HALF, temp);

      // float -> bfloat16
      vec_t outv = float_to_bf16(outf);
      aie::store_v(out_ptr + i, outv);
    }
  event1();
  }
}

// // No conversion
// void gelu_bfloat16(bfloat16 input_x[4][768], bfloat16 output_x[4][768]) {
//   event0();
 
//   constexpr int SEQ_TILE       = 4;
//   constexpr int FEATURE_DIM    = 768;
//   constexpr int vec_factor     = 16;

//   using vec_t = aie::vector<bfloat16, vec_factor>;
 
//   const vec_t C1       = aie::broadcast<bfloat16, vec_factor>(0.7978845608f);  // sqrt(2/π)
//   const vec_t C2       = aie::broadcast<bfloat16, vec_factor>(0.044715f);
//   const vec_t ONE      = aie::broadcast<bfloat16, vec_factor>(1.0f);
//   const vec_t ONE_HALF = aie::broadcast<bfloat16, vec_factor>(0.5f);
 
//   for (int s = 0; s < SEQ_TILE; ++s) {
//     bfloat16 *__restrict in_ptr  = &input_x[s][0];
//     bfloat16 *__restrict out_ptr = &output_x[s][0];
 
//     for (int i = 0; i < FEATURE_DIM; i += vec_factor) {

//       // bfloat16 -> float
//       vec_t x = aie::load_v<vec_factor>(in_ptr + i);

//       // inner = C1 · x · (1 + C2·x²)
//       vec_t temp  = aie::mul(x,  x);
//       temp = aie::mul(temp, x);
//       vec_t cubic = aie::mul(C2, temp);
//       temp  = aie::add(x, cubic);
//       vec_t inner = aie::mul(C1, temp);

//       // tanh(inner)
//       vec_t t = get_tanh_vec<bfloat16, vec_factor>(inner);

//       // out = 0.5 · x · (1 + tanh(inner))
//       vec_t t1   = aie::add(t, ONE);
//       temp = aie::mul(x, t1);
//       vec_t outv = aie::mul(ONE_HALF, temp);

//       aie::store_v(out_ptr + i, outv);
//     }
//   event1();
//   }
// }

extern "C" {

void gelu(bfloat16 input_x[4][768], bfloat16 output_x[4][768]) {
gelu_bfloat16(input_x, output_x);
}

} // extern "C"
