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

// Sigmoid approx -----------------------------
float div_fac(int num) {
  float product = 1; 
  while (num > 0) {
    product *= num;
    num --;
  }
  return 1/product;
}

template <typename T_in, int vec_factor>
aie::vector<T_in, vec_factor> linear_approx(aie::vector<T_in, vec_factor> &x) {
  using vec_t = aie::vector<T_in, vec_factor>;
  vec_t c0_0041 = aie::broadcast<T_in, vec_factor>(0.0041f);
  vec_t c0_9736 = aie::broadcast<T_in, vec_factor>(0.9736f);
  vec_t outv = aie::mul(x, c0_0041);
  return aie::add(outv, c0_9736);
}

// // Conversion: bfloat16 -> float -> bfloat16 
// void silu_bfloat16(bfloat16 input_x[4][768], bfloat16 output_x[4][768]) {
//   event0();
//   constexpr int SEQ_TILE         = 4;
//   constexpr int FEATURE_DIM_TILE = 768;
//   constexpr int vec_factor       = 32;

//   using vec_t  = aie::vector<bfloat16, vec_factor>; 
//   using fvec_t = aie::vector<float,    vec_factor>;

//   const fvec_t one       = aie::broadcast<float, vec_factor>(1.0f);
//   const fvec_t c2divfac  = aie::broadcast<float, vec_factor>(1.0f * div_fac(2));
//   const fvec_t c3divfac  = aie::broadcast<float, vec_factor>(1.0f * div_fac(3));
//   const fvec_t c4divfac  = aie::broadcast<float, vec_factor>(1.0f * div_fac(4));
//   const fvec_t c5divfac  = aie::broadcast<float, vec_factor>(1.0f * div_fac(5));
//   const fvec_t c6divfac  = aie::broadcast<float, vec_factor>(1.0f * div_fac(6));
//   const fvec_t c7divfac  = aie::broadcast<float, vec_factor>(1.0f * div_fac(7));
//   const fvec_t c8divfac  = aie::broadcast<float, vec_factor>(1.0f * div_fac(8));
//   const fvec_t c9divfac  = aie::broadcast<float, vec_factor>(1.0f * div_fac(9));
//   const fvec_t c10divfac = aie::broadcast<float, vec_factor>(1.0f * div_fac(10));
//   const fvec_t c11divfac = aie::broadcast<float, vec_factor>(1.0f * div_fac(11));
//   const fvec_t c12divfac = aie::broadcast<float, vec_factor>(1.0f * div_fac(12));
//   const fvec_t c13divfac = aie::broadcast<float, vec_factor>(1.0f * div_fac(13));
//   const fvec_t c14divfac = aie::broadcast<float, vec_factor>(1.0f * div_fac(14));
//   const fvec_t c15divfac = aie::broadcast<float, vec_factor>(1.0f * div_fac(15));
//   const fvec_t c16divfac = aie::broadcast<float, vec_factor>(1.0f * div_fac(16));

//   const fvec_t fzero = aie::broadcast<float, vec_factor>(0.0f);
//   const fvec_t fpos4 = aie::broadcast<float, vec_factor>(4.0f);
//   const fvec_t fpos7 = aie::broadcast<float, vec_factor>(7.0f);
//   const fvec_t fneg7 = aie::broadcast<float, vec_factor>(-7.0f);

//   for (int s = 0; s < SEQ_TILE; ++s) {
//     bfloat16 *__restrict input_ptr  = &input_x[s][0];
//     bfloat16 *__restrict output_ptr = &output_x[s][0];

//     for (int inner_it = 0; inner_it < FEATURE_DIM_TILE; inner_it += vec_factor) {

//       // bfloat16 -> float
//       vec_t x_bf16 = aie::load_v<vec_factor>(input_ptr + inner_it);
//       fvec_t x = bf16_to_float(x_bf16);

//       // Taylor series for exp(-x)
//       fvec_t sum  = aie::negmul(x, one);  // -x
//       fvec_t powx = aie::mul(x, x);       // x^2
//       fvec_t temp = aie::mul(powx, c2divfac);
//       sum = aie::add(sum, temp);          // + x^2/2!

//       powx = aie::mul(powx, x);           // x^3
//       temp = aie::negmul(powx, c3divfac);
//       sum = aie::add(sum, temp);          // - x^3/3!

//       powx = aie::mul(powx, x);           // x^4
//       temp = aie::mul(powx, c4divfac);
//       sum = aie::add(sum, temp);          // + x^4/4!

//       powx = aie::mul(powx, x);           // x^5
//       temp = aie::negmul(powx, c5divfac);
//       sum = aie::add(sum, temp);          // - x^5/5!

//       powx = aie::mul(powx, x);           // x^6
//       temp = aie::mul(powx, c6divfac);
//       sum = aie::add(sum, temp);          // + x^6/6!

//       powx = aie::mul(powx, x);           // x^7
//       temp = aie::negmul(powx, c7divfac);
//       sum = aie::add(sum, temp);          // - x^7/7!

//       powx = aie::mul(powx, x);           // x^8
//       temp = aie::mul(powx, c8divfac);
//       sum = aie::add(sum, temp);          // + x^8/8!

//       powx = aie::mul(powx, x);           // x^9
//       temp = aie::negmul(powx, c9divfac);
//       sum = aie::add(sum, temp);          // - x^9/9!

//       powx = aie::mul(powx, x);           // x^10
//       temp = aie::mul(powx, c10divfac);
//       sum = aie::add(sum, temp);          // + x^10/10!

//       powx = aie::mul(powx, x);           // x^11
//       temp = aie::negmul(powx, c11divfac);
//       sum = aie::add(sum, temp);          // - x^11/11!

//       powx = aie::mul(powx, x);           // x^12
//       temp = aie::mul(powx, c12divfac);
//       sum = aie::add(sum, temp);          // + x^12/12!

//       powx = aie::mul(powx, x);           // x^13
//       temp = aie::negmul(powx, c13divfac);
//       sum = aie::add(sum, temp);          // - x^13/13!

//       powx = aie::mul(powx, x);           // x^14
//       temp = aie::mul(powx, c14divfac);
//       sum = aie::add(sum, temp);          // + x^14/14!

//       powx = aie::mul(powx, x);           // x^15
//       temp = aie::negmul(powx, c15divfac);
//       sum = aie::add(sum, temp);          // - x^15/15!

//       powx = aie::mul(powx, x);           // x^16
//       temp = aie::mul(powx, c16divfac);
//       sum = aie::add(sum, temp);          // + x^16/16!

//       fvec_t expnegx = aie::add(one, sum);

//       fvec_t sigmoid = aie::div(one, aie::add(one, expnegx));
//       fvec_t linear  = linear_approx<float, vec_factor>(x);
//       sigmoid = aie::select(sigmoid, linear, aie::gt(x, fpos4));
//       sigmoid = aie::select(sigmoid, one,    aie::gt(x, fpos7));
//       sigmoid = aie::select(sigmoid, fzero,  aie::lt(x, fneg7));

//       fvec_t r = aie::mul(x, sigmoid);

//       // float -> bfloat16
//       vec_t outv = float_to_bf16(r);
//       aie::store_v(output_ptr + inner_it, outv);
//     }
//   }
//   event1();
// }

// No conversion
void silu_bfloat16(bfloat16 input_x[4][768], bfloat16 output_x[4][768]) {
  event0();
  constexpr int SEQ_TILE         = 4;
  constexpr int FEATURE_DIM_TILE = 768;
  constexpr int vec_factor       = 32;

  using vec_t  = aie::vector<bfloat16, vec_factor>; 

  const vec_t one       = aie::broadcast<bfloat16, vec_factor>(1.0f);
  const vec_t c2divfac  = aie::broadcast<bfloat16, vec_factor>(1.0f * div_fac(2));
  const vec_t c3divfac  = aie::broadcast<bfloat16, vec_factor>(1.0f * div_fac(3));
  const vec_t c4divfac  = aie::broadcast<bfloat16, vec_factor>(1.0f * div_fac(4));
  const vec_t c5divfac  = aie::broadcast<bfloat16, vec_factor>(1.0f * div_fac(5));
  const vec_t c6divfac  = aie::broadcast<bfloat16, vec_factor>(1.0f * div_fac(6));
  const vec_t c7divfac  = aie::broadcast<bfloat16, vec_factor>(1.0f * div_fac(7));
  const vec_t c8divfac  = aie::broadcast<bfloat16, vec_factor>(1.0f * div_fac(8));
  const vec_t c9divfac  = aie::broadcast<bfloat16, vec_factor>(1.0f * div_fac(9));
  const vec_t c10divfac = aie::broadcast<bfloat16, vec_factor>(1.0f * div_fac(10));
  const vec_t c11divfac = aie::broadcast<bfloat16, vec_factor>(1.0f * div_fac(11));
  const vec_t c12divfac = aie::broadcast<bfloat16, vec_factor>(1.0f * div_fac(12));
  const vec_t c13divfac = aie::broadcast<bfloat16, vec_factor>(1.0f * div_fac(13));
  const vec_t c14divfac = aie::broadcast<bfloat16, vec_factor>(1.0f * div_fac(14));
  const vec_t c15divfac = aie::broadcast<bfloat16, vec_factor>(1.0f * div_fac(15));
  const vec_t c16divfac = aie::broadcast<bfloat16, vec_factor>(1.0f * div_fac(16));

  const vec_t fzero = aie::broadcast<bfloat16, vec_factor>(0.0f);
  const vec_t fpos4 = aie::broadcast<bfloat16, vec_factor>(4.0f);
  const vec_t fpos7 = aie::broadcast<bfloat16, vec_factor>(7.0f);
  const vec_t fneg7 = aie::broadcast<bfloat16, vec_factor>(-7.0f);

  for (int s = 0; s < SEQ_TILE; ++s) {
    bfloat16 *__restrict input_ptr  = &input_x[s][0];
    bfloat16 *__restrict output_ptr = &output_x[s][0];

    for (int inner_it = 0; inner_it < FEATURE_DIM_TILE; inner_it += vec_factor) {

      // bfloat16 -> float
      vec_t x = aie::load_v<vec_factor>(input_ptr + inner_it);

      // Taylor series for exp(-x)
      vec_t sum  = aie::negmul(x, one);  // -x
      vec_t powx = aie::mul(x, x);       // x^2
      vec_t temp = aie::mul(powx, c2divfac);
      sum = aie::add(sum, temp);          // + x^2/2!

      powx = aie::mul(powx, x);           // x^3
      temp = aie::negmul(powx, c3divfac);
      sum = aie::add(sum, temp);          // - x^3/3!

      powx = aie::mul(powx, x);           // x^4
      temp = aie::mul(powx, c4divfac);
      sum = aie::add(sum, temp);          // + x^4/4!

      powx = aie::mul(powx, x);           // x^5
      temp = aie::negmul(powx, c5divfac);
      sum = aie::add(sum, temp);          // - x^5/5!

      powx = aie::mul(powx, x);           // x^6
      temp = aie::mul(powx, c6divfac);
      sum = aie::add(sum, temp);          // + x^6/6!

      powx = aie::mul(powx, x);           // x^7
      temp = aie::negmul(powx, c7divfac);
      sum = aie::add(sum, temp);          // - x^7/7!

      powx = aie::mul(powx, x);           // x^8
      temp = aie::mul(powx, c8divfac);
      sum = aie::add(sum, temp);          // + x^8/8!

      powx = aie::mul(powx, x);           // x^9
      temp = aie::negmul(powx, c9divfac);
      sum = aie::add(sum, temp);          // - x^9/9!

      powx = aie::mul(powx, x);           // x^10
      temp = aie::mul(powx, c10divfac);
      sum = aie::add(sum, temp);          // + x^10/10!

      powx = aie::mul(powx, x);           // x^11
      temp = aie::negmul(powx, c11divfac);
      sum = aie::add(sum, temp);          // - x^11/11!

      powx = aie::mul(powx, x);           // x^12
      temp = aie::mul(powx, c12divfac);
      sum = aie::add(sum, temp);          // + x^12/12!

      powx = aie::mul(powx, x);           // x^13
      temp = aie::negmul(powx, c13divfac);
      sum = aie::add(sum, temp);          // - x^13/13!

      powx = aie::mul(powx, x);           // x^14
      temp = aie::mul(powx, c14divfac);
      sum = aie::add(sum, temp);          // + x^14/14!

      powx = aie::mul(powx, x);           // x^15
      temp = aie::negmul(powx, c15divfac);
      sum = aie::add(sum, temp);          // - x^15/15!

      powx = aie::mul(powx, x);           // x^16
      temp = aie::mul(powx, c16divfac);
      sum = aie::add(sum, temp);          // + x^16/16!

      vec_t expnegx = aie::add(one, sum);

      vec_t sigmoid = aie::div(one, aie::add(one, expnegx));
      vec_t linear  = linear_approx<bfloat16, vec_factor>(x);       // linear region (4 < x < 7)
      sigmoid = aie::select(sigmoid, linear, aie::gt(x, fpos4));    // use linear approx if x > 4
      sigmoid = aie::select(sigmoid, one,    aie::gt(x, fpos7));    // 1 if x > 7
      sigmoid = aie::select(sigmoid, fzero,  aie::lt(x, fneg7));    // 0 if x < -7

      vec_t outv = aie::mul(x, sigmoid);

      aie::store_v(output_ptr + inner_it, outv);
    }
  }
  event1();
}

extern "C" {

void silu_bf16(bfloat16 input_x[4][768], bfloat16 output_x[4][768]) {
silu_bfloat16(input_x, output_x);
}

} // extern "C"
