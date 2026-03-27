/*
 * SiLU bf16 for per-core tile [4][256] - test variant
 */
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#define NOCPP

float div_fac(int num) {
  float product = 1;
  while (num > 0) { product *= num; num--; }
  return 1 / product;
}

template <typename T_in, int vec_factor>
aie::vector<T_in, vec_factor> linear_approx(aie::vector<T_in, vec_factor> &x) {
  using vec_t = aie::vector<T_in, vec_factor>;
  vec_t c0_0041 = aie::broadcast<T_in, vec_factor>(0.0041f);
  vec_t c0_9736 = aie::broadcast<T_in, vec_factor>(0.9736f);
  vec_t outv = aie::mul(x, c0_0041);
  return aie::add(outv, c0_9736);
}

void silu_bfloat16_256(bfloat16 input_x[4][256], bfloat16 output_x[4][256]) {
  event0();
  constexpr int SEQ_TILE         = 4;
  constexpr int FEATURE_DIM_TILE = 256;
  constexpr int vec_factor       = 32;
  using vec_t = aie::vector<bfloat16, vec_factor>;

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
  const vec_t fpos4 = aie::broadcast<bfloat16, vec_factor>(2.5f);
  const vec_t fpos7 = aie::broadcast<bfloat16, vec_factor>(7.0f);
  const vec_t fneg7 = aie::broadcast<bfloat16, vec_factor>(-7.0f);

  for (int s = 0; s < SEQ_TILE; ++s) {
    bfloat16 *__restrict input_ptr  = &input_x[s][0];
    bfloat16 *__restrict output_ptr = &output_x[s][0];
    for (int inner_it = 0; inner_it < FEATURE_DIM_TILE; inner_it += vec_factor) {
      vec_t x = aie::load_v<vec_factor>(input_ptr + inner_it);
      vec_t sum  = aie::negmul(x, one);
      vec_t powx = aie::mul(x, x);
      vec_t temp = aie::mul(powx, c2divfac);
      sum = aie::add(sum, temp);
      powx = aie::mul(powx, x); temp = aie::negmul(powx, c3divfac); sum = aie::add(sum, temp);
      powx = aie::mul(powx, x); temp = aie::mul(powx, c4divfac);    sum = aie::add(sum, temp);
      powx = aie::mul(powx, x); temp = aie::negmul(powx, c5divfac); sum = aie::add(sum, temp);
      powx = aie::mul(powx, x); temp = aie::mul(powx, c6divfac);    sum = aie::add(sum, temp);
      powx = aie::mul(powx, x); temp = aie::negmul(powx, c7divfac); sum = aie::add(sum, temp);
      powx = aie::mul(powx, x); temp = aie::mul(powx, c8divfac);    sum = aie::add(sum, temp);
      powx = aie::mul(powx, x); temp = aie::negmul(powx, c9divfac); sum = aie::add(sum, temp);
      powx = aie::mul(powx, x); temp = aie::mul(powx, c10divfac);   sum = aie::add(sum, temp);
      powx = aie::mul(powx, x); temp = aie::negmul(powx, c11divfac);sum = aie::add(sum, temp);
      powx = aie::mul(powx, x); temp = aie::mul(powx, c12divfac);   sum = aie::add(sum, temp);
      powx = aie::mul(powx, x); temp = aie::negmul(powx, c13divfac);sum = aie::add(sum, temp);
      powx = aie::mul(powx, x); temp = aie::mul(powx, c14divfac);   sum = aie::add(sum, temp);
      powx = aie::mul(powx, x); temp = aie::negmul(powx, c15divfac);sum = aie::add(sum, temp);
      powx = aie::mul(powx, x); temp = aie::mul(powx, c16divfac);   sum = aie::add(sum, temp);
      vec_t expnegx = aie::add(one, sum);
      vec_t sigmoid = aie::div(one, aie::add(one, expnegx));
      vec_t linear  = linear_approx<bfloat16, vec_factor>(x);
      sigmoid = aie::select(sigmoid, linear, aie::gt(x, fpos4));
      sigmoid = aie::select(sigmoid, one,    aie::gt(x, fpos7));
      sigmoid = aie::select(sigmoid, fzero,  aie::lt(x, fneg7));
      vec_t outv = aie::mul(x, sigmoid);
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
