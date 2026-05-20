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

// constexpr int SIN_LUT_SIZE = 257;
// constexpr float SIN_STEP = TWO_PI_F / (SIN_LUT_SIZE - 1);


// static const float SIN_LUT[SIN_LUT_SIZE] = {
// 0.00000000f, 0.02454123f, 0.04906767f, 0.07356456f, 0.09801714f, 0.12241068f, 0.14673047f, 0.17096189f, 
// 0.19509032f, 0.21910124f, 0.24298018f, 0.26671276f, 0.29028468f, 0.31368174f, 0.33688985f, 0.35989504f, 
// 0.38268343f, 0.40524131f, 0.42755509f, 0.44961133f, 0.47139674f, 0.49289819f, 0.51410274f, 0.53499762f, 
// 0.55557023f, 0.57580819f, 0.59569930f, 0.61523159f, 0.63439328f, 0.65317284f, 0.67155895f, 0.68954054f, 
// 0.70710678f, 0.72424708f, 0.74095113f, 0.75720885f, 0.77301045f, 0.78834643f, 0.80320753f, 0.81758481f, 
// 0.83146961f, 0.84485357f, 0.85772861f, 0.87008699f, 0.88192126f, 0.89322430f, 0.90398929f, 0.91420976f, 
// 0.92387953f, 0.93299280f, 0.94154407f, 0.94952818f, 0.95694034f, 0.96377607f, 0.97003125f, 0.97570213f, 
// 0.98078528f, 0.98527764f, 0.98917651f, 0.99247953f, 0.99518473f, 0.99729046f, 0.99879546f, 0.99969882f, 
// 1.00000000f, 0.99969882f, 0.99879546f, 0.99729046f, 0.99518473f, 0.99247953f, 0.98917651f, 0.98527764f, 
// 0.98078528f, 0.97570213f, 0.97003125f, 0.96377607f, 0.95694034f, 0.94952818f, 0.94154407f, 0.93299280f, 
// 0.92387953f, 0.91420976f, 0.90398929f, 0.89322430f, 0.88192126f, 0.87008699f, 0.85772861f, 0.84485357f, 
// 0.83146961f, 0.81758481f, 0.80320753f, 0.78834643f, 0.77301045f, 0.75720885f, 0.74095113f, 0.72424708f, 
// 0.70710678f, 0.68954054f, 0.67155895f, 0.65317284f, 0.63439328f, 0.61523159f, 0.59569930f, 0.57580819f, 
// 0.55557023f, 0.53499762f, 0.51410274f, 0.49289819f, 0.47139674f, 0.44961133f, 0.42755509f, 0.40524131f, 
// 0.38268343f, 0.35989504f, 0.33688985f, 0.31368174f, 0.29028468f, 0.26671276f, 0.24298018f, 0.21910124f, 
// 0.19509032f, 0.17096189f, 0.14673047f, 0.12241068f, 0.09801714f, 0.07356456f, 0.04906767f, 0.02454123f, 
// 0.00000000f, -0.02454123f, -0.04906767f, -0.07356456f, -0.09801714f, -0.12241068f, -0.14673047f, -0.17096189f, 
// -0.19509032f, -0.21910124f, -0.24298018f, -0.26671276f, -0.29028468f, -0.31368174f, -0.33688985f, -0.35989504f, 
// -0.38268343f, -0.40524131f, -0.42755509f, -0.44961133f, -0.47139674f, -0.49289819f, -0.51410274f, -0.53499762f, 
// -0.55557023f, -0.57580819f, -0.59569930f, -0.61523159f, -0.63439328f, -0.65317284f, -0.67155895f, -0.68954054f, 
// -0.70710678f, -0.72424708f, -0.74095113f, -0.75720885f, -0.77301045f, -0.78834643f, -0.80320753f, -0.81758481f, 
// -0.83146961f, -0.84485357f, -0.85772861f, -0.87008699f, -0.88192126f, -0.89322430f, -0.90398929f, -0.91420976f, 
// -0.92387953f, -0.93299280f, -0.94154407f, -0.94952818f, -0.95694034f, -0.96377607f, -0.97003125f, -0.97570213f, 
// -0.98078528f, -0.98527764f, -0.98917651f, -0.99247953f, -0.99518473f, -0.99729046f, -0.99879546f, -0.99969882f, 
// -1.00000000f, -0.99969882f, -0.99879546f, -0.99729046f, -0.99518473f, -0.99247953f, -0.98917651f, -0.98527764f, 
// -0.98078528f, -0.97570213f, -0.97003125f, -0.96377607f, -0.95694034f, -0.94952818f, -0.94154407f, -0.93299280f, 
// -0.92387953f, -0.91420976f, -0.90398929f, -0.89322430f, -0.88192126f, -0.87008699f, -0.85772861f, -0.84485357f, 
// -0.83146961f, -0.81758481f, -0.80320753f, -0.78834643f, -0.77301045f, -0.75720885f, -0.74095113f, -0.72424708f, 
// -0.70710678f, -0.68954054f, -0.67155895f, -0.65317284f, -0.63439328f, -0.61523159f, -0.59569930f, -0.57580819f, 
// -0.55557023f, -0.53499762f, -0.51410274f, -0.49289819f, -0.47139674f, -0.44961133f, -0.42755509f, -0.40524131f, 
// -0.38268343f, -0.35989504f, -0.33688985f, -0.31368174f, -0.29028468f, -0.26671276f, -0.24298018f, -0.21910124f, 
// -0.19509032f, -0.17096189f, -0.14673047f, -0.12241068f, -0.09801714f, -0.07356456f, -0.04906767f, -0.02454123f, 
// 0.00000000f, };


// static inline float mod2pi(float x) {
//   float q = x * INV_TWO_PI;
//   int n = (int)q;                 // trunc toward zero
//   float r = x - (float)n * TWO_PI_F;

//   if (r < 0.0f) r += TWO_PI_F;
//   if (r >= TWO_PI_F) r -= TWO_PI_F;
//   return r;
// }

// // vector version
// template<int vec_factor>
// aie::vector<float, vec_factor> mod2pi_vec(const aie::vector<float, vec_factor>& x) {
//   aie::vector<float, vec_factor> y;
//   #pragma unroll
//   // for (int lane = 0; lane < vec_factor; ++lane){
//   //   y[lane] = mod2pi(x[lane]);
//   // }
//   for (int lane = 0; lane < vec_factor; lane += 8){
//     y[lane] = mod2pi(x[lane]);
//     y[lane + 1] = mod2pi(x[lane + 1]);
//     y[lane + 2] = mod2pi(x[lane + 2]);
//     y[lane + 3] = mod2pi(x[lane + 3]);
//     y[lane + 4] = mod2pi(x[lane + 4]);
//     y[lane + 5] = mod2pi(x[lane + 5]);
//     y[lane + 6] = mod2pi(x[lane + 6]);
//     y[lane + 7] = mod2pi(x[lane + 7]);
//   }
//   return y;
// }


// static inline float sin_lut2pi(float x) {
//   float r = mod2pi(x);               // [0, 2π)
//   constexpr int   SIN_LUT_SIZE = 257;
//   constexpr float SIN_STEP     = TWO_PI_F / (SIN_LUT_SIZE - 1);

//   float idx_f = r / SIN_STEP;        // [0, SIN_LUT_SIZE-1]
//   int   i     = (int)idx_f;          // trunc
//   float frac  = idx_f - (float)i;

//   if (i >= SIN_LUT_SIZE - 1) { i = SIN_LUT_SIZE - 2; frac = 1.0f; }

//   extern const float SIN_LUT[SIN_LUT_SIZE];
//   float y0 = SIN_LUT[i];
//   float y1 = SIN_LUT[i + 1];
//   return y0 + frac * (y1 - y0);
// }

// // sine LUT for vectorized 
// template<int vec_factor>
// aie::vector<float, vec_factor> sin_lut2pi_vec(const aie::vector<float, vec_factor>& x) {
//   aie::vector<float, vec_factor> y;
//   #pragma unroll
//   // for (int lane = 0; lane < vec_factor; ++lane){
//   //   y[lane] = sin_lut2pi(x[lane]);
//   // }
//   for (int lane = 0; lane < vec_factor; lane += 8){
//     y[lane] = sin_lut2pi(x[lane]);
//     y[lane + 1] = sin_lut2pi(x[lane + 1]);
//     y[lane + 2] = sin_lut2pi(x[lane + 2]);
//     y[lane + 3] = sin_lut2pi(x[lane + 3]);
//     y[lane + 4] = sin_lut2pi(x[lane + 4]);
//     y[lane + 5] = sin_lut2pi(x[lane + 5]);
//     y[lane + 6] = sin_lut2pi(x[lane + 6]);
//     y[lane + 7] = sin_lut2pi(x[lane + 7]);
//   }
//   return y;
// }

// // vectorized version sine
// static inline void sin_kernel_32x64(float in_mat[32][64], float out_mat[32][64]) {
//   constexpr int TILE_ROWS  = 32;
//   constexpr int COLS       = 64;
//   constexpr int vec_factor = 8; 

//   for (int r = 0; r < TILE_ROWS; ++r) {
//     float*  __restrict in_row  = &in_mat[r][0];
//     float* __restrict out_row = &out_mat[r][0];

//     for (int c = 0; c < COLS; c += 2 * vec_factor) {
//       // load 2 vectors = 2*vec_factor elements
//       aie::vector<float, vec_factor> v0 = aie::load_v<vec_factor>(in_row + c);
//       aie::vector<float, vec_factor> v1 = aie::load_v<vec_factor>(in_row + c + vec_factor);

//       // // convert to float for LUT
//       // aie::vector<float, vec_factor> v0 = aie::vector_cast<float>(v0_in);
//       // aie::vector<float, vec_factor> v1 = aie::vector_cast<float>(v1_in);

//       // compute
//       aie::vector<float, vec_factor> y0 = sin_lut2pi_vec<vec_factor>(v0);
//       aie::vector<float, vec_factor> y1 = sin_lut2pi_vec<vec_factor>(v1);

//       // convert back
//       // aie::vector<float, vec_factor> y0 = aie::vector_cast<float>(y0_f);
//       // aie::vector<float, vec_factor> y1 = aie::vector_cast<float>(y1_f);

//       // store
//       aie::store_v(out_row + c, y0);
//       aie::store_v(out_row + c + vec_factor, y1);
//     }
//   }
// }

// ----------------------------------------------------------------
// input restriction 0 < x < π/2 (for accuracy) 
template<unsigned vec_factor>
static inline aie::vector<bfloat16, vec_factor> mod_halfpi_vec(
  const aie::vector<bfloat16, vec_factor>& x,
  aie::mask<vec_factor>& neg_mask)
{
  using vec_t      = aie::vector<bfloat16, vec_factor>;
  using float_vec_t = aie::vector<float, vec_factor>;

  // bfloat16 → float
  float_vec_t f;
  for (int i = 0; i < vec_factor; i++) {
    f[i] = (float)x[i];
  }

  const float_vec_t inv_two_pi = aie::broadcast<float, vec_factor>(INV_TWO_PI);
  const float_vec_t two_pi     = aie::broadcast<float, vec_factor>(TWO_PI_F);
  const float_vec_t pi         = aie::broadcast<float, vec_factor>(PI_F);
  const float_vec_t half_pi    = aie::broadcast<float, vec_factor>(HALF_PI_F);
  const float_vec_t zero       = aie::broadcast<float, vec_factor>(0.0f);

  float_vec_t q = aie::mul(f, inv_two_pi);
  auto n_i      = aie::to_fixed(q, 0);
  auto n_f      = aie::to_float(n_i, 0);
  q             = aie::negmul(n_f, two_pi);
  float_vec_t r = aie::add(f, q);

  auto m_lt0   = aie::lt(r, zero);
  r = aie::select(r, aie::add(r, two_pi), m_lt0);
  auto m_ge2pi = aie::ge(r, two_pi);
  r = aie::select(r, aie::sub(r, two_pi), m_ge2pi);

  neg_mask = aie::ge(r, pi);
  r = aie::select(r, aie::sub(r, pi), neg_mask);

  auto m_gt_hpi = aie::gt(r, half_pi);
  r = aie::select(r, aie::sub(pi, r), m_gt_hpi);

  // float → bfloat16
  vec_t outv;
  for (int i = 0; i < vec_factor; i++){
    outv[i] = (bfloat16)r[i];
  }

  return outv;
}

float div_fac(int num) {
  float product = 1; 
  while (num > 0) {
    product *= num;
    num --;
  }
  return 1/product;
}

// low-degree Taylor polynomial
void sin_kernel_32x64(bfloat16 input_x[32][64], bfloat16 output_x[32][64]) {
  event0();
  constexpr int SEQ_TILE          = 32;
  constexpr int FEATURE_DIM_TILE  = 64;
  // constexpr int vec_factor        = 16;
  constexpr int vec_factor        = 32;
  using vec_t = aie::vector<bfloat16, vec_factor>;

  const vec_t cdiv3fac = aie::broadcast<bfloat16, vec_factor>(1.0f * div_fac(3));
  const vec_t cdiv5fac = aie::broadcast<bfloat16, vec_factor>(1.0f * div_fac(5));
  // const vec_t cdiv7fac = aie::broadcast<bfloat16, vec_factor>(1.0f * div_fac(7));

  for (int s = 0; s < SEQ_TILE; ++s) {
    bfloat16 *__restrict input_ptr  = &input_x[s][0];
    bfloat16 *__restrict output_ptr = &output_x[s][0];

    for (int inner_it = 0; inner_it < FEATURE_DIM_TILE; inner_it += vec_factor) {
      vec_t x = aie::load_v<vec_factor>(input_ptr + inner_it);

      // Reduce to [0, π/2]
      // neg_mask=true where sin should be negated
      aie::mask<vec_factor> neg_mask;
      // x = aie::to_float<float>(x);
      x = mod_halfpi_vec(x, neg_mask);

      // Taylor: x - x³/3! + x⁵/5! - x^7/7!
      vec_t xtemp          = aie::mul(x, x);              // x^2
      vec_t x3             = aie::mul(xtemp, x);          // x^3
      vec_t x3div3fac      = aie::negmul(x3, cdiv3fac);   // -x^3/3!
      xtemp                = aie::mul(x3, x);             // x^4
      vec_t x5             = aie::mul(xtemp, x);          // x^5
      vec_t x5div5fac      = aie::mul(x5, cdiv5fac);      // +x^5/5!
      // xtemp                = aie::mul(x5, x);             // x^6
      // vec_t x7             = aie::mul(xtemp, x);          // x^7
      // vec_t x7div7fac      = aie::negmul(x7, cdiv7fac);   // -x^7/7!


      xtemp = aie::add(x3div3fac, x5div5fac);
      // xtemp = aie::add(xtemp, x7div7fac);

      // vec_t outv = aie::add(x, x3div3fac);
      vec_t outv = aie::add(x, xtemp);

      // Restore sign for quadrants where sin is negative
      outv = aie::select(outv, aie::neg(outv), neg_mask);

      // Store 16 bfloat16s
      aie::store_v(output_ptr + inner_it, outv);
    }
  }
  event1();
}

extern "C" {

// // Non-vectorized
// void sin_kernel_32x64(bfloat16 in_mat[32][64], bfloat16 out_mat[32][64]) {
//   constexpr int TILE_ROWS = 32;
//   constexpr int SEQ_COLS  = 64; 
//   constexpr int VEC_SIZE  = 32;
//   // constexpr int VEC_SIZE  = 8;

//   for (int r = 0; r < TILE_ROWS; ++r) {
//     bfloat16* __restrict in_row  = &in_mat[r][0];
//     bfloat16* __restrict out_row = &out_mat[r][0];

//     aie::vector<bfloat16, VEC_SIZE> v0 = aie::load_v<VEC_SIZE>(in_row);
//     aie::vector<bfloat16, VEC_SIZE> v1 = aie::load_v<VEC_SIZE>(in_row + VEC_SIZE);

//     for (int k = 0; k < VEC_SIZE; ++k) {
//       out_row[k]            = sin_lut2pi(v0[k]);
//       out_row[VEC_SIZE + k] = sin_lut2pi(v1[k]);
//     }
//   }
// }

// vector version sine
void sin_bfloat16(bfloat16 in_mat[32][64], bfloat16 out_mat[32][64]) {
  sin_kernel_32x64(in_mat, out_mat);
}

} // extern "C"