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

// // Inlined LUT
// static const float SIN_LUT[SIN_LUT_SIZE] = {
//   0.00000000f, 0.02454123f, 0.04906767f, 0.07356456f, 0.09801714f, 0.12241068f, 0.14673047f, 0.17096189f, 
//   0.19509032f, 0.21910124f, 0.24298018f, 0.26671276f, 0.29028468f, 0.31368174f, 0.33688985f, 0.35989504f, 
//   0.38268343f, 0.40524131f, 0.42755509f, 0.44961133f, 0.47139674f, 0.49289819f, 0.51410274f, 0.53499762f, 
//   0.55557023f, 0.57580819f, 0.59569930f, 0.61523159f, 0.63439328f, 0.65317284f, 0.67155895f, 0.68954054f, 
//   0.70710678f, 0.72424708f, 0.74095113f, 0.75720885f, 0.77301045f, 0.78834643f, 0.80320753f, 0.81758481f, 
//   0.83146961f, 0.84485357f, 0.85772861f, 0.87008699f, 0.88192126f, 0.89322430f, 0.90398929f, 0.91420976f, 
//   0.92387953f, 0.93299280f, 0.94154407f, 0.94952818f, 0.95694034f, 0.96377607f, 0.97003125f, 0.97570213f, 
//   0.98078528f, 0.98527764f, 0.98917651f, 0.99247953f, 0.99518473f, 0.99729046f, 0.99879546f, 0.99969882f, 
//   1.00000000f, 0.99969882f, 0.99879546f, 0.99729046f, 0.99518473f, 0.99247953f, 0.98917651f, 0.98527764f, 
//   0.98078528f, 0.97570213f, 0.97003125f, 0.96377607f, 0.95694034f, 0.94952818f, 0.94154407f, 0.93299280f, 
//   0.92387953f, 0.91420976f, 0.90398929f, 0.89322430f, 0.88192126f, 0.87008699f, 0.85772861f, 0.84485357f, 
//   0.83146961f, 0.81758481f, 0.80320753f, 0.78834643f, 0.77301045f, 0.75720885f, 0.74095113f, 0.72424708f, 
//   0.70710678f, 0.68954054f, 0.67155895f, 0.65317284f, 0.63439328f, 0.61523159f, 0.59569930f, 0.57580819f, 
//   0.55557023f, 0.53499762f, 0.51410274f, 0.49289819f, 0.47139674f, 0.44961133f, 0.42755509f, 0.40524131f, 
//   0.38268343f, 0.35989504f, 0.33688985f, 0.31368174f, 0.29028468f, 0.26671276f, 0.24298018f, 0.21910124f, 
//   0.19509032f, 0.17096189f, 0.14673047f, 0.12241068f, 0.09801714f, 0.07356456f, 0.04906767f, 0.02454123f, 
//   0.00000000f, -0.02454123f, -0.04906767f, -0.07356456f, -0.09801714f, -0.12241068f, -0.14673047f, -0.17096189f, 
//   -0.19509032f, -0.21910124f, -0.24298018f, -0.26671276f, -0.29028468f, -0.31368174f, -0.33688985f, -0.35989504f, 
//   -0.38268343f, -0.40524131f, -0.42755509f, -0.44961133f, -0.47139674f, -0.49289819f, -0.51410274f, -0.53499762f, 
//   -0.55557023f, -0.57580819f, -0.59569930f, -0.61523159f, -0.63439328f, -0.65317284f, -0.67155895f, -0.68954054f, 
//   -0.70710678f, -0.72424708f, -0.74095113f, -0.75720885f, -0.77301045f, -0.78834643f, -0.80320753f, -0.81758481f, 
//   -0.83146961f, -0.84485357f, -0.85772861f, -0.87008699f, -0.88192126f, -0.89322430f, -0.90398929f, -0.91420976f, 
//   -0.92387953f, -0.93299280f, -0.94154407f, -0.94952818f, -0.95694034f, -0.96377607f, -0.97003125f, -0.97570213f, 
//   -0.98078528f, -0.98527764f, -0.98917651f, -0.99247953f, -0.99518473f, -0.99729046f, -0.99879546f, -0.99969882f, 
//   -1.00000000f, -0.99969882f, -0.99879546f, -0.99729046f, -0.99518473f, -0.99247953f, -0.98917651f, -0.98527764f, 
//   -0.98078528f, -0.97570213f, -0.97003125f, -0.96377607f, -0.95694034f, -0.94952818f, -0.94154407f, -0.93299280f, 
//   -0.92387953f, -0.91420976f, -0.90398929f, -0.89322430f, -0.88192126f, -0.87008699f, -0.85772861f, -0.84485357f, 
//   -0.83146961f, -0.81758481f, -0.80320753f, -0.78834643f, -0.77301045f, -0.75720885f, -0.74095113f, -0.72424708f, 
//   -0.70710678f, -0.68954054f, -0.67155895f, -0.65317284f, -0.63439328f, -0.61523159f, -0.59569930f, -0.57580819f, 
//   -0.55557023f, -0.53499762f, -0.51410274f, -0.49289819f, -0.47139674f, -0.44961133f, -0.42755509f, -0.40524131f, 
//   -0.38268343f, -0.35989504f, -0.33688985f, -0.31368174f, -0.29028468f, -0.26671276f, -0.24298018f, -0.21910124f, 
//   -0.19509032f, -0.17096189f, -0.14673047f, -0.12241068f, -0.09801714f, -0.07356456f, -0.04906767f, -0.02454123f, 
//   0.00000000f,
// };

// // Fast range reduction to [0, 2π) without fmod/floor.
// static inline float mod2pi(float x) {
//   float q = x * INV_TWO_PI;
//   int n = (int)q;                 // trunc toward zero
//   float r = x - (float)n * TWO_PI_F;
//   if (r < 0.0f)  r += TWO_PI_F;
//   if (r >= TWO_PI_F) r -= TWO_PI_F;
//   return r;
// }

// static inline float sin_lut2pi(float x) {
//   float r = mod2pi(x);               // [0, 2π)
//   float idx_f = r / SIN_STEP;        // [0, SIN_LUT_SIZE-1]
//   int   i     = (int)idx_f;          // trunc
//   float frac  = idx_f - (float)i;
//   if (i >= SIN_LUT_SIZE - 1) { i = SIN_LUT_SIZE - 2; frac = 1.0f; }
//   float y0 = SIN_LUT[i];
//   float y1 = SIN_LUT[i + 1];
//   return y0 + frac * (y1 - y0);
// }

// static inline float cos_lut2pi(float x) {
//   return sin_lut2pi(x + HALF_PI_F);
// }

// // sine LUT for vectorized 
// template<int vec_factor>
// aie::vector<float, vec_factor> cos_lut2pi_vec(const aie::vector<float, vec_factor>& x) {
//   aie::vector<float, vec_factor> y;
//   #pragma unroll
//   // for (int lane = 0; lane < vec_factor; ++lane){
//   //   y[lane] = cos_lut2pi(x[lane]);
//   // }
//   for (int lane = 0; lane < vec_factor; lane += 8){
//     y[lane] = cos_lut2pi(x[lane]);
//     y[lane + 1] = cos_lut2pi(x[lane + 1]);
//     y[lane + 2] = cos_lut2pi(x[lane + 2]);
//     y[lane + 3] = cos_lut2pi(x[lane + 3]);
//     y[lane + 4] = cos_lut2pi(x[lane + 4]);
//     y[lane + 5] = cos_lut2pi(x[lane + 5]);
//     y[lane + 6] = cos_lut2pi(x[lane + 6]);
//     y[lane + 7] = cos_lut2pi(x[lane + 7]);
//   }
//   return y;
// }

// // vector version cosine
// template <typename T_in, typename T_out>
// static inline void cos_kernel_32x64(T_in in_mat[32][64], T_out out_mat[32][64]) {
//   constexpr int TILE_ROWS  = 32;
//   constexpr int COLS       = 64;
//   constexpr int vec_factor = 8; 

//   for (int r = 0; r < TILE_ROWS; ++r) {
//     T_in*  __restrict in_row  = &in_mat[r][0];
//     T_out* __restrict out_row = &out_mat[r][0];

//     for (int c = 0; c < COLS; c += 2 * vec_factor) {
//       // load 2 vectors = 2*vec_factor elements
//       aie::vector<T_in, vec_factor> v0_in = aie::load_v<vec_factor>(in_row + c);
//       aie::vector<T_in, vec_factor> v1_in = aie::load_v<vec_factor>(in_row + c + vec_factor);

//       // convert to float for LUT
//       aie::vector<float, vec_factor> v0 = aie::vector_cast<float>(v0_in);
//       aie::vector<float, vec_factor> v1 = aie::vector_cast<float>(v1_in);

//       // compute
//       aie::vector<float, vec_factor> y0_f = cos_lut2pi_vec<vec_factor>(v0);
//       aie::vector<float, vec_factor> y1_f = cos_lut2pi_vec<vec_factor>(v1);

//       // convert back
//       aie::vector<T_out, vec_factor> y0 = aie::vector_cast<T_out>(y0_f);
//       aie::vector<T_out, vec_factor> y1 = aie::vector_cast<T_out>(y1_f);

//       // store
//       aie::store_v(out_row + c, y0);
//       aie::store_v(out_row + c + vec_factor, y1);
//     }
//   }
// }

// // COSINE TAYLOR APPROX. ----------------------------------------------------------------
// // input restriction 0 < x < π (for accuracy) 
// template<int vec_factor>
// static inline aie::vector<float, vec_factor> mod_halfpi_vec(
//   const aie::vector<float, vec_factor>& x,
//   aie::mask<vec_factor>&neg_mask) {
//   using vec_t = aie::vector<float, vec_factor>;
  
//   const vec_t inv_two_pi = aie::broadcast<float, vec_factor>(INV_TWO_PI);
//   const vec_t two_pi     = aie::broadcast<float, vec_factor>(TWO_PI_F);
//   const vec_t pi         = aie::broadcast<float, vec_factor>(PI_F);
//   const vec_t half_pi    = aie::broadcast<float, vec_factor>(HALF_PI_F);
//   const vec_t zero       = aie::broadcast<float, vec_factor>(0.0f);
//   const vec_t one        = aie::broadcast<float, vec_factor>(1.0f);
  
//   vec_t q = aie::mul(x, inv_two_pi);
//   auto n_i = aie::to_fixed(q, 0);
//   auto n_f  = aie::to_float(n_i, 0);
//   q = aie::mul(n_f, two_pi);
//   vec_t r = aie::sub(x, q);

//   auto m_lt0   = aie::lt(r, zero);
//   r = aie::select(r, aie::add(r, two_pi), m_lt0);
//   auto m_ge2pi = aie::ge(r, two_pi);
//   r = aie::select(r, aie::sub(r, two_pi), m_ge2pi);    // r ∈ [0, 2π)

//   auto m_ge_pi = aie::ge(r, pi);
//   r = aie::select(r, aie::sub(two_pi, r), m_ge_pi);        // r ∈ [0, π)

//   neg_mask = aie::gt(r, half_pi);
//   r = aie::select(r, aie::sub(pi, r), neg_mask);        // r ∈ [0, π/2]
  
//   return r;
// }

// float div_fac(int num) {
//   float product = 1; 
//   while (num > 0) {
//     product *= num;
//     num --;
//   }
//   return 1/product;
// }

// // low-degree Taylor polynomial
// void cos_kernel_32x64(float input_x[32][64], float output_x[32][64]) {
//   constexpr int SEQ_TILE = 32;
//   constexpr int FEATURE_DIM_TILE = 64;
//   constexpr int vec_factor = 16;
//   // constexpr int vec_factor = 32;
//   using vec_t = aie::vector<float, vec_factor>;

//   // constants
//   const vec_t c1       = aie::broadcast<float, vec_factor>(1.0f);
//   const vec_t cdiv2fac = aie::broadcast<float, vec_factor>(1.0f * div_fac(2));
//   const vec_t cdiv4fac = aie::broadcast<float, vec_factor>(1.0f * div_fac(4));

//   for (int s = 0; s < SEQ_TILE; ++s) {
//     float *__restrict input_ptr = &input_x[s][0];
//     float *__restrict output_ptr = &output_x[s][0];

//     for (int inner_it = 0; inner_it < FEATURE_DIM_TILE; inner_it += vec_factor) {
//       // Load 16 floats
//       vec_t x = aie::load_v<vec_factor>(input_ptr + inner_it);
//       aie::mask<vec_factor> neg_mask;
//       x = mod_halfpi_vec<vec_factor>(x, neg_mask);  

//       // Taylor: 1 - x^2/2! + x^4/4!
//       vec_t x2             = aie::mul(x, x);              // x^2
//       vec_t x2div2fac      = aie::negmul(x2, cdiv2fac);   // -x^2/2!
//       vec_t xtemp          = aie::mul(x2, x);             // x^3
//       vec_t x4             = aie::mul(xtemp, x);          // x^4
//       vec_t x4div4fac      = aie::mul(x4, cdiv4fac);      // +x^4/4!

//       xtemp = aie::add(x2div2fac, x4div4fac);

//       // vec_t outv = aie::add(c1, x2div2fac);
//       vec_t outv = aie::add(c1, xtemp);
//       outv = aie::select(outv, aie::neg(outv), neg_mask);

//       // Store 16 floats
//       aie::store_v(output_ptr + inner_it, outv);
//     }
//   }
// }

// SHIFT SINE BY PI/2 --------------------------------------------------------
// input restriction 0 < x < π/2 (for accuracy) 
template<unsigned vec_factor>
static inline aie::vector<float, vec_factor> reduce_halfpi_float(
  aie::vector<float, vec_factor> f,
  aie::mask<vec_factor>& neg_mask)
{
  using fvec_t = aie::vector<float, vec_factor>;

  const fvec_t inv_two_pi = aie::broadcast<float, vec_factor>(INV_TWO_PI);
  const fvec_t two_pi     = aie::broadcast<float, vec_factor>(TWO_PI_F);
  const fvec_t pi         = aie::broadcast<float, vec_factor>(PI_F);
  const fvec_t half_pi    = aie::broadcast<float, vec_factor>(HALF_PI_F);
  const fvec_t zero       = aie::broadcast<float, vec_factor>(0.0f);

  // cos(x) = sin(x + pi/2)
  f = aie::add(f, half_pi);

  fvec_t q = aie::mul(f, inv_two_pi);
  auto n_i = aie::to_fixed(q, 0);
  auto n_f = aie::to_float(n_i, 0);
  q        = aie::negmul(n_f, two_pi);
  fvec_t r = aie::add(f, q);

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

// low-degree Taylor polynomial
void cos_kernel_32x64(bfloat16 input_x[32][64], bfloat16 output_x[32][64]) {
  event0();

  constexpr int SEQ_TILE         = 32;
  constexpr int FEATURE_DIM_TILE = 64;
  constexpr int vec_factor       = 16;

  using vec_t  = aie::vector<bfloat16, vec_factor>;
  using fvec_t = aie::vector<float, vec_factor>;

  const fvec_t cdiv3fac = aie::broadcast<float, vec_factor>(1.0f * div_fac(3));
  const fvec_t cdiv5fac = aie::broadcast<float, vec_factor>(1.0f * div_fac(5));
  const fvec_t cdiv7fac = aie::broadcast<float, vec_factor>(1.0f * div_fac(7));

  for (int s = 0; s < SEQ_TILE; ++s) {
    bfloat16 *__restrict input_ptr  = &input_x[s][0];
    bfloat16 *__restrict output_ptr = &output_x[s][0];

    for (int inner_it = 0; inner_it < FEATURE_DIM_TILE; inner_it += vec_factor) {
      vec_t x = aie::load_v<vec_factor>(input_ptr + inner_it);

      // bfloat16 -> float
      fvec_t f;
      for (int i = 0; i < vec_factor; i++) {
        f[i] = (float)x[i];
      }

      // Reduce directly in float
      aie::mask<vec_factor> neg_mask;
      f = reduce_halfpi_float<vec_factor>(f, neg_mask);

      // Taylor on reduced value
      fvec_t x2        = aie::mul(f, f);
      fvec_t x3        = aie::mul(x2, f);
      fvec_t x3div3fac = aie::negmul(x3, cdiv3fac);

      fvec_t x4        = aie::mul(x3, f);
      fvec_t x5        = aie::mul(x4, f);
      fvec_t x5div5fac = aie::mul(x5, cdiv5fac);

      fvec_t x6        = aie::mul(x5, f);
      fvec_t x7        = aie::mul(x6, f);
      fvec_t x7div7fac = aie::negmul(x7, cdiv7fac);

      fvec_t outv = aie::add(f, aie::add(x3div3fac, x5div5fac));
      outv = aie::add(outv, x7div7fac);

      outv = aie::select(outv, aie::neg(outv), neg_mask);

      // float -> bfloat16
      vec_t r;
      for (int i = 0; i < vec_factor; i++) {
        r[i] = (bfloat16)outv[i];
      }

      aie::store_v(output_ptr + inner_it, r);
    }
  }

  event1();
}

extern "C" {

// // Compute cosine elementwise for a 32×64 tile
// void cos_float32(float in_mat[32][64], float out_mat[32][64]) {
//   constexpr int TILE_ROWS = 32;
//   constexpr int SEQ_COLS  = 64;
//   constexpr int VEC_SIZE  = 32;

//   for (int r = 0; r < TILE_ROWS; ++r) {
//     float* __restrict in_row  = &in_mat[r][0];
//     float* __restrict out_row = &out_mat[r][0];

//     aie::vector<float, VEC_SIZE> v0 = aie::load_v<VEC_SIZE>(in_row);
//     aie::vector<float, VEC_SIZE> v1 = aie::load_v<VEC_SIZE>(in_row + VEC_SIZE);

//     // Elementwise cos using the same LUT (phase shift)
//     for (int k = 0; k < VEC_SIZE; ++k) {
//       out_row[k]            = cos_lut2pi(v0[k]);
//       out_row[VEC_SIZE + k] = cos_lut2pi(v1[k]);
//     }
//   }
// }

// vector version cosine
void cos_bfloat16(bfloat16 in_mat[32][64], bfloat16 out_mat[32][64]) {
  // cos_kernel_32x64<float, float>(in_mat, out_mat);
    cos_kernel_32x64(in_mat, out_mat);
}

} // extern "C"