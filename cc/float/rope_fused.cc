// rope_fused.cc — Single-kernel RoPE rotation for one head tile [64][64]
//
// Fuses: split x into halves, multiply by sin/cos, subtract/add, join
// Sin/cos are precomputed on host and packed into sin_cos[64][64]:
//   sin_cos[r][0..31]  = sin(radians)
//   sin_cos[r][32..63] = cos(radians)
//
// Inputs:  x[64][64]       — one head tile (SEQ x HEAD_DIM)
//          sin_cos[64][64]  — packed sin (left) and cos (right)
// Output:  out[64][64]      — rotated result

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

void rope_fused_float32(float x[32][64],
                        float sin_cos[32][64],
                        float out[32][64]) {
  constexpr int SEQ  = 32;
  constexpr int HALF = 32;

  for (int r = 0; r < SEQ; ++r) {
    float* __restrict x_row  = &x[r][0];
    float* __restrict sc_row = &sin_cos[r][0];
    float* __restrict o_row  = &out[r][0];

    // Load x left and right halves
    aie::vector<float, HALF> v_xL = aie::load_v<HALF>(x_row);
    aie::vector<float, HALF> v_xR = aie::load_v<HALF>(x_row + HALF);

    // Load sin and cos from packed buffer
    aie::vector<float, HALF> v_sin = aie::load_v<HALF>(sc_row);
    aie::vector<float, HALF> v_cos = aie::load_v<HALF>(sc_row + HALF);

    // out_L = xL*cos - xR*sin
    aie::vector<float, HALF> v_outL = aie::sub(aie::mul(v_xL, v_cos),
                                                aie::mul(v_xR, v_sin));
    // out_R = xR*cos + xL*sin
    aie::vector<float, HALF> v_outR = aie::add(aie::mul(v_xR, v_cos),
                                                aie::mul(v_xL, v_sin));

    aie::store_v(o_row,        v_outL);
    aie::store_v(o_row + HALF, v_outR);
  }
}

} // extern "C"
