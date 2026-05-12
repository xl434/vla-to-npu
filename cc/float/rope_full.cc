// rope_full.cc — Fully fused RoPE kernel
//
// Computes sin/cos from positions and inv_timescale internally, then applies
// the rotation in a single kernel.
//
// params[64] packs two 32-element arrays to stay within AIE's 3-port limit:
//   params[0..31]  = positions[r]   (float, one per sequence row)
//   params[32..63] = inv_timescale[k] (float, one per head-dim half)
//
// For each row r and half-dim k:
//   radians = positions[r] * inv_timescale[k]
//   sin_val = sinf(radians),  cos_val = cosf(radians)
//
// Inputs:  x[32][64]     — one head tile (SEQ x HEAD_DIM)
//          params[64]    — packed positions (0..31) and inv_timescale (32..63)
// Output:  out[32][64]   — rotated result

#include <aie_api/aie.hpp>
#include <math.h>
#include <stdint.h>
#define NOCPP

extern "C" {

void rope_full_float32(float x[32][64],
                       float params[64],
                       float out[32][64])
{
    constexpr int SEQ  = 32;
    constexpr int HALF = 32;  // HEAD_DIM / 2
    constexpr int VEC  = 32;

    // Hoisted outside the loop: avoids repeated stack allocation on AIE's
    // limited stack, and lets alignas(64) guarantee the alignment required
    // by aie::load_v<16> (16 floats × 4 bytes = 64-byte vector).
    // alignas(64) float sin_row[HALF];
    // alignas(64) float cos_row[HALF];
    float sin_row[HALF];
    float cos_row[HALF];

    for (int r = 0; r < SEQ; ++r) {
        float pos_r = params[r];
        float* __restrict x_row = &x[r][0];
        float* __restrict o_row = &out[r][0];

        // Compute sin and cos for this row
        for (int k = 0; k < HALF; ++k) {
            float rad   = pos_r * params[SEQ + k];
            sin_row[k]  = sin(rad);
            cos_row[k]  = cos(rad);
        }

        // Load xL [cols 0..31] and xR [cols 32..63]
        aie::vector<float, VEC> v_xL0 = aie::load_v<VEC>(x_row);
        // aie::vector<float, VEC> v_xL1 = aie::load_v<VEC>(x_row + VEC);
        aie::vector<float, VEC> v_xR0 = aie::load_v<VEC>(x_row + HALF);
        // aie::vector<float, VEC> v_xR1 = aie::load_v<VEC>(x_row + HALF + VEC);

        // Load computed sin [0..31] and cos [0..31]
        aie::vector<float, VEC> v_sin0 = aie::load_v<VEC>(sin_row);
        // aie::vector<float, VEC> v_sin1 = aie::load_v<VEC>(sin_row + VEC);
        aie::vector<float, VEC> v_cos0 = aie::load_v<VEC>(cos_row);
        // aie::vector<float, VEC> v_cos1 = aie::load_v<VEC>(cos_row + VEC);

        // outL = xL*cos - xR*sin
        aie::vector<float, VEC> v_outL0 = aie::sub(aie::mul(v_xL0, v_cos0), aie::mul(v_xR0, v_sin0));
        aie::store_v(o_row,              v_outL0);
        // v_outL0 = aie::sub(aie::mul(v_xL1, v_cos1), aie::mul(v_xR1, v_sin1));
        // aie::store_v(o_row + VEC,        v_outL0);
        
        // outR = xR*cos + xL*sin
        v_outL0 = aie::add(aie::mul(v_xR0, v_cos0), aie::mul(v_xL0, v_sin0));
        aie::store_v(o_row + HALF,       v_outL0);
        // v_outL0 = aie::add(aie::mul(v_xR1, v_cos1), aie::mul(v_xL1, v_sin1));
        // aie::store_v(o_row + HALF + VEC, v_outL0);

    }
}

} // extern "C"
