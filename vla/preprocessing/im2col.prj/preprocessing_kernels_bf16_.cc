#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C"
{

// =============================================================================
// conv3ch — fused 3-channel conv2d patch embedding (pipelined version)
//
// Called per AIE core after 4×4 spatial sharding of the [256,256] input.
// A_packed[192][64]: 3 channels interleaved per row group.
//   rows [ch*64 : (ch+1)*64] = channel ch's [64,64] spatial subtile.
// kernel3[48][16]: 3 stacked [16,16] kernels (replicated to all cores).
// output[4][4]: channel-accumulated result for this subtile (4×4 patches).
//
// Vectorization: inner kh loop loads a 16-element row from each channel and
// MAC-accumulates with the corresponding kernel row.
// =============================================================================
void conv3ch(
    bfloat16 A_packed[192][64],
    bfloat16 kernel3[48][16],
    bfloat16 output[4][4])
{
    constexpr int CH_H        = 64;
    constexpr int IN_W        = 64;
    constexpr int PATCH_SIZE  = 16;
    constexpr int vec_factor  = 16;
    constexpr int NUM_PATCHES = CH_H / PATCH_SIZE;   // 4
    constexpr int CHANNELS    = 3;
    using vec_t = aie::vector<bfloat16, vec_factor>;

    event0();

    for (int ph = 0; ph < NUM_PATCHES; ++ph)
    {
        for (int pw = 0; pw < NUM_PATCHES; ++pw)
        {
            aie::accum<accfloat, vec_factor> acc = aie::zeros<accfloat, vec_factor>();

            for (int ch = 0; ch < CHANNELS; ++ch)
            {
                int ch_row_base = ch * CH_H + ph * PATCH_SIZE;
                int k_row_base  = ch * PATCH_SIZE;

                for (int kh = 0; kh < PATCH_SIZE; ++kh)
                {
                    vec_t x = aie::load_v<vec_factor>(
                        &A_packed[ch_row_base + kh][pw * PATCH_SIZE]);
                    vec_t k = aie::load_v<vec_factor>(
                        &kernel3[k_row_base + kh][0]);
                    acc = aie::mac(acc, x, k);
                }
            }

            aie::vector<float, vec_factor> sum_vec = acc.to_vector<float>();
            output[ph][pw] = (bfloat16) aie::reduce_add(sum_vec);
        }
    }

    event1();
}

// =============================================================================
// im2col — image-to-column rearrangement (fused im2col+GEMM version)
//
// Converts one quarter-row of image patches to column-major patch format,
// ready for the GEMM patch-embedding step.
//
// Inputs (per-core, single core mapping=[1]):
//   img[48][128]  — image strip quarter: [C*KH=48, PIX_QT=128]
//                   img[c*KH + ky, pw*KW + kx] = pixel at channel c, row ky,
//                   patch pw, kernel-col kx.
//
// Output:
//   pat[8][768]   — patches quarter: [PW_QT=8, K_FULL=768]
//                   pat[pw, c*KH*KW + ky*KW + kx] for c in 0..2, ky in 0..15,
//                   kx in 0..15.
//
// Vectorization: inner kx dimension (size 16) maps to a contiguous 16-element
// vector load from img and a contiguous store into pat — zero scalar fallback.
// Loop trip count: CHANNELS*PW_QT*KERNEL_H = 3*8*16 = 384 vector ops.
// =============================================================================
void im2col(
    bfloat16 img[48][128],
    bfloat16 pat[8][768])
{
    constexpr int PW_QT    = 8;
    constexpr int CHANNELS = 3;
    constexpr int KERNEL_H = 16;
    constexpr int KERNEL_W = 16;
    constexpr int K_FULL   = CHANNELS * KERNEL_H * KERNEL_W;  // 768
    constexpr int vec_factor = KERNEL_W;                       // 16
    using vec_t = aie::vector<bfloat16, vec_factor>;

    event0();

    for (int pw = 0; pw < PW_QT; ++pw)
    {
        for (int c = 0; c < CHANNELS; ++c)
        {
            for (int ky = 0; ky < KERNEL_H; ++ky)
            {
                // Load 16 consecutive pixels: img[c*KH + ky, pw*KW .. pw*KW+15]
                vec_t v = aie::load_v<vec_factor>(
                    &img[c * KERNEL_H + ky][pw * KERNEL_W]);

                // Store to contiguous position in pat: pat[pw, c*KH*KW + ky*KW .. +KW]
                aie::store_v(
                    &pat[pw][c * KERNEL_H * KERNEL_W + ky * KERNEL_W],
                    v);
            }
        }
    }

    event1();
}

} // extern "C"
