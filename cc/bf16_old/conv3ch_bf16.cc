#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C"
{
    // Fused 3-channel conv2d. Called per AIE core after 4×4 spatial sharding.
    // A_packed[192][64] = 3 channels stacked: rows 0..63 = ch0, 64..127 = ch1, 128..191 = ch2.
    // kernel3[48][16] = 3 stacked [16][16] kernels (replicated to all cores).
    // output[4][4] = channel-accumulated result for this spatial subtile.
    void conv3ch(
        bfloat16 A_packed[192][64],
        bfloat16 kernel3[48][16],
        bfloat16 output[4][4])
    {
        constexpr int CH_H        = 64;    // rows per channel per core (64 = IN_H / 4)
        constexpr int IN_W        = 64;    // cols per core (64 = IN_W / 4)
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
                aie::accum<accfloat, vec_factor> acc;
                acc = aie::zeros<accfloat, vec_factor>();

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

} // extern "C"
