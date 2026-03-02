#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C"
{
    void conv(
        float input[64][64],
        float kernel[16][16],
        float output[4][4]
    )
    {
        // Flatten for easier indexing
        float *in = reinterpret_cast<float *>(input);
        float *ker = reinterpret_cast<float *>(kernel);
        float *out = reinterpret_cast<float *>(output);

        constexpr int IN_H = 64;
        constexpr int IN_W = 64;
        constexpr int PATCH_SIZE = 16;
        constexpr int NUM_PATCHES_H = IN_H / PATCH_SIZE;           // 4
        constexpr int NUM_PATCHES_W = IN_W / PATCH_SIZE;           // 4
        constexpr int NUM_PATCHES = NUM_PATCHES_H * NUM_PATCHES_W; // 16

        event0();

        for (int ph = 0; ph < NUM_PATCHES_H; ph++)
        {
            for (int pw = 0; pw < NUM_PATCHES_W; pw++)
            {

                int patch_idx = ph * NUM_PATCHES_W + pw;

                float sum = 0.0f;
                for (int kh = 0; kh < PATCH_SIZE; kh++)
                {
                    for (int kw = 0; kw < PATCH_SIZE; kw++)
                    {

                        // Input pixel position
                        int in_h = ph * PATCH_SIZE + kh;
                        int in_w = pw * PATCH_SIZE + kw;

                        // Read input value: input[in_h][in_w]
                        float input_val = in[in_h * IN_W + in_w];

                        // Read kernel value: kernel[kh][kw]
                        float kernel_val = ker[kh * PATCH_SIZE + kw];

                        sum += input_val * kernel_val;
                    }
                }
                out[patch_idx] = sum;
            }
        }

        event1();
    }

} // extern "C"