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
        float output[4][4])
    {
        constexpr int IN_H = 64;
        constexpr int IN_W = 64;
        constexpr int PATCH_SIZE = 16;
        constexpr int vec_factor = 16;
        constexpr int NUM_PATCHES_H = IN_H / PATCH_SIZE;           // 4
        constexpr int NUM_PATCHES_W = IN_W / PATCH_SIZE;           // 4
        using vec_t = aie::vector<float, vec_factor>;

        event0();

        for (int ph = 0; ph < NUM_PATCHES_H; ++ph)
        {
            for (int pw = 0; pw < NUM_PATCHES_W; ++pw)
            {
                float *__restrict output_ptr = &output[ph][pw];
                float *input_ptr0 = &input[ph * PATCH_SIZE + 0][pw * PATCH_SIZE];
                float *kernel_ptr0 = &kernel[0][0];
                vec_t x0 = aie::load_v<vec_factor>(input_ptr0);
                vec_t k0 = aie::load_v<vec_factor>(kernel_ptr0);
                aie::accum<accfloat, vec_factor> acc = aie::mul(x0, k0);


                for (int kh = 1; kh < PATCH_SIZE; ++kh)
                {
                    float *__restrict input_ptr = &input[ph * PATCH_SIZE + kh][pw * PATCH_SIZE];
                    float *__restrict kernel_ptr = &kernel[kh][0];
                    vec_t x = aie::load_v<vec_factor>(input_ptr);
                    vec_t k = aie::load_v<vec_factor>(kernel_ptr);
                    acc = aie::mac(acc, x, k); // acc += x * k elementwise
                }

                vec_t sum_vec = acc.to_vector<float>();
                float result = aie::reduce_add(sum_vec);
                *output_ptr = result; 
            }
        }

        event1();
    }
        
} // extern "C"