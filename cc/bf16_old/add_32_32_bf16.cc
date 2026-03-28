#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C"
{
    void add(
        bfloat16 input_x[32][32],
        bfloat16 input_y[32][32],
        bfloat16 output[32][32])
    {
        constexpr int vec_factor = 32;
        using vec_t = aie::vector<bfloat16, vec_factor>;
        using float_vec_t = aie::vector<float, vec_factor>;

        event0();
        
        for (int i=0; i<32; ++i) {
            bfloat16 *__restrict x_ptr = &input_x[i][0];
            bfloat16 *__restrict y_ptr = &input_y[i][0];
            bfloat16 *__restrict out_ptr = &output[i][0];
            vec_t x = aie::load_v<vec_factor>(x_ptr);
            vec_t y = aie::load_v<vec_factor>(y_ptr);
            
            float_vec_t x_f;
            float_vec_t y_f;

            for (int j = 0; j < vec_factor; j++) {
                x_f[j] = (float)x[j];
                y_f[j] = (float)y[j];
            }

            float_vec_t out_f = aie::add(x_f, y_f);

            vec_t out;
            for (int j = 0; j < vec_factor; j++) {
                out[j] = (bfloat16)out_f[j];
            }           

            aie::store_v(out_ptr, out);
        }

        event1();
    }
        
} // extern "C"