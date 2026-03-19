#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C"
{
    void add(
        bfloat16 input_x[64][64],
        bfloat16 input_y[64][64],
        bfloat16 output[64][64])
    {
        constexpr int vec_factor = 64;
        using vec_t = aie::vector<bfloat16, vec_factor>;

        event0();
        
        for (int i=0; i<64; ++i) {
            bfloat16 *__restrict x_ptr = &input_x[i][0];
            bfloat16 *__restrict y_ptr = &input_y[i][0];
            bfloat16 *__restrict out_ptr = &output[i][0];
            vec_t x = aie::load_v<vec_factor>(x_ptr);
            vec_t y = aie::load_v<vec_factor>(y_ptr);
            vec_t out = aie::add(x, y);
            aie::store_v(out_ptr, out);
        }

        event1();
    }
        
} // extern "C"