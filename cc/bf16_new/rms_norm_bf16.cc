
/*
* Copyright Allo authors. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
*/

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define NOCPP

#define EPS 1e-6f // epsilon

template <const int SEQ_LEN, const int HIDDEN>
void rms_norm_single_batch(bfloat16 *input_tensor, bfloat16 *weight,
                        bfloat16 *output_tensor) {
    constexpr int vec_factor = 32;
    using vec_t = aie::vector<bfloat16, vec_factor>;
    
    event0();
    for (int iter = 0; iter < SEQ_LEN; iter++) {
        bfloat16 *__restrict input_ptr = input_tensor;
        bfloat16 *__restrict weight_ptr = weight;
        bfloat16 *__restrict output_ptr = output_tensor;
        float square_sum = 0.0f;
        const int F = HIDDEN / vec_factor;
        for (int i = 0; i < F; i++) {
        vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
        input_ptr += vec_factor;
        vec_t square_vec = aie::mul(input_vec, input_vec);
        square_sum += aie::reduce_add(square_vec);
        }
        vec_t square_sum_vec =
            aie::broadcast<bfloat16, vec_factor>(square_sum / HIDDEN + EPS);
        vec_t rms = aie::invsqrt(square_sum_vec);
        input_ptr = input_tensor;
        for (int i = 0; i < F; i++) {
        vec_t input_vec = aie::load_v<vec_factor>(input_ptr);
        input_ptr += vec_factor;
        vec_t normed = aie::mul(input_vec, rms);
        vec_t weight_vec = aie::load_v<vec_factor>(weight_ptr);
        weight_ptr += vec_factor;
        vec_t result = aie::mul(normed, weight_vec);
        aie::store_v(output_ptr, result);
        output_ptr += vec_factor;
        }
        input_tensor += HIDDEN;
        output_tensor += HIDDEN;
    }
    event1();
}

extern "C" {

void rms_norm(bfloat16 A_in[4][768], bfloat16 B_in[768], bfloat16 C_out[4][768]) {
rms_norm_single_batch<4, 768>(&A_in[0][0], B_in, &C_out[0][0]);
}

} // extern "C"
