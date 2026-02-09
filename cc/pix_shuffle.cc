/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <aie_api/aie.hpp>
#define NOCPP

extern "C" {

void pix_shuffle(float in[1024][768], float out[64][12288]) {
    constexpr int DIM = 32;
    constexpr int SF = 4;
    
}

} // extern "C"