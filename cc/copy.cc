/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {

// Interleaved reshape kernel: [4, 768] -> [4, 768] (simple copy per core)
// Global: [16, 768] -> [4, 3072] with mapping=[4]
// Output row i = [A[i,:], A[i+4,:], A[i+8,:], A[i+12,:]]
void copy_float32(float input[4][768], float output[4][768]) {
    constexpr int IN_ROWS = 4;
    constexpr int IN_COLS = 768;
    for (int i = 0; i < IN_ROWS; i++) {
        for (int j = 0; j < IN_COLS; j++) {
            output[i][j] = input[i][j];
        }
    }
}

} // extern "C"