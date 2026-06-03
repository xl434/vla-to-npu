/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * flash_attn_tile_bf16.cc — Flash attention score tile kernel for AMD AIE2.
 *
 * Computes: Q_tile[32][64] @ K_tile_T[64][64] → scores[32][64] * (1/sqrt(64))
 *
 * Allo ExternalModule convention:
 *   input_idx=[0, 2], output_idx=[1]
 *   slot3=Q_tile[32][64], slot4=scores[32][64], slot5=K_tile_T[64][64]
 *
 * This is a CORRECTNESS PROTOTYPE — scalar (non-vectorized).
 * Not expected to be faster than the current tiled GEMM approach due to
 * NPU stateless dispatch constraints (one hw_context at a time).
 */

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#define NOCPP

static constexpr int Q_TILE    = 32;
static constexpr int KV_TILE   = 64;
static constexpr int HEAD_DIM  = 64;

// scale = 1 / sqrt(64) = 0.125
static constexpr float ATTN_SCALE = 0.125f;

extern "C" {

/*
 * flash_attn_score_bf16 — single-tile attention score.
 *
 * Computes scores[i][j] = sum_k Q_tile[i][k] * K_tile_T[k][j], then scales.
 * Q_tile:    [Q_TILE=32,  HEAD_DIM=64] bf16 input
 * scores:    [Q_TILE=32,  KV_TILE=64]  bf16 output
 * K_tile_T:  [HEAD_DIM=64, KV_TILE=64] bf16 input (K transposed, so K^T)
 *
 * Slot order matches Allo ExternalModule(input_idx=[0,2], output_idx=[1]):
 *   slot3 = Q_tile, slot4 = scores (out), slot5 = K_tile_T
 */
void flash_attn_score_bf16(
    bfloat16 Q_tile[Q_TILE][HEAD_DIM],
    bfloat16 scores[Q_TILE][KV_TILE],
    bfloat16 K_tile_T[HEAD_DIM][KV_TILE])
{
    for (int i = 0; i < Q_TILE; ++i) {
        bfloat16 *__restrict q_row = &Q_tile[i][0];
        for (int j = 0; j < KV_TILE; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < HEAD_DIM; ++k) {
                acc += (float)q_row[k] * (float)K_tile_T[k][j];
            }
            scores[i][j] = (bfloat16)(acc * ATTN_SCALE);
        }
    }
}

} // extern "C"
