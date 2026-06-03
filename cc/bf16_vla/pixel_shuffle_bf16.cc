// pixel_shuffle_bf16.cc — Pixel shuffle tile: [16][768] → [4][3072]
//
// Each output row r concatenates 4 consecutive input rows r*4+0..r*4+3.
// Called with mapping=[16]: one AIE core produces 4 output rows from 16 input rows.
//
// SRAM: in[16][768]=24KB + out[4][3072]=24KB = 48KB < 64KB ✓

#include <aie_api/aie.hpp>

extern "C" {

void pixel_shuffle_bf16(bfloat16 in[16][768], bfloat16 out[4][3072]) {
    constexpr int OUT_ROWS = 4;
    constexpr int CHUNKS   = 4;    // input rows per output row
    constexpr int COLS     = 768;
    constexpr int VEC      = 32;   // bf16 vector width on AIE2

    for (int r = 0; r < OUT_ROWS; ++r) {
        for (int chunk = 0; chunk < CHUNKS; ++chunk) {
            bfloat16* __restrict src = &in[r * CHUNKS + chunk][0];
            bfloat16* __restrict dst = &out[r][chunk * COLS];
            for (int v = 0; v < COLS / VEC; ++v) {
                aie::vector<bfloat16, VEC> vec = aie::load_v<VEC>(src + v * VEC);
                aie::store_v(dst + v * VEC, vec);
            }
        }
    }
}

} // extern "C"
