# NPU Kernel Programming Guide

This guide explains how to write kernels for AMD Phoenix NPU using the Allo compiler framework, and how to optimize them with unified binaries.

## Overview: Three Levels of Programming

```
Level 1: Single-tile kernels (C++ with AIE API)
         └─ cc/bf16/silu_128_bf16.cc

Level 2: Allo compiler (Python mapping to multiple tiles)
         └─ vla/text_encoder_bf16/silu.py → silu.prj/build/final.xclbin

Level 3: Unified binary (Claude-generated C++ combining multiple kernels)
         └─ vla/text_encoder_bf16/unified.prj/build/text_encoder
```

---

## Level 1: Single-Tile Kernel (C++ with AIE API)

### Example: SiLU Activation

**File:** `cc/bf16_vla/silu_128_bf16.cc`

This implements SiLU (x * sigmoid(x)) for a 4×128 tile of bfloat16 data:

```cpp
/*
 * SiLU bf16 for per-core tile [4][128]
 * (SmolVLA action expert: FFN_HID=2048, tiled 2048/16=128 per core)
 */

#include <aie_api/aie.hpp>
#include <stdint.h>

void silu_bfloat16_128(bfloat16 input_x[4][256], bfloat16 output_x[4][256]) {
  constexpr int SEQ_TILE         = 4;      // 4 rows
  constexpr int FEATURE_DIM_TILE = 128;    // 128 features per core
  constexpr int vec_factor       = 32;     // 32-element vectors

  using vec_t = aie::vector<bfloat16, vec_factor>;

  // Precomputed sigmoid polynomial coefficients
  const vec_t one = aie::broadcast<bfloat16, vec_factor>(1.0f);
  
  // Main computation loop
  for (int i = 0; i < SEQ_TILE; i++) {
    for (int j = 0; j < FEATURE_DIM_TILE; j += vec_factor) {
      vec_t x = aie::load_v<vec_factor>(input_x[i] + j);
      
      // Compute sigmoid(x) ≈ polynomial approximation
      vec_t sig = sigmoid_approx(x);
      
      // SiLU = x * sigmoid(x)
      vec_t result = aie::mul(x, sig);
      
      aie::store_v(output_x[i] + j, result);
    }
  }
  
  event1();
}
```

**Key concepts:**
- `aie::vector<T, N>` — vectorized operations (32 bfloat16 elements at once)
- `aie::load_v` / `aie::store_v` — bulk data movement
- Sigmoid via Padé polynomial approximation (fast, accurate on NPU)
- Fixed tile size [4][128] optimized for SmolVLA action expert

**Test:** `kernel_testing/silu/test_silu_bf16_new.py`

---

## Level 2: Allo Compiler (Multi-Tile Mapping)

### Example: LayerNorm with Allo

LayerNorm needs to:
1. Compute mean and variance across the feature dimension
2. Normalize each element: (x - mean) / sqrt(var + eps)
3. Scale and shift: y = gamma * normalized + beta

Allo handles:
- **Tiling strategy**: Split 960-dim into chunks across 16 cores
- **Communication**: Point-to-point (P2P) for partial reductions
- **Synchronization**: Proper locks for multi-core execution

**File:** `vla/text_encoder_bf16/rms_norm.py`

```python
import allo
from allo.ir.types import bfloat16 as bf16

def rms_norm_allo(N: int, D: int, eps: float):
    """
    RMS Norm: y = (x / rms(x)) * gamma + beta
    
    N: sequence length (1 for text encoder state)
    D: feature dimension (960 for SmolVLA)
    eps: small epsilon for numerical stability
    """
    
    # Define inputs
    x = allo.placeholder((N, D), dtype=bf16, name="x")
    gamma = allo.placeholder((D,), dtype=bf16, name="gamma")
    beta = allo.placeholder((D,), dtype=bf16, name="beta")
    
    def rms_norm_kernel(x, gamma, beta):
        # Step 1: Compute RMS across feature dimension
        rms = allo.output((N, 1), dtype=bf16, name="rms")
        for i in allo.grid(N):
            sum_sq = 0.0
            for j in allo.grid(D):
                sum_sq += x[i, j] * x[i, j]
            rms[i, 0] = allo.sqrt(sum_sq / D + eps)
        
        # Step 2: Normalize and scale
        output = allo.output((N, D), dtype=bf16, name="output")
        for i in allo.grid(N):
            for j in allo.grid(D):
                normalized = x[i, j] / rms[i, 0]
                output[i, j] = normalized * gamma[j] + beta[j]
        
        return output
    
    # Schedule: map to 16 AIE cores
    s = allo.customize(rms_norm_kernel, [x, gamma, beta])
    
    # Tile strategy
    s.tile(rms, 1, 64)           # Split RMS computation
    s.tile(output, 1, 64)        # Split output computation
    
    # Vectorization
    s.vectorize(output, [2])
    
    # Compile and return
    return s.build()
```

**Key Allo concepts:**
- `allo.grid()` — parallel loop dimensions
- `allo.tile()` — split computation across cores
- `allo.vectorize()` — enable SIMD on core
- Point-to-point communication handled automatically for reductions

**Generated output:**
- `vla/text_encoder_bf16/rms_norm.prj/` — Allo-generated project
- `rms_norm.prj/build/final.xclbin` — Compiled binary
- `rms_norm.prj/top.mlir` — Intermediate representation

**Test:** `kernel_testing/layer_norm/test_layer_bf16_new.py`

---

## Level 3: Unified Binary (Claude-Generated C++)

### The Problem with Allo Alone

When composing multiple kernels with Allo:

```
vla.py:
  x = vision_output  # [1024, 768]
  x = rms_norm(x)    # ← Launch kernel → return result
  x = attention(x)   # ← Launch kernel → return result
  x = mlp(x)         # ← Launch kernel → return result
```

**Overhead per kernel:**
- XRT context switch
- Allocate buffers in NPU memory
- DMA transfer input data
- Launch AIE executable
- DMA transfer output data
- Deallocate buffers

**Result:** 20+ kernels = 20 × context switch overhead = ~17 seconds total

### Solution: Unified.prj (Single Executable)

Claude generates a single C++ executable that:
1. Calls all kernels in sequence
2. Keeps intermediate data in NPU memory (no DMA)
3. Single XRT launch for entire pipeline

**Example:** `vla/text_encoder_bf16/unified.prj/test.cpp`

```cpp
// This executable runs the entire text encoder in one go
// Instead of 20+ separate kernel calls, it's 1 call

#include "text_encoder_llama_12layer.h"

int main() {
    // Input: [1, 48, 960] token embeddings
    bfloat16* input = allocate_bf16(1 * 48 * 960);
    
    // ===== Layer 1 =====
    rms_norm_kernel(input, w1_gamma, w1_beta);  // In-place normalize
    attn_kernel(input, w1_q, w1_k, w1_v);       // Compute attention
    mlp_kernel(input, w1_ffn_up, w1_ffn_down);  // FFN
    
    // ===== Layer 2 =====
    rms_norm_kernel(input, w2_gamma, w2_beta);
    attn_kernel(input, w2_q, w2_k, w2_v);
    mlp_kernel(input, w2_ffn_up, w2_ffn_down);
    
    // ... (repeat for 12 layers)
    
    // Output: [1, 48, 960]
    return 0;
}
```

**Benefits:**
- ✅ Single XRT launch (vs 24 launches for 12 layers × 2 ops/layer)
- ✅ Zero DMA between kernels (data stays in NPU memory)
- ✅ Predictable, composable performance

---

## How to Generate Unified.prj

### User Workflow

**Step 1:** User writes individual kernel sources in `cc/bf16/`

```
cc/bf16_vla/
├── rms_norm_960_bf16.cc
├── silu_128_bf16.cc
├── gemm_attn_q.cc
└── ...
```

**Step 2:** User writes Allo code referencing these kernels

```
vla/text_encoder_bf16/
├── rms_norm.py              ← Allo code that uses rms_norm_960_bf16.cc
├── silu.py                  ← Allo code that uses silu_128_bf16.cc
└── ...
```

**Step 3:** Claude generates unified.prj by combining them

```cpp
// unified.prj/test.cpp (generated by Claude)

// Include all kernel headers
#include "../../cc/bf16_vla/rms_norm_960_bf16.cc"
#include "../../cc/bf16_vla/silu_128_bf16.cc"

int main() {
    // Call kernels in sequence
    rms_norm_bfloat16_960(x, gamma, beta);
    silu_bfloat16_128(x, x);  // In-place
    // ... more kernels ...
    
    return 0;
}
```

**Step 4:** Compile unified.prj

```bash
cd vla/text_encoder_bf16/unified.prj
mkdir -p build && cd build
cmake .. && make -j4
```

**Result:** Single `text_encoder` binary that runs entire component

### Claude's Role

When you want to create or update a unified.prj, use this prompt:

```
You are generating a unified C++ executable for the text encoder.

Given:
- Kernel files in cc/bf16_vla/: [list of kernel files]
- Allo Python code in vla/text_encoder_bf16/: [list of .py files with kernel specs]

Generate unified.prj/test.cpp that:
1. Includes all kernel headers
2. Calls kernels in the correct order (RMSNorm → Attention → MLP, repeat 12L)
3. Manages data flow (input → layer 1 → layer 2 → ... → layer 12 → output)
4. Uses in-place computation where possible to minimize memory usage
5. Compiles without errors

Ensure the executable matches the behavior of the Allo individual kernels.
```

---

## Allo Limitations & Why We Use Claude

### Limitation 1: Per-Kernel Overhead

**Allo generates:**
```
for each kernel in pipeline:
  └─ Call AIE.run(kernel_executable)
     └─ XRT context switch
     └─ Buffer allocation/deallocation
     └─ DMA in/out
```

**Result:** Linear slowdown with kernel count (20 kernels = 20× overhead)

### Limitation 2: No Host Code Generation

Allo does NOT generate:
- C++ main() function
- Buffer management
- Kernel composition logic
- Data flow orchestration

**Why:** Allo focuses on AIE kernel generation, not full-system integration.

**Why Claude helps:** Claude writes the "glue" code that ties kernels together efficiently.

### Limitation 3: Multi-Kernel Coordination

Allo can optimize individual kernels but struggles with:
- Keeping intermediate data in NPU memory between kernels
- Overlapping computation and DMA
- Managing locks for multi-core point-to-point communication

**Claude solution:** Generates unified.prj that handles these automatically.

---

## Summary: The Complete Flow

```
cc/bf16_vla/silu_128_bf16.cc         ← You write
  ↓
vla/text_encoder_bf16/silu.py        ← You write Allo code
  ↓ (Allo compiler runs)
silu.prj/build/final.xclbin          ← Allo generates (compiled binary)
  
cc/bf16_vla/rms_norm_960_bf16.cc     ← You write
  ↓
vla/text_encoder_bf16/rms_norm.py    ← You write Allo code
  ↓ (Allo compiler runs)
rms_norm.prj/build/final.xclbin      ← Allo generates
  
[... more kernels ...]

  ↓ (Claude generates unified.prj)
  
unified.prj/test.cpp                 ← Claude writes (combines all kernels)
  ↓ (CMake + make)
unified.prj/build/text_encoder       ← Single executable (fast!)
  ↓
vla_standalone.py calls unified.prj/build/text_encoder
  ↓
Result: 6.16 seconds end-to-end (vs 23 seconds with Allo alone)
```

---

## Resources

- **Allo documentation**: https://github.com/heterogeneous-computing-lab/allo
- **AIE API reference**: https://github.com/Xilinx/AI-Engine-Intrinsics
- **Kernel examples**: `cc/bf16/`, `cc/bf16_vla/`
- **Allo Python examples**: `vla/*/[component]_bf16.py`
- **Unified.prj examples**: `vla/*/unified.prj/`

---

## Next Steps

1. **Write a kernel**: Follow the SiLU example in `cc/bf16_vla/silu_128_bf16.cc`
2. **Test it individually**: Create a test in `kernel_testing/my_kernel/test_my_kernel.py`
3. **Use in Allo**: Write `vla/my_component/my_kernel.py` that uses your kernel
4. **Generate unified.prj**: Use Claude to combine multiple kernels
5. **Benchmark end-to-end**: Run `vla_standalone.py` to see the speedup
