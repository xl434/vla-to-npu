# VLA NPU Performance Analysis: Why It's Slower Than CPU

Date: 2026-04-08

## 1. Kernel Dispatch Count

Every `*_mod(...)` call across the full pipeline:

| Component | NPU Dispatches | Notes |
|---|---|---|
| **Preprocessing** | **14,592** | 768 embd × (12 conv + 3 add + 4 copy) |
| **Vision Encoder** (1L) | **1,761** | 1,536 from softmax alone (12 heads × 128 batches) |
| **Connector** | **3,328** | 192 K-chunks × (1 GEMM + 15 adds) + 256 copies |
| **Text Encoder** (×2) | **1,576** | 456 RoPE calls + 240 softmax tiles + GEMMs |
| **Action Expert Self** (×1) | **284** | 228 RoPE calls dominate |
| **Action Expert Cross** (×1) | **240** | 169 RoPE + 45 attn per-head |
| **Postprocessing** | **2** | 1 norm + 1 GEMM |
| **TOTAL** | **~21,783** | |

## 2. Dispatch Overhead Is ~99% of Runtime

From existing measurements:
- Vision block: **Allo = 56.8s** vs **PyTorch = 0.017s** (3,341x slower)
- Text encoder: **Allo = ~21s** vs **PyTorch = ~7ms** (3,000x slower)

If the AIE kernel compute is ~100-170us per call (from kernel test results), then for the vision block:
- **Compute time**: 1,761 calls × 150us = **0.26s**
- **Actual time**: **56.8s**
- **Dispatch overhead**: (56.8 - 0.26) / 1,761 = **~32ms per dispatch**

This means **~99.5% of the runtime is dispatch overhead**, not compute.

For the full pipeline:
- 21,783 × 32ms = **~697s** in dispatch overhead alone
- vs PyTorch CPU total of a few seconds

## 3. Where the Dispatch Overhead Comes From

Each `*_mod(...)` call triggers this full sequence:

```
Python -> numpy slice/copy -> allo runtime -> XRT driver ->
  configure DMA buffer descriptors ->
  copy host->device (PCIe/shared mem) ->
  configure & trigger AIE tile(s) ->
  kernel executes (100-170us -- the FAST part) ->
  copy device->host ->
  return to Python
```

The ~32ms overhead breaks down roughly as:
- **XRT/driver setup**: ~10-15ms (PDI/XCLBIN context switch, BD programming)
- **DMA transfers**: ~5-10ms (even for small buffers, PCIe latency dominates)
- **Python/numpy overhead**: ~5-10ms (slicing, contiguity checks, GIL)

## 4. Root Causes

### 4a. Architectural Anti-Pattern: Fine-grained Op Dispatch

The code treats the NPU as a **library of individual ops** called from Python. Every GEMM, every norm tile, every RoPE sub-operation is a separate round-trip. This is like using a GPU by launching one CUDA kernel at a time with `cudaDeviceSynchronize()` after each.

### 4b. Extreme Per-Head Loops

RoPE alone generates **11 dispatches per head per tile**:
```python
copyL, copyR, copyL(sin), copyL(cos),  # 4 copies
mul, mul, sub,                          # yL computation
mul, mul, add,                          # yR computation
join                                    # recombine
```
For Q (15 heads, 2 tiles in text encoder): 2 × (4 + 15×11) = **338 dispatches** just for Q RoPE.

### 4c. K-Chunking Multiplies Dispatch Count

Large K dimensions can't fit in one GEMM call, so:
- Connector: K=12288, chunk=64 -> **192 GEMM calls** + 192×15 accumulate adds
- FFN down (text encoder): K=2560, chunk=320 -> **8 GEMM calls** per layer
- FFN down (action expert): K=2048, chunk=256 -> **8 GEMM calls** per layer

### 4d. Preprocessing Is The Worst Offender

```python
for i in range(768):         # each embedding dim
    for j in range(3):       # each channel
        for k in range(2):   # 2x2 spatial tiles
            for l in range(2):
                conv_mod(...)  # 768x3x4 = 9,216 conv dispatches!
```
**14,592 dispatches** for what PyTorch does in a single `nn.Conv2d` call.

## 5. Is It Allo-Specific or NPU-Fundamental?

| Factor | Allo Issue? | NPU Issue? |
|---|---|---|
| No kernel fusion / graph compilation | **Yes** -- allo dispatches one op at a time | No -- AIE supports dataflow chains |
| No on-device buffering between ops | **Yes** -- every result returns to host | No -- AIE tiles can stream to each other |
| Python loop dispatch | **Yes** -- could be C++ runtime | No |
| High per-dispatch fixed cost | Partially -- allo doesn't batch BDs | Partially -- XRT has inherent overhead |
| K-chunking | **Yes** -- GEMM module's data memory limit | Partially -- AIE tile local memory is small (32KB) |
| No XCLBIN reuse across calls | **Yes** -- each module is separate XCLBIN | No -- could be combined |

## 6. What Would Fix It

Ranked by impact:

1. **Fuse the entire pipeline into a single XCLBIN** (~1000x speedup potential)
   - Dataflow the whole model: tile outputs stream directly to next tile's input
   - Eliminate all host round-trips except initial input and final output
   - This is what allo's `df.region()` + `df.kernel()` with streams was designed for -- but currently each module is called separately from Python

2. **Fuse RoPE into a single kernel** (~10x fewer RoPE dispatches)
   - Replace 11 sub-ops per head with one fused C++ kernel
   - `rope_fused_bf16(Q_packed, sin_cos_table, Q_out)` -- 1 call vs 169

3. **Batch the preprocessing conv** (~1000x fewer conv dispatches)
   - Fuse 768x3x4 individual conv calls into one tiled conv kernel
   - Or: do patch embedding as a reshape + GEMM (equivalent to conv with stride=kernel_size)

4. **Increase GEMM K-chunk size** (~4-8x fewer GEMM calls)
   - If data memory allows K=256 or K=512 instead of K=64, the connector goes from 192 to 24 GEMM calls

5. **Batch softmax tiles** (~10x fewer softmax calls)
   - Process multiple tiles per dispatch instead of one 8-row tile at a time

## 7. Summary

```
Total NPU dispatch calls:          ~21,783
Estimated dispatch overhead:        ~32ms each
Dispatch overhead as % of runtime:  ~99.5%
Actual kernel compute time:         < 1% of total

Biggest offenders:
  Preprocessing:  14,592 dispatches (67%)
  Connector:       3,328 dispatches (15%)
  Vision Block:    1,761 dispatches  (8%)
  Text Encoder:    1,576 dispatches  (7%)
  Action Expert:     524 dispatches  (2%)
  Postprocessing:      2 dispatches  (<1%)
```

The NPU kernels themselves are **fast** (~100-170us). The problem is calling them **21,783 times** with ~32ms overhead each time. This is fundamentally a **dispatch overhead problem**, not a compute problem. The fix is kernel fusion and graph-level compilation, not faster individual kernels.

## 8. Measured Dispatch Overhead by Kernel Type

Empirical measurements from `vla/measure_dispatch.py` (30 iterations, 5 warmup per kernel):

```
Kernel                                          Min   Median     Mean      P95      Max
----------------------------------------------------------------------------------
add 32x32 (1 tile, 2KB)                       24.56    27.24    27.29    29.58    29.91
RMSNorm 16x960 (4 tiles, 30KB)                25.78    28.44    28.13    30.13    30.49
masked_softmax 8x128 (1 tile, 2KB)            24.23    26.12    26.01    27.82    28.39
SiLU 4x2560 (16 tiles, 20KB)                  27.66    31.90    31.71    34.94    37.69
RoPE radians 64x32 (1 tile, f32)              23.37    26.49    26.23    28.04    28.69
RoPE pack 64x32->64x64 (1 tile, f32)          23.44    26.24    26.09    28.43    30.80
RoPE sin 64x64 (2 tiles, f32)                 24.26    27.49    27.17    29.25    29.27
RoPE cos 64x64 (2 tiles, f32)                 23.62    26.52    26.29    28.27    28.80
RoPE copyL 64x64->64x32 (1 tile, f32)         23.59    25.86    25.73    26.81    27.32
RoPE mul32 64x32 (1 tile, f32)                24.37    26.07    26.42    30.27    30.71
RoPE join 2x64x32->64x64 (2 tiles, f32)       24.36    27.01    27.06    29.99    30.82
GEMM KV 128x320x960 (medium)                  30.12    32.33    32.47    33.79    35.17
GEMM Q 128x960x960 (large)                    33.55    37.57    37.24    39.28    39.31
GEMM FFN_up 128x2560x960 (xlarge)             41.86    45.28    45.07    47.77    47.91
GEMM exp_Q 32x960x768 (action)                30.02    34.13    33.88    36.19    36.51
GEMM vision 1024x768x768 (huge)               38.01    41.73    41.76    45.25    45.29
```

All times in milliseconds.

### Breakdown by Category

| Category | Avg Min (ms) | Range (ms) | Tiles | Data Size |
|---|---|---|---|---|
| **RoPE sub-ops** | 23.86 | 23.37 - 24.37 | 1-2 | 8-16 KB |
| **Element-wise** (norm, softmax, SiLU) | 25.56 | 24.23 - 27.66 | 1-16 | 2-30 KB |
| **GEMM** | 34.71 | 30.02 - 41.86 | many | 700KB - 4.5MB |

### Key Findings

1. **Fixed dispatch floor: ~23 ms.** Even a 2KB add on 1 tile costs 23ms. This is the
   irreducible XRT driver + XCLBIN context + BD setup + Python overhead.

2. **DMA transfer adds ~5-18 ms on top**, scaling with data size:
   - Tiny (2KB): ~0 ms extra -> 23 ms total
   - Medium (700KB): ~7 ms extra -> 30 ms total
   - Huge (4.5MB): ~15 ms extra -> 42 ms total

3. **Dispatch overhead = ~23ms fixed + ~3-4ms per MB of DMA.**

4. **For the full pipeline (21,783 calls):**
   - Fixed overhead alone: 21,783 x 23ms = **~501s** (unavoidable without fusion)
   - With DMA: adds another ~100-200s
   - Total: **~600-700s**

5. **RoPE is pure waste**: 11 sub-ops per head, each costing 23ms for ~microseconds of
   compute. That's **253ms of dispatch for ~10us of useful work** per head per tile.


Yes, exactly. The ~24ms overhead hits on every single call, even back-to-back calls to the same kernel with the same XCLBIN
   already loaded.

  Look at the data — the warmup calls (3) didn't help. The 20 measured iterations of the same kernel all pay the full cost:

  RoPE copyL (same kernel, 20 calls):
    wall times: all ~24-27ms each
    NPU times:  all ~0.4-0.7ms each

  This means it's not a one-time XCLBIN loading cost. Every call goes through:

  prepare args → sync buffers to device → launch → wait → sync back
           ~24ms of this                    ~0.5ms

  The XRT runtime doesn't cache or batch anything between calls. Each kernel(...) + run.wait() is a full round-trip even if
  nothing changed.

  This is why 21,783 calls × 24ms = ~523 seconds of pure overhead — even though the NPU does useful work for only ~33 seconds
   total (21,783 × 1.5ms).

  For comparison, CUDA kernel launch overhead is ~5-10 microseconds, not milliseconds — that's a 2,400x difference in
  dispatch cost. The XRT/amdxdna driver was designed for large, infrequent kernel launches (like running an entire DNN), not
  fine-grained op-by-op dispatch

## 9. Overhead Breakdown: Where the ~28ms Actually Goes

Measured by instrumenting each phase of `mod(A, B, C)` (RMSNorm 16x960, 10 trials):

```
Average per-call breakdown:
  Python file write (numpy → disk):       0.24 ms   (0.8%)
  Subprocess total:                      28.08 ms   (98.5%)
    ├── Process startup (fork/exec):      0.93 ms   (3.3%)
    ├── XRT setup + C++ file I/O:        25.67 ms   (90.1%)  ← THE BOTTLENECK
    │   (xrt::device, xrt::xclbin, register_xclbin,
    │    xrt::hw_context, xrt::kernel, xrt::bo × N,
    │    read input*.data, bo.sync TO_DEVICE)
    └── NPU (launch + compute):           1.51 ms   (5.3%)
  Python file read (disk → numpy):        0.16 ms   (0.6%)
  ──────────────────────────────────────────────────
  Total wall-clock:                      28.50 ms

Baselines:
  Empty subprocess (`true`):              0.93 ms
  Raw file I/O (30KB write+read):         0.06 ms
```

The overhead is **90% XRT setup** — creating device, loading xclbin, registering, creating
context, kernel handle, and buffer objects. Python file I/O and process spawn are negligible.

### Cold vs Warm NPU Calls

The "NPU time" reported in single-shot mode (~1ms) is inflated because it's the **first cold
call** in a fresh XRT context. Measured by running the same C++ binary in profile mode:

```
Single shot (1 call per process, cold):
  NPU execution time: ~1070 μs

Profile mode (same process, warmed up):
  warmup=0,  5 iters: avg=352 μs, min=164 μs  (first call is ~1ms, rest ~160μs)
  warmup=5,  5 iters: avg=221 μs, min=210 μs
  warmup=50, 100 iters: avg=161 μs, min=140 μs  ← steady state
```

The first kernel call in a process costs ~1ms (AIE array configuration). Subsequent calls
in the same context cost **~160μs**. Our measurements capture the cold first call because
allo spawns a new process every time.

## 10. Root Cause: Allo's Test Harness Architecture

Allo's AIE backend generates each kernel as a standalone `test.cpp` that is compiled into
an independent executable. There is no persistent host-side runtime. The dispatch path:

```
Every mod(A, B, C) call:
  Python: write A,B to disk files (input0.data, input1.data)
  Fork subprocess: ./build/top -x final.xclbin -i insts.txt -k MLIR_AIE
    C++: xrt::device(0)                ← open device
    C++: xrt::xclbin("final.xclbin")   ← parse xclbin from disk
    C++: device.register_xclbin()       ← register with device
    C++: xrt::hw_context()              ← create HW context
    C++: xrt::kernel()                  ← get kernel handle
    C++: xrt::bo() × N                 ← allocate buffer objects
    C++: read input*.data from disk     ← file I/O
    C++: bo.sync(TO_DEVICE)             ← DMA host → device
    C++: kernel(); run.wait()           ← ~160μs actual NPU work
    C++: bo.sync(FROM_DEVICE)           ← DMA device → host
    C++: write output*.data to disk     ← file I/O
    C++: exit()                         ← destroy everything
  Python: read output*.data → numpy arrays
```

This architecture was designed for **testing individual kernels**, not for running a pipeline.
Every call pays the full XRT initialization cost because nothing persists between calls.

### What a Persistent Runtime Would Cost

If allo kept the XRT context alive between calls (no subprocess, no re-initialization):

```
Once at startup (amortized):
  xrt::device, xrt::xclbin, register, hw_context, kernel, bo alloc

Per call:
  memcpy input → buffer:        ~0.01 ms
  bo.sync(TO_DEVICE):           ~0.10 ms  (for small data)
  kernel(); run.wait():         ~0.16 ms  (steady-state NPU)
  bo.sync(FROM_DEVICE):         ~0.10 ms
  memcpy buffer → output:       ~0.01 ms
  ───────────────────────────────────────
  Total per call:               ~0.4 ms   (vs 28ms now = 70x faster)

Full pipeline estimate:
  21,783 calls × 0.4ms = ~9s   (vs ~610s now = 68x faster)
```

This is **not an NPU limitation** — the NPU computes in ~160μs. It is an **allo runtime
architecture issue**: subprocess-per-call with full XRT teardown/rebuild each time.