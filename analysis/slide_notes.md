# Slide Notes: NPU Dispatch Overhead Analysis

## Slide: Per-Call Time Breakdown

- Wall-clock per kernel call: **~28ms**
- Actual NPU compute (steady-state): **~0.16ms** — only **0.6%** of wall-clock
- **90% of overhead is XRT re-initialization**
  - `xrt::device()`, `xrt::xclbin()`, `register_xclbin()`, `xrt::hw_context()`, `xrt::kernel()`, buffer allocation
- Current allo version generates each kernel as a **standalone C++ test binary**
- It **spawns a new process and rebuilds XRT from scratch on every call**
- There is **no persistent host-side runtime** — no way to keep XRT context alive between calls
- Python file I/O and process fork are negligible (<1ms combined)

## Slide: Why Is XRT Setup So Expensive?

- Allo generates each kernel as a **standalone C++ test binary**
- Every `mod(A, B, C)` call **spawns a new process**
  - Serializes numpy arrays → disk files
  - Forks subprocess → C++ binary opens XRT device from scratch
  - Creates fresh HW context, kernel handle, buffer objects
  - After execution: writes results to disk → Python reads them back
- **Nothing persists between calls** — full teardown/rebuild every time
- This architecture was designed for **kernel testing**, not pipeline execution

## Slide: Cold vs Warm Kernel Calls

- First call in a new process: **~1,070 μs** (cold — AIE array configuration)
- Subsequent calls in same process: **~160 μs** (warm — 6.7x faster)
- Allo always pays the cold call cost because it creates a new process each time
- With warmup (profile mode, 50+ iterations): min = **140 μs**, avg = **161 μs**

## Slide: Full Pipeline Impact

- SmolVLA end-to-end requires **~21,783 kernel dispatches**
  - Preprocessing: 14,592 (67%) — conv loop over 768 embeddings × 3 channels
  - Connector: 3,328 (15%) — 192 K-chunks × 16 ops each
  - Vision block: 1,761 (8%) — 1,536 softmax tiles alone
  - Text encoder ×2: 1,576 (7%)
  - Action expert: 524 (2%)
- Current cost: 21,783 × 28ms = **~610 seconds**
- With persistent XRT runtime: 21,783 × 0.4ms = **~9 seconds** (68x speedup)
- With full kernel fusion (1 dispatch): **< 1 second**

## Slide: Overhead Is Constant Across Kernels

- Measured 6 kernels, 10 repeated trials each — overhead is flat on every call
- Tiny kernel (add 32×32, 1 tile): **~25ms overhead**, 0.5ms NPU
- Large kernel (GEMM 1024×768×768): **~37ms overhead**, 3.5ms NPU
- Fixed floor of **~24ms** regardless of kernel size
- Variable component: ~3-4ms per MB of DMA transfer data

## Slide: RoPE — Worst Overhead-to-Compute Ratio

- RoPE is decomposed into **11 sub-operations per head per tile**
  - copyL, copyR, copyL(sin), copyL(cos), 4× mul, sub, add, join
- Each sub-op: **~24ms dispatch** for **~0.4ms of NPU work**
- Text encoder Q (15 heads, 2 tiles): 2 × (4 shared + 15×11) = **338 dispatches**
- Total RoPE overhead per text encoder layer: **338 × 24ms = ~8 seconds**
- Actual compute: **338 × 0.16ms = 54ms**
- **99.3% overhead, 0.7% useful work**

## Slide: Not an NPU Problem

| Factor | Cause | Fix |
|---|---|---|
| 90% of overhead: XRT re-init every call | Allo subprocess-per-call architecture | Persistent XRT runtime |
| Cold first call (1ms vs 0.16ms steady) | New process = new HW context | Keep context alive |
| 21,783 dispatches | Fine-grained op-by-op execution | Kernel fusion / graph compilation |
| File-based data passing | test.cpp reads/writes .data files | In-memory buffer passing |

- The NPU itself is fast: **~160μs per kernel** in steady state
- XRT setup is the bottleneck: **~25ms per call** (unavoidable with current allo architecture)
- This is fundamentally a **software runtime problem**, not a hardware problem

## Slide: Path Forward

1. **Persistent XRT runtime** (68x speedup, ~9s total)
   - Keep device/context/kernel/buffers alive across calls
   - Pass data via shared memory instead of disk files
   - Eliminates 25ms XRT setup per call

2. **Fuse RoPE into single kernel** (10x fewer RoPE dispatches)
   - One C++ kernel replaces 11 sub-ops per head

3. **Batch preprocessing** (1000x fewer conv dispatches)
   - Replace 768×3×4 individual conv calls with tiled kernel or reshape+GEMM

4. **Full pipeline fusion** (1 dispatch, <1s total)
   - Dataflow entire model on AIE array
   - Tile outputs stream directly to next tile
   - Single host round-trip for input/output
