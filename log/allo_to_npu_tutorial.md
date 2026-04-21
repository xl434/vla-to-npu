# From Allo Kernel to NPU Execution — A Beginner Tutorial

Using `vla/gelu_bf16` as the running example throughout.

---

## The Factory Analogy

Before diving into files and commands, here is an analogy that makes the whole system easier to reason about. **Think of the NPU as a factory.**

- **The factory** has 16 workers arranged on a floor, plus a loading dock that connects to the outside world (your laptop's RAM).
- **Each worker** (compute tile) is a small, specialized processor that only knows how to do one thing: read a box of numbers from the incoming conveyor belt, apply a math function, and put the result on the outgoing belt.
- **The loading dock** (shim tiles, row 0) is the only part of the factory that can reach outside — it pulls raw input data from your laptop's RAM and ships finished output back.
- **The warehouse** (memory tiles, row 1) sits between the loading dock and the workers. It receives large incoming shipments and splits them into smaller packages, one per worker.
- **Conveyor belts** (object FIFOs) connect the loading dock → warehouse → workers → warehouse → loading dock. Data flows along these belts automatically, without any worker having to think about it.

Everything Allo does — the compilation pipeline, the generated files, the runtime driver — exists to set up and operate this factory.

---

## The NPU Hardware: What Are Tiles?

The AMD NPU (Ryzen AI / Phoenix) has a 2D array of processing elements called **AIE tiles**:

```
Row 5:  [tile 0,5] [tile 1,5] [tile 2,5] [tile 3,5]   ← compute tiles (workers)
Row 4:  [tile 0,4] [tile 1,4] [tile 2,4] [tile 3,4]   ← compute tiles (workers)
Row 3:  [tile 0,3] [tile 1,3] [tile 2,3] [tile 3,3]   ← compute tiles (workers)
Row 2:  [tile 0,2] [tile 1,2] [tile 2,2] [tile 3,2]   ← compute tiles (workers)
Row 1:  [mem  0,1] [mem  1,1] [mem  2,1] [mem  3,1]   ← memory tiles  (warehouse)
Row 0:  [shim 0,0] [shim 1,0] [shim 2,0] [shim 3,0]   ← shim tiles    (loading dock)
         col=0      col=1      col=2      col=3
```

- **Shim tiles (row 0)** — the loading dock. The only tiles that can read/write your laptop's DDR RAM via DMA. All data entering or leaving the NPU passes through here.
- **Memory tiles (row 1)** — the warehouse. Larger local SRAM (~512KB each) used as a staging buffer. They receive a big block from the shim, split it into per-worker slices, and fan it out to the compute tiles above.
- **Compute tiles (rows 2–5)** — the workers. Each runs a small C program (the `.elf`) in a tight loop. They have ~32KB of local memory and are the only tiles that do actual math.

For gelu_bf16, Allo used 16 compute tiles (all 4 columns × rows 2–5), each processing a `4×768` bfloat16 slice simultaneously.

---

## Big Picture: The Compilation Pipeline

```
Your Allo Python (.py)
      |
      | allo.build()
      v
original.mlir          ← "what each worker does" — no routing yet
      |
      | Allo/MLIR-AIE lowering passes
      v
top.mlir               ← "full factory floor plan" — tiles, belts, routing
      |
      | aiecc (AIE compiler toolchain)
      |-- compile gelu_bf16_.cc → core_X_Y.elf  (worker manuals)
      |-- package ELFs + routing → final.xclbin  (factory blueprint)
      |-- encode DMA sequence  → insts.txt        (shipping orders)
      v
gelu_bf16.prj/
  ├── gelu_bf16_.cc      ← C source of the compute kernel
  ├── core_0_2.elf  ...  ← compiled binary for each tile (one per worker)
  ├── final.xclbin       ← full NPU package (all workers + wiring)
  ├── insts.txt          ← DMA control program (run once per job)
  ├── input0.data        ← reference input  (raw binary)
  ├── output1.data       ← reference output (written after run)
  ├── test.cpp           ← auto-generated host driver (XRT)
  ├── CMakeLists.txt     ← build system for test.cpp
  └── build/
        ├── top          ← compiled host executable
        └── final.xclbin ← copy here for convenience
```

### Stage 1: Allo Python → `original.mlir`

When you call `allo.build()`, Allo reads your Python kernel and emits `original.mlir`. This is a **dataflow IR** — it describes *what* each tile should compute, but not yet which physical tile on the NPU hardware runs it, or how data moves between them.

**Factory analogy:** This is like writing a job description: "I need 16 workers, each of whom applies gelu to a `4×768` block of numbers." You have not yet assigned workers to positions on the floor, nor laid down any conveyor belts. You just know what the job is.

In the file, every tile gets its own function that calls the actual math:
```mlir
// 16 identical-looking functions — one per tile
func.func @core_0_2(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) {
    call @gelu_bf16(%arg0, %arg1)
}
func.func @core_0_3(%arg0: memref<4x768xbf16>, %arg1: memref<4x768xbf16>) {
    call @gelu_bf16(%arg0, %arg1)
}
// ... 14 more
```

The reason there are 16 copies is **data parallelism**: the full input is `64×768` bfloat16 values. Allo splits this into 16 independent slices of `4×768`, one per tile, so all 16 workers run simultaneously on different rows. There is no communication between tiles — each one works on its own chunk independently.

At this stage the IR has no mention of DMA, no FIFOs, no tile coordinates. That comes next.

### Stage 2: `original.mlir` → `top.mlir` (Allo/MLIR-AIE lowering)

Allo runs a series of MLIR transformation passes that lower the abstract job description into a complete physical floor plan. `top.mlir` is the result — it is the most detailed IR in the pipeline and contains everything needed to actually configure the NPU hardware.

**Factory analogy:** This is where you go from "I need 16 gelu workers" to a full architectural blueprint: which worker stands at which position on the floor, exactly where each conveyor belt runs, which belt carries input and which carries output, and how the warehouse splits the incoming shipment.

**Tile instantiation.** Abstract function names become concrete hardware coordinates:
```mlir
%shim_noc_tile_0_0 = aie.tile(0, 0)   // loading dock, column 0
%mem_tile_0_1      = aie.tile(0, 1)   // warehouse, column 0
%tile_0_2          = aie.tile(0, 2)   // worker at (col=0, row=2)
%tile_0_3          = aie.tile(0, 3)   // worker at (col=0, row=3)
// ... all 24 tiles declared
```

**Object FIFOs (conveyor belts).** Allo inserts named data channels between tiles. For column 0, the full routing chain is:
```
host DDR → shim(0,0) → mem_tile(0,1) ──┬→ tile(0,2)  [worker gets slice 0]
                                        ├→ tile(0,3)  [worker gets slice 1]
                                        ├→ tile(0,4)  [worker gets slice 2]
                                        └→ tile(0,5)  [worker gets slice 3]
tile(0,2) ──┐
tile(0,3) ──┤→ mem_tile(0,1) → shim(0,0) → host DDR  [results go back]
tile(0,4) ──┤
tile(0,5) ──┘
```

In `top.mlir` this is written as:
```mlir
// shim → mem tile (incoming big block from host)
aie.objectfifo @fifo_4(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2)
// mem tile → 4 workers (one slice each)
aie.objectfifo @fifo_0(%mem_tile_0_1, {%tile_0_2}, 2)
aie.objectfifo @fifo_1(%mem_tile_0_1, {%tile_0_3}, 2)
// ... fifo_2, fifo_3 for the other two workers
// fan-out link: mem tile splits fifo_4 into 4 slices at byte offsets:
aie.objectfifo.link [@fifo_4] -> [@fifo_0, @fifo_1, @fifo_2, @fifo_3]([] [0, 3072, 6144, 9216])
```

The offsets `[0, 3072, 6144, 9216]` tell the warehouse where each worker's slice starts in the incoming block. `3072 = 4 rows × 768 columns × 2 bytes per bfloat16`.

The `2` in each FIFO declaration means **double-buffered** (ping-pong). Two physical buffers are allocated per FIFO. While a worker computes on buffer A, the DMA can already fill buffer B with the next batch. When the worker finishes A and releases it, B is ready immediately — so the worker never has to wait for data. This is how the factory keeps all 16 workers busy without idle time.

**Tile core bodies.** Each worker's actual runtime behavior is also written out in `top.mlir`:
```mlir
%core_0_2 = aie.core(%tile_0_2) {
  scf.for %i = 0 to MAX_INT {             // worker loops forever
    %in  = aie.objectfifo.acquire @fifo_0(Consume, 1)   // wait for input belt
    %out = aie.objectfifo.acquire @fifo_20(Produce, 1)  // wait for output belt
    func.call @gelu_bf16(%in[0], %out[0])               // do the math
    aie.objectfifo.release @fifo_0(Consume, 1)          // belt: I'm done with input
    aie.objectfifo.release @fifo_20(Produce, 1)         // belt: output is ready
  }
} {link_with = "external0.o"}
```

The worker runs an **infinite loop**, waiting for data to appear on its input belt, computing, then signaling the output belt. It never stops running; the DMA manages when data arrives and when it is collected. `link_with = "external0.o"` tells the compiler to link the tile's program with the compiled `gelu_bf16_.cc` object file that contains the actual math.

### Stage 3: `top.mlir` → ELFs + xclbin + insts.txt (aiecc)

`top.mlir` is handed to `aiecc` — the AMD AIE compiler toolchain, built on a customized LLVM. It produces three things in parallel:

**Compiling tile kernels → ELFs.**
The C kernel file `gelu_bf16_.cc` is compiled using a modified Clang that targets the **AIE ISA** — a custom VLIW processor architecture inside the NPU tiles. This is completely different from x86 or ARM. The resulting `.elf` files (`core_0_2.elf`, `core_0_3.elf`, etc.) are bare-metal programs that can only run inside their specific tile. You can see the compiler identity embedded in the binary: `clang version 19.0.0 (Xilinx/llvm-aie)`.

*Factory analogy: the ELF is each worker's printed instruction manual. It only makes sense to that one type of worker; a regular person (x86 CPU) cannot read it.*

**Packaging → final.xclbin.**
All 16 ELFs plus the routing configuration from `top.mlir` (which tiles are used, how the FIFOs are wired, which switch boxes route data where) are bundled into a single `final.xclbin`. This is the complete factory blueprint. When `test.cpp` calls `device.register_xclbin(xclbin)`, XRT reads this file, loads each ELF onto the correct tile, and programs all the routing switches.

*Factory analogy: the xclbin is the full factory blueprint — all worker manuals plus the entire wiring diagram — delivered in one package. You hand it to the factory foreman (XRT) once at startup.*

**Generating insts.txt.**
The FIFO descriptions in `top.mlir` are also compiled into a binary sequence of 32-bit DMA control words saved to `insts.txt`. This is the **per-job shipping order**. Every time you call `kernel(...)` in `test.cpp`, this sequence is sent to the NPU's DMA controller, which then:
1. Fetches input from host DDR → shim → warehouse → workers' input belts
2. Waits for all workers to finish
3. Collects output from workers' output belts → warehouse → shim → host DDR

*Factory analogy: the xclbin sets up how the factory is built and wired. insts.txt says "go" — move this specific batch of data through the factory right now. The xclbin is loaded once; insts.txt runs on every single job.*

**Generating test.cpp and reference data.**
Allo's Python also runs the kernel on the CPU using numpy/PyTorch and saves the input array as `input0.data` (raw binary). It auto-generates `test.cpp` with the exact buffer sizes hardcoded — `49152 * sizeof(bfloat16_t)` comes from `16 tiles × 4 rows × 768 features`. `CMakeLists.txt` is generated alongside so the host program builds with a standard `cmake + make`.

---

## The Three Key Files: ELF vs xclbin vs insts.txt

This is the most important distinction to internalize:

| File | Loaded | When | Factory analogy |
|------|--------|------|-----------------|
| `.elf` (inside xclbin) | Once at startup | `device.register_xclbin()` | Worker's instruction manual |
| `final.xclbin` | Once at startup | `device.register_xclbin()` | Full factory blueprint (all manuals + wiring) |
| `insts.txt` | Every run | `kernel(opcode, bo_instr, ...)` | Shipping order for one batch of data |

**The `.elf`** is each worker's program. It only knows one thing: wait for data on the input belt, compute gelu, push to output belt, repeat. It has no idea about DMA, host memory, or other workers. It is compiled for the AIE VLIW ISA — your CPU cannot execute it.

**The `final.xclbin`** is the factory setup package. It contains all ELFs plus the hardware routing config. You hand it to XRT once, XRT programs the NPU, and the workers are ready and waiting. After this, the factory is "wired up" but idle — no data is flowing yet.

**The `insts.txt`** is the per-job trigger. It does not contain any math — it is purely a sequence of DMA commands: "copy N bytes from host address X into shim tile buffer, wait for output, copy M bytes from output buffer back to host address Y." Sending it to the NPU via `kernel(...)` is what actually makes data flow and computation happen. The xclbin is loaded once; insts.txt runs every time you push a new batch through.

---

## File-by-File Reference

### `original.mlir` — abstract dataflow IR

One `func.func @core_X_Y` per tile, each calling the math function. No routing, no hardware addresses.

### `top.mlir` — full hardware IR

The complete factory floor plan:
- `aie.tile(col, row)` — declares every tile used
- `aie.objectfifo` — declares every conveyor belt
- `aie.objectfifo.link` — wires FIFOs together through the warehouse with byte offsets
- `aie.core(%tile)` — the infinite-loop program for each compute tile
- `{link_with = "external0.o"}` — tells the compiler to link the tile program against the kernel object

### `gelu_bf16_.cc` — the compute kernel (C, for AIE cores)

Standard C with AIE intrinsics. Compiled against the AIE ISA by Xilinx LLVM. This is the actual math that runs on each tile.

### `core_X_Y.elf` — compiled AIE tile binary

Bare-metal program for one tile. Cannot run on your CPU. One `.elf` per compute tile, named by (column, row) coordinates.

### `final.xclbin` — the NPU firmware package

Contains all tile ELFs + routing config. Loaded once at startup by XRT. After loading, all tiles are programmed and waiting.

### `insts.txt` — DMA control program (binary, not text)

Sent to the NPU's DMA controller on every kernel launch. Orchestrates all data movement for one job. Despite the `.txt` extension it is raw binary, not human-readable.

### `input0.data` / `output1.data` — reference I/O (raw binary)

`input0.data` is generated by Allo's Python build step (CPU reference run). `test.cpp` reads it and sends it to the NPU. After the NPU finishes, results are written to `output1.data`. By default `--verify 1` compares the two to check correctness.

---

## `test.cpp` — The Host Driver (XRT)

**XRT** (Xilinx Runtime) is the driver library that lets your CPU program talk to the NPU. `test.cpp` is auto-generated by Allo. Here is what it does, step by step:

### Step 1: Load the instruction sequence
```cpp
std::vector<uint32_t> instr_v =
    test_utils::load_instr_binary(vm["instr"].as<std::string>());
```
Reads `insts.txt` as raw binary into a `vector<uint32_t>`. This is the shipping order that will be sent to the DMA controller on each launch.

### Step 2: Open the NPU and load the xclbin
```cpp
auto device = xrt::device(0);                               // open NPU device
auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());  // parse final.xclbin
device.register_xclbin(xclbin);                             // program the NPU
xrt::hw_context context(device, xclbin.get_uuid());         // create run context
auto kernel = xrt::kernel(context, "MLIR_AIE");             // get kernel handle
```
After `register_xclbin`, all 16 workers are loaded with their ELFs and the conveyor belts are wired. The factory is ready.

### Step 3: Allocate buffer objects (BOs)
```cpp
// Instruction buffer — shared between CPU and NPU
auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                         XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));

// Input buffer — lives in host-accessible DDR
auto bo_in0 = xrt::bo(device, 49152 * sizeof(bfloat16_t),
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));

// Output buffer
auto bo_out1 = xrt::bo(device, 49152 * sizeof(bfloat16_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
```

A **buffer object (BO)** is a region of memory that both the CPU and NPU can access. `49152 = 16 tiles × 4 rows × 768 features`. The `kernel.group_id(N)` argument selects which memory bank to use; the numbers match the argument slots in the kernel call signature: `kernel(opcode, instr, instr_size, input, output, trace)`.

### Step 4: Fill input and push to device
```cpp
ifile0.read(reinterpret_cast<char*>(vec0.data()), 49152 * sizeof(bfloat16_t));
memcpy(bufIn0, srcVec0.data(), ...);

bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);   // push instructions to NPU
bo_in0.sync(XCL_BO_SYNC_BO_TO_DEVICE);     // push input data to NPU
```
`sync(...TO_DEVICE)` flushes the CPU's cache so the NPU DMA sees the latest data in DDR.

### Step 5: Launch and wait
```cpp
unsigned int opcode = 3;
auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in0, bo_out1, bo_trace);
ert_cmd_state r = run.wait();   // blocks until NPU signals done
```

This is the "go" signal. The DMA controller receives `insts.txt`, then:
1. Streams `bo_in0` from DDR → shim tiles → warehouse → workers' input belts
2. All 16 workers simultaneously apply gelu to their `4×768` slice
3. Results flow back: output belts → warehouse → shim tiles → `bo_out1` in DDR
4. NPU signals completion → `run.wait()` returns

The printed `NPU execution time: Xus` is the wall-clock time from launch to return.

### Step 6: Pull results back
```cpp
bo_out1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);   // invalidate CPU cache, read fresh DDR
std::bfloat16_t *bufOut1 = bo_out1.map<std::bfloat16_t *>();
ofile1.write(reinterpret_cast<const char*>(bufOut1), 49152 * sizeof(bfloat16_t));
```
`sync(...FROM_DEVICE)` invalidates the CPU cache so the read sees the NPU-written values, not a stale cached version.

---

## The Run Command

```bash
cd /home/xl434/vla-to-npu/vla/gelu_bf16.prj
build/top -x build/final.xclbin -i insts.txt -k MLIR_AIE
```

| Argument | What it is |
|----------|-----------|
| `-x build/final.xclbin` | Factory blueprint: loads all workers and wires up belts |
| `-i insts.txt` | Shipping order: tells DMA where to move data this run |
| `-k MLIR_AIE` | Kernel name to look up inside the xclbin |

Must be run from inside the `.prj/` directory — `test.cpp` opens `input0.data` with a hardcoded relative path.

### Optional flags
```bash
build/top -x build/final.xclbin -i insts.txt -k MLIR_AIE \
  -v 1          # verbose: prints tile names, buffer sizes, etc.
  --verify 0    # skip comparing output against reference
  -p 1          # profiling: run 100 iterations, report avg + min time
  --warmup 10   # warmup iterations before profiling
```

---

## How to Modify test.cpp

### Change input data
The input is loaded from `input0.data` around line 103. Replace with your own:
```cpp
// Instead of reading from file:
for (int i = 0; i < 49152; i++) {
    bufIn0[i] = (std::bfloat16_t)(your_data[i]);
}
// Still call:
bo_in0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
```

### Read the output into your own buffer
After `bo_out1.sync(XCL_BO_SYNC_BO_FROM_DEVICE)` around line 175:
```cpp
std::bfloat16_t *bufOut1 = bo_out1.map<std::bfloat16_t *>();
for (int i = 0; i < 49152; i++) {
    float val = (float)bufOut1[i];
    // use val
}
```

### Chain two kernels (output of kernel 1 → input of kernel 2)
```cpp
// After kernel 1 finishes and results are in bufOut1...
memcpy(bufIn0_k2, bufOut1, 49152 * sizeof(std::bfloat16_t));
bo_in0_k2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
auto run2 = kernel2(opcode, bo_instr2, instr_v2.size(), bo_in0_k2, bo_out1_k2, ...);
run2.wait();
```

### Rebuild after editing
```bash
cd /home/xl434/vla-to-npu/vla/gelu_bf16.prj/build
make
```

---

## Summary of All Files

| File | Generated by | Purpose |
|------|-------------|---------|
| `original.mlir` | Allo Python | Abstract dataflow: what each tile computes, no routing |
| `top.mlir` | Allo lowering | Full hardware IR: tile grid, FIFO belts, DMA routing |
| `gelu_bf16_.cc` | Allo | C source for the compute kernel (runs on AIE tiles) |
| `core_X_Y.elf` | aiecc (AIE Clang) | Compiled AIE binary for tile at (col=X, row=Y) |
| `final.xclbin` | aiecc | NPU firmware package: all ELFs + routing config |
| `insts.txt` | aiecc | Binary DMA control program, sent on every kernel launch |
| `input0.data` | Allo Python | Reference input (raw bfloat16 binary) |
| `output1.data` | test.cpp (after run) | NPU output (raw bfloat16 binary) |
| `test.cpp` | Allo Python | XRT host driver (auto-generated) |
| `CMakeLists.txt` | Allo Python | Build system for test.cpp |
| `build/top` | CMake/g++ | Compiled host executable |
