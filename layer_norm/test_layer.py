import os
import time
import numpy as np
import torch
import allo
import allo.dataflow as df
from allo.ir.types import float32, Stream
from allo.memory import Layout
from allo.backend.aie import ExternalModule

# ----------------------------
# Config
# ----------------------------
S = Layout.Shard
R = Layout.Replicate
Ty = float32
SEQ = 64
EMBD = 768

RTOL = 1e-2
ATOL = 1e-3

KERNEL_LIB_PATH = "../cc/float/"

def _mismatch_stats(actual: np.ndarray, expected: np.ndarray, rtol: float, atol: float):
    """
    Return (mismatch_pct, mismatch_count, total_count)
    using the same per-element rule as numpy.allclose/assert_allclose.
    """
    diff = np.abs(actual - expected)
    tol = atol + rtol * np.abs(expected)
    mismatch_mask = diff > tol

    total = mismatch_mask.size
    mismatches = int(np.count_nonzero(mismatch_mask))
    mismatch_pct = 100.0 * mismatches / total if total else 0.0
    return mismatch_pct, mismatches, total

# External module: layer norm (no bias inside kernel; bias added separately)
norm = ExternalModule(
    top="layer_norm",
    impl_path=KERNEL_LIB_PATH + "layer_norm.cc",
    input_idx=[0, 1],
    output_idx=[2],
)

NORM_P0 = 4
NORM_SEQ_TILE = 16
NORM_TILE = NORM_SEQ_TILE // NORM_P0

norm_io_layout = [S(0), R]
norm_arg_layout = [R]

@df.region()
def layer_norm_kernel(
    input_x: Ty[NORM_SEQ_TILE, EMBD],
    weight: Ty[EMBD],
    bias: Ty[EMBD],
    output_x: Ty[NORM_SEQ_TILE, EMBD],
):
    pipe: Stream[Ty[NORM_TILE, EMBD], 1][NORM_P0]

    @df.kernel(mapping=[NORM_P0], args=[input_x, weight])
    def norm_no_bias(
        local_input_x: Ty[NORM_SEQ_TILE, EMBD] @ norm_io_layout,
        local_weight: Ty[EMBD] @ norm_arg_layout,
    ):
        pi = df.get_pid()
        tmp: Ty[NORM_TILE, EMBD] = 0
        norm(local_input_x, local_weight, tmp)
        pipe[pi].put(tmp)

    @df.kernel(mapping=[NORM_P0], args=[bias, output_x])
    def norm_add_bias(
        local_bias: Ty[EMBD] @ norm_arg_layout,
        local_output_x: Ty[NORM_SEQ_TILE, EMBD] @ norm_io_layout,
    ):
        pi = df.get_pid()
        data = pipe[pi].get()
        local_output_x[:, :] = allo.add(data, local_bias)

# Build once
layer_norm_mod = df.build(
    layer_norm_kernel,
    target="aie",
    project="norm.prj",
    profile=True,
)

def layernorm_full_seq(
    input_x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    layer_norm_fn: torch.nn.Module,
):
    assert input_x.shape == (SEQ, EMBD)
    assert weight.shape == (EMBD,)
    assert bias.shape == (EMBD,)

    num_tiles = SEQ // NORM_SEQ_TILE
    out = np.empty((SEQ, EMBD), dtype=np.float32)

    npu_tile_times_us = []
    cpu_tile_times_us = []

    for i in range(num_tiles):
        tile_in = input_x[i * NORM_SEQ_TILE : (i + 1) * NORM_SEQ_TILE, :]
        tile_out = out[i * NORM_SEQ_TILE : (i + 1) * NORM_SEQ_TILE, :]

        # NPU tile call timing (host-side wall clock)
        t0 = time.perf_counter()
        layer_norm_mod(tile_in, weight, bias, tile_out)
        t1 = time.perf_counter()
        npu_tile_time_us = (t1 - t0) * 1_000_000
        npu_tile_times_us.append(npu_tile_time_us)

        # CPU tile timing
        with torch.no_grad():
            t2 = time.perf_counter()
            tile_in_numpy_cpu = tile_in.copy()                                      # input data prep
            cpu_out = layer_norm_fn(torch.from_numpy(tile_in_numpy_cpu))            # compute
            _ = cpu_out.cpu().numpy()                                               # output retrieval
            t3 = time.perf_counter()
        cpu_tile_time_us = (t3 - t2) * 1_000_000
        cpu_tile_times_us.append(cpu_tile_time_us)

        print(f"CPU execution time: {cpu_tile_time_us:.2f} us")

    avg_npu = sum(npu_tile_times_us) / len(npu_tile_times_us)
    min_npu = min(npu_tile_times_us)
    avg_cpu_tile = sum(cpu_tile_times_us) / len(cpu_tile_times_us)
    min_cpu_tile = min(cpu_tile_times_us)

    return out, avg_npu, min_npu, avg_cpu_tile, min_cpu_tile

def _test_layer_norm():
    torch.manual_seed(0)
    input_tensor = torch.rand(SEQ, EMBD, dtype=torch.float32) * 2.0 - 1.0
    weight_tensor = torch.rand(EMBD, dtype=torch.float32)
    bias_tensor = torch.rand(EMBD, dtype=torch.float32)

    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor (first 5 values): {input_tensor.flatten()[:5].tolist()}")

    # CPU (PyTorch) reference for full tensor
    layer_norm_fn = torch.nn.LayerNorm(EMBD, elementwise_affine=True)
    with torch.no_grad():
        layer_norm_fn.weight.copy_(weight_tensor)
        layer_norm_fn.bias.copy_(bias_tensor)
        start = time.perf_counter()
        input_numpy_cpu = input_tensor.cpu().numpy()                        # input data prep
        ref_out = layer_norm_fn(torch.from_numpy(input_numpy_cpu))          # compute
        ref_numpy = ref_out.cpu().numpy()                                   # output retrieval
        end = time.perf_counter()
    cpu_time_us = (end - start) * 1_000_000

    input_numpy = input_tensor.cpu().numpy()
    weight_numpy = weight_tensor.cpu().numpy()
    bias_numpy = bias_tensor.cpu().numpy()

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        output_allo, avg_npu_us, min_npu_us, avg_cpu_tile_us, min_cpu_tile_us = layernorm_full_seq(
            input_numpy, weight_numpy, bias_numpy, layer_norm_fn
        )

        print(f"\nCPU execution time: {cpu_time_us:.2f} us")
        print(f"Average NPU time:  {avg_npu_us:.2f} us")
        print(f"Min NPU time:      {min_npu_us:.2f} us")
        print(f"Average CPU tile time: {avg_cpu_tile_us:.2f} us")
        print(f"Min CPU tile time:     {min_cpu_tile_us:.2f} us")

        diff = np.abs(output_allo - ref_numpy)
        rel_diff = diff / (np.abs(ref_numpy) + 1e-12)
        max_abs_diff = np.max(diff)
        max_rel_diff = np.max(rel_diff)
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)

        print(f"\nMax absolute difference: {max_abs_diff:.6e}")
        print(f"Max relative difference: {max_rel_diff:.6e}")

        try:
            np.testing.assert_allclose(output_allo, ref_numpy, rtol=RTOL, atol=ATOL)
            print(f"\nPASSED layer_norm! (rtol={RTOL}, atol={ATOL})")
        except AssertionError:
            pct, mism, total = _mismatch_stats(output_allo, ref_numpy, RTOL, ATOL)
            print(f"\nLayer norm mismatch detected.")
            print(f"Mismatch rate: {pct:.4f}% ({mism}/{total})  (rtol={RTOL}, atol={ATOL})")
            r, c = max_idx
            print(f"Worst mismatch at index {max_idx}:")
            print(f"  Input        = {input_numpy[r, c]:.6f}")
            print(f"  Allo output  = {output_allo[r, c]:.6f}")
            print(f"  Torch output = {ref_numpy[r, c]:.6f}")

        print("\nSample comparisons (first 5 elements of first row):")
        for j in range(5):
            print(f"  [{0},{j}] allo={output_allo[0, j]:.6f}  torch={ref_numpy[0, j]:.6f}  diff={diff[0, j]:.6e}")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend run. "
              "Set it to execute the Allo kernel.")

if __name__ == "__main__":
    _test_layer_norm()