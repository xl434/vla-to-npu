#!/usr/bin/env python3
"""
Agentic Tiling Explorer — incremental, resumable.
Saves results after EVERY run so no data is lost on interruption.

Usage:
  python3 run_tiling_explorer.py
"""

import json, os, re, subprocess, sys
from datetime import datetime
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
AGENT_DIR      = Path("/home/ec935/agent-gemm")
SCRIPTS_DIR    = Path("/home/ec935/vla-to-npu/gemm/scripts")
SHAPES_FILE    = AGENT_DIR / "references/smol-vla-dataset/all_practical_shapes.json"
PROFILING_JSON = AGENT_DIR / "references/memory/gemm-data/npu_execution_profiling.json"
ERRORS_JSON    = AGENT_DIR / "references/memory/gemm-data/profiling_errors.json"
PROFILE_JSON   = AGENT_DIR / "prompts/agent-tiling-profile.json"

# ─── Session ──────────────────────────────────────────────────────────────────
DTYPE        = "bf16"
MAX_TILES    = 5
TILE_POOL    = [16, 32, 64, 128, 256, 512]
DSIZE        = 2        # bf16 bytes per element
MEMORY_LIMIT = 32256    # bytes, tile memory budget

SESSION_ID = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

# ─── Tile validity helpers ────────────────────────────────────────────────────
def valid_MN(Dp, t):
    """M or N tile: must divide Dp AND quotient divisible by 3, 4, or 5."""
    if t > Dp or Dp % t != 0:
        return False
    return any((Dp // t) % c == 0 for c in (3, 4, 5))

def valid_K(Kp, k):
    """K tile: divisibility only."""
    return k <= Kp and Kp % k == 0

def fits_memory(m, n, k):
    return (m * n + n * k + k * m) * DSIZE <= MEMORY_LIMIT

def priority_key(tile):
    """Lower = higher priority."""
    m, n, k = tile
    p1 = 0 if all(64 <= x <= 128 for x in (m, n, k)) else 1
    p2 = 0 if n == 128 else 1
    p3 = 0 if m == n == k else 1
    p4 = -(m * n * k)
    p5 = min(m, n, k)
    return (p1, p2, p3, p4, p5)

# ─── Load all data ────────────────────────────────────────────────────────────
print("Loading data files...", flush=True)

with open(SHAPES_FILE) as f:
    raw_shapes = json.load(f)

with open(PROFILING_JSON) as f:
    profiling = json.load(f)

if ERRORS_JSON.exists():
    with open(ERRORS_JSON) as f:
        errors_db = json.load(f)
else:
    errors_db = {}

if PROFILE_JSON.exists():
    with open(PROFILE_JSON) as f:
        agent_profile = json.load(f)
else:
    agent_profile = {
        "description": (
            "Tile exploration run log. Each entry is a single (shape, dtype, tile) "
            "trial run by the agentic tiling explorer."
        ),
        "schema": {
            "session_id": "ISO-8601 timestamp of session start",
            "shape": "Mp_Np_Kp string key",
            "input": {"M": "int", "N": "int", "K": "int"},
            "padded": {"M": "int", "N": "int", "K": "int"},
            "dtype": "i8 | i16 | bf16",
            "tile": [0, 0, 0],
            "pad_impl": "manual_copy | numpy_pad | none",
            "result": {
                "status": "passed | failed | npu_unavailable",
                "npu_avg_us": "float or null",
                "npu_min_us": "float or null",
                "padding_us": "float or null",
                "correctness": "true | false | null",
                "error_class": "string or null",
                "error_message": "string or null"
            }
        },
        "runs": []
    }

# Build set of already-explored tiles from agent_profile (this session's dtype only)
explored = {}  # shape_key -> set of (m, n, k)
for run in agent_profile["runs"]:
    if run.get("dtype") != DTYPE:
        continue
    sk = run["shape"]
    explored.setdefault(sk, set()).add(tuple(run["tile"]))

print(f"Loaded {len(raw_shapes)} shapes, {len(agent_profile['runs'])} profile runs", flush=True)

# ─── Build plan ───────────────────────────────────────────────────────────────
# All shapes in all_practical_shapes.json are in the current ALLOWED lists →
# no padding needed; Mp=M, Np=N, Kp=K for every shape.
unique_combos = sorted(set((s["M"], s["N"], s["K"]) for s in raw_shapes))

plan = []   # list of (M, N, K, novel_tiles)
for (M, N, K) in unique_combos:
    sk = f"{M}_{N}_{K}"

    # Build candidate tile pool
    vm = [t for t in TILE_POOL if valid_MN(M, t)]
    vn = [t for t in TILE_POOL if valid_MN(N, t)]
    vk = [t for t in TILE_POOL if valid_K(K, t)]
    candidates = [
        (m, n, k)
        for m in vm for n in vn for k in vk
        if fits_memory(m, n, k) and (m, n, k) != (128, 128, 128)
    ]

    # Already-known tiles for this shape/dtype
    # Guard against non-dict entries (some profiling entries are lists or None)
    _raw_pd = profiling.get(sk, {}).get(DTYPE)
    pd_entry = _raw_pd if isinstance(_raw_pd, dict) else {}
    already_working = {tuple(v["size"]) for v in pd_entry.get("Working m,n,k", {}).values()}
    already_failing = {tuple(t) for t in pd_entry.get("Failing m,n,k", [])}

    errors_failing = set()
    for tk in errors_db.get(sk, {}).get(DTYPE, {}).keys():
        parts = tk.split("_")
        if len(parts) == 3:
            try:
                errors_failing.add(tuple(int(x) for x in parts))
            except ValueError:
                pass

    already_explored = explored.get(sk, set())

    novel = [
        t for t in candidates
        if t not in already_working
        and t not in already_failing
        and t not in errors_failing
        and t not in already_explored
    ]
    if not novel:
        continue

    novel.sort(key=priority_key)
    plan.append((M, N, K, novel[:MAX_TILES]))

total_runs = sum(len(tiles) for _, _, _, tiles in plan)
print(f"Plan: {len(plan)} combos, {total_runs} runs", flush=True)

# ─── NPU check ───────────────────────────────────────────────────────────────
npu_check = subprocess.run(
    ["python3", "-c", "from allo.backend.aie import is_available; print(is_available())"],
    capture_output=True, text=True, cwd=str(SCRIPTS_DIR)
)
npu_available = "True" in npu_check.stdout
print(f"NPU available: {npu_available}", flush=True)

# ─── Atomic save helpers ──────────────────────────────────────────────────────
def _atomic_save(path, data):
    """Write JSON to a temp file then rename, to avoid corruption on kill."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)

def save_agent_profile():
    _atomic_save(PROFILE_JSON, agent_profile)

def save_profiling():
    _atomic_save(PROFILING_JSON, profiling)

def save_errors():
    _atomic_save(ERRORS_JSON, errors_db)

# ─── Per-run record + save ────────────────────────────────────────────────────
def record_run(M, N, K, tile, status, npu_avg, npu_min, pad_us, correctness,
               error_class, error_msg):
    """Append one run entry and immediately save all affected files."""
    m, n, k = tile
    sk = f"{M}_{N}_{K}"

    # ── agent_profile ──────────────────────────────────────────────────────────
    agent_profile["runs"].append({
        "session_id": SESSION_ID,
        "shape": sk,
        "input":  {"M": M,  "N": N,  "K": K},
        "padded": {"M": M,  "N": N,  "K": K},
        "dtype":  DTYPE,
        "tile":   [m, n, k],
        "pad_impl": "none",
        "result": {
            "status":        status,
            "npu_avg_us":    npu_avg,
            "npu_min_us":    npu_min,
            "padding_us":    pad_us,
            "correctness":   correctness,
            "error_class":   error_class,
            "error_message": error_msg
        }
    })
    save_agent_profile()

    # ── profiling.json ─────────────────────────────────────────────────────────
    if sk not in profiling:
        profiling[sk] = {}
    # Ensure DTYPE entry is a proper dict (guard against list/None entries)
    if not isinstance(profiling[sk].get(DTYPE), dict):
        profiling[sk][DTYPE] = {
            "Best m,n,k": None,
            "Working m,n,k": {},
            "Failing m,n,k": []
        }
    pd = profiling[sk][DTYPE]
    tile_key = f"{m}_{n}_{k}"

    if status == "passed" and npu_avg is not None:
        w = pd.setdefault("Working m,n,k", {})
        if tile_key in w:
            old = w[tile_key]
            cnt = old["count"] + 1
            old["avg"] = (old["avg"] * old["count"] + npu_avg) / cnt
            old["count"] = cnt
            if npu_min is not None and npu_min < old.get("min", npu_min + 1):
                old["min"] = npu_min
        else:
            w[tile_key] = {"size": [m, n, k], "avg": npu_avg, "min": npu_min, "count": 1}
        # Update best — handle both key formats in existing data:
        #   old: {"size": [...], "Best NPU average time": X}
        #   new: {"tile": [...], "avg": X, "min": Y}
        best = pd.get("Best m,n,k")
        def _best_avg(b):
            if b is None: return float("inf")
            if "Best NPU average time" in b: return b["Best NPU average time"]
            if "avg" in b: return b["avg"]
            return float("inf")
        if _best_avg(best) > npu_avg:
            pd["Best m,n,k"] = {"size": [m, n, k], "Best NPU average time": npu_avg}

    elif status == "failed":
        fails = pd.setdefault("Failing m,n,k", [])
        if [m, n, k] not in fails:
            fails.append([m, n, k])

    save_profiling()


def record_error_trace(M, N, K, tile, tb_lines):
    """Store error traceback in errors_db and save."""
    m, n, k = tile
    sk = f"{M}_{N}_{K}"
    tile_key = f"{m}_{n}_{k}"
    errors_db.setdefault(sk, {}).setdefault(DTYPE, {})
    dtype_err = errors_db[sk][DTYPE]
    if tile_key not in dtype_err:
        dtype_err[tile_key] = tb_lines
    else:
        i = 2
        while f"{tile_key}__{i}" in dtype_err:
            i += 1
        dtype_err[f"{tile_key}__{i}"] = tb_lines
    save_errors()


# ─── Run one tile ─────────────────────────────────────────────────────────────
def run_tile(M, N, K, m, n, k):
    """
    Execute one NPU test via CLI.
    Returns (status, npu_avg, npu_min, pad_us, error_class, error_msg, tb_lines)
    """
    cmd = [
        "python3", "v2_test_mapping_large_gemm.py",
        "--M", str(M), "--N", str(N), "--K", str(K),
        "--m", str(m), "--n", str(n), "--k", str(k),
        "--dtype", DTYPE,
    ]
    env = os.environ.copy()
    env["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(SCRIPTS_DIR), env=env, timeout=180
        )
        output = result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired:
        return ("failed", None, None, None,
                "Timeout", "Compile/run timeout >180s",
                ["TimeoutExpired: process killed after 180s"])

    # ── Parse timing ───────────────────────────────────────────────────────────
    avg_m = re.search(r"Avg NPU execution time:\s*([\d.]+)us", output)
    min_m = re.search(r"Min NPU execution time:\s*([\d.]+)us", output)
    pad_m = re.search(r"\[bf16\] Padding time:\s*([\d.]+)us", output)

    npu_avg = float(avg_m.group(1)) if avg_m else None
    npu_min = float(min_m.group(1)) if min_m else None
    pad_us  = float(pad_m.group(1)) if pad_m else None

    # ── Classify ───────────────────────────────────────────────────────────────
    passed = "PASSED!" in output

    # bf16 + K=2048 false-positive: PASSED followed by CDO error
    if passed and "Failed to generate cdo" in output:
        return ("failed", None, None, pad_us,
                "CDOError", "Failed to generate cdo (bf16 K=2048 false positive)",
                [l for l in output.splitlines() if l.strip()][-5:])

    if passed and npu_avg is not None:
        return ("passed", npu_avg, npu_min, pad_us, None, None, [])

    # NPU offline
    if "XDNA driver not found" in output or "is_available" in output.lower():
        return ("npu_unavailable", None, None, None,
                "NPUUnavailable", "XDNA driver not found", [])

    tb_tail = [l for l in output.splitlines() if l.strip()][-10:]

    if "Failed to generate cdo" in output:
        return ("failed", None, None, pad_us,
                "CDOError", "Failed to generate cdo", tb_tail)
    if "ZeroDivisionError" in output:
        return ("failed", None, None, pad_us,
                "ZeroDivisionError", "ZeroDivisionError", tb_tail)
    if "Unresolvable mapping" in output:
        msg_m = re.search(r"Unresolvable mapping[^\n]*", output)
        return ("failed", None, None, pad_us,
                "UnresolvableMapping",
                msg_m.group(0) if msg_m else "Unresolvable mapping", tb_tail)
    if "allocated buffers exceeded" in output or ("AIE core" in output and "error" in output.lower()):
        return ("failed", None, None, pad_us,
                "MemoryExceeded", "allocated buffers exceeded", tb_tail)
    if "Tile sizes do not divide" in output:
        return ("failed", None, None, pad_us,
                "DivisibilityError", "Tile sizes do not divide padded shape", tb_tail)
    if "bfloat16 have accuracy issue" in output:
        return ("failed", None, None, pad_us,
                "AccuracyFailure", "bf16 accuracy check failed", tb_tail)
    # Script ran but no PASSED and no known error
    if passed and npu_avg is None:
        return ("failed", None, None, pad_us,
                "NoPASSED", "Script ran but no timing found", tb_tail)
    if not passed:
        first_err = next((l.strip() for l in output.splitlines() if "Error" in l or "error" in l), "unknown")
        return ("failed", None, None, pad_us,
                "Unknown", first_err[:120], tb_tail)

    return ("failed", None, None, pad_us, "Unknown", "unknown error", tb_tail)


# ─── Main execution loop ──────────────────────────────────────────────────────
# Capture prior bests before any run updates them
prior_bests = {}
for sk, data in profiling.items():
    pd_val = data.get(DTYPE) if isinstance(data, dict) else None
    if isinstance(pd_val, dict) and pd_val.get("Best m,n,k"):
        prior_bests[sk] = dict(pd_val["Best m,n,k"])

run_idx    = 0
passed_cnt = 0
failed_cnt = 0
new_bests  = []   # (sk, old_tile, old_avg, new_tile, new_avg)
npu_dead   = False

combo_results = {}  # sk -> list of (tile, status, avg, min, pad)

if not npu_available:
    print("NPU not available — logging all planned runs as npu_unavailable", flush=True)

for (M, N, K, tiles) in plan:
    sk = f"{M}_{N}_{K}"
    combo_results[sk] = []

    for tile in tiles:
        m, n, k = tile
        run_idx += 1

        if not npu_available or npu_dead:
            record_run(M, N, K, tile, "npu_unavailable",
                       None, None, None, None,
                       "NPUUnavailable", "NPU not available for this session")
            combo_results[sk].append((tile, "npu_unavailable", None, None, None))
            print(f"[{run_idx}/{total_runs}] ({sk}) {DTYPE}  tile={list(tile)}  →  NPU_UNAVAILABLE",
                  flush=True)
            continue

        status, npu_avg, npu_min, pad_us, err_cls, err_msg, tb = run_tile(M, N, K, m, n, k)

        if status == "npu_unavailable":
            npu_dead = True

        correctness = True if status == "passed" else (False if status == "failed" else None)
        record_run(M, N, K, tile, status, npu_avg, npu_min, pad_us,
                   correctness, err_cls, err_msg)
        if status == "failed" and tb:
            record_error_trace(M, N, K, tile, tb)

        combo_results[sk].append((tile, status, npu_avg, npu_min, pad_us))

        if status == "passed":
            passed_cnt += 1
            avg_s = f"avg={npu_avg:.1f}µs" if npu_avg else "avg=N/A"
            min_s = f"min={npu_min:.1f}µs" if npu_min else "min=N/A"
            pad_s = f"pad={pad_us:.1f}µs" if pad_us else "pad=N/A"
            print(f"[{run_idx}/{total_runs}] ({sk}) {DTYPE}  tile={list(tile)}"
                  f"  →  PASSED  {avg_s}  {min_s}  {pad_s}", flush=True)
        else:
            failed_cnt += 1
            print(f"[{run_idx}/{total_runs}] ({sk}) {DTYPE}  tile={list(tile)}"
                  f"  →  FAILED  avg=N/A  min=N/A  pad=N/A  ERR={err_cls}",
                  flush=True)

    # ── Per-combo summary ──────────────────────────────────────────────────────
    c_pass = sum(1 for _, s, *_ in combo_results[sk] if s == "passed")
    c_fail = sum(1 for _, s, *_ in combo_results[sk] if s in ("failed",))
    print(f"  --- Combo {sk} {DTYPE}: {c_pass} PASSED, {c_fail} FAILED ---", flush=True)

    # Check for new best (handle both key formats)
    cur_best  = profiling.get(sk, {}).get(DTYPE, {}).get("Best m,n,k") if isinstance(profiling.get(sk, {}).get(DTYPE), dict) else None
    prev_best = prior_bests.get(sk)
    if cur_best and prev_best:
        cur_avg  = cur_best.get("Best NPU average time") or cur_best.get("avg")
        prev_avg = prev_best.get("Best NPU average time") or prev_best.get("avg")
        cur_tile  = cur_best.get("size") or cur_best.get("tile")
        prev_tile = prev_best.get("size") or prev_best.get("tile")
        if cur_avg and prev_avg and cur_avg < prev_avg:
            new_bests.append((sk, prev_tile, prev_avg, cur_tile, cur_avg))

# ─── Write session tiling log ─────────────────────────────────────────────────
log_dir = AGENT_DIR / "references/memory/tiling-logs"
log_dir.mkdir(parents=True, exist_ok=True)
now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_name = f"tiling_log_{now_str}.md"
log_path = log_dir / log_name

with open(log_path, "w") as lf:
    lf.write(f"# Tiling Exploration Log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    lf.write(f"**Session ID**: {SESSION_ID}  \n")
    lf.write(f"**shapes_file**: all_practical_shapes.json  \n")
    lf.write(f"**dtype**: bf16  \n")
    lf.write(f"**max_tiles_per_combo**: {MAX_TILES}  \n\n")
    lf.write(f"## Summary\n\n")
    lf.write(f"- Combos explored: {len(plan)}\n")
    lf.write(f"- Total runs: {run_idx}\n")
    lf.write(f"- Passed: {passed_cnt}\n")
    lf.write(f"- Failed: {failed_cnt}\n")
    lf.write(f"- New best tiles: {len(new_bests)}\n\n")
    if new_bests:
        lf.write("## New Best Tiles\n\n")
        lf.write("| Shape | Old tile | Old avg | New tile | New avg | Δ |\n")
        lf.write("|-------|----------|---------|----------|---------|---|\n")
        for sk, ot, oa, nt, na in new_bests:
            delta = f"{(na - oa) / oa * 100:+.1f}%"
            lf.write(f"| {sk} | {ot} | {oa:.1f}µs | {nt} | {na:.1f}µs | {delta} |\n")

print(f"\nSession log written to {log_path}", flush=True)

# ─── Final banner ─────────────────────────────────────────────────────────────
print("\n" + "═" * 70, flush=True)
print("║          TILING EXPLORATION — SESSION COMPLETE" + " " * 22 + "║", flush=True)
print("═" * 70, flush=True)
print(f"  Date:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(f"  Shapes tested: {len(plan)}  (padded combos: {len(plan)})", flush=True)
print(f"  Total runs:    {run_idx}  │  Passed: {passed_cnt}  │  Failed: {failed_cnt}", flush=True)
print(f"  New tiles in profiling DB: {passed_cnt}", flush=True)
print(f"  New best tiles found:      {len(new_bests)}", flush=True)
print("═" * 70, flush=True)

# Per-shape results
print("\n── Per-shape results ──", flush=True)
for (M, N, K, tiles) in plan:
    sk = f"{M}_{N}_{K}"
    rows = combo_results.get(sk, [])
    if not rows:
        continue
    prev_best = prior_bests.get(sk)
    prev_avg  = (prev_best.get("Best NPU average time") or prev_best.get("avg")) if prev_best else None
    print(f"\nShape ({M},{N},{K})  dtype={DTYPE}", flush=True)
    print(f"{'Tile':<20} {'Status':<10} {'Avg NPU':>9} {'Min NPU':>9} {'Pad':>9}  vs. prior best", flush=True)
    print("-" * 70, flush=True)
    for (tile, status, avg, mn, pad) in rows:
        avg_s = f"{avg:.1f}µs" if avg else "—"
        min_s = f"{mn:.1f}µs"  if mn  else "—"
        pad_s = f"{pad:.1f}µs" if pad else "—"
        if status == "passed" and avg is not None and prev_avg is not None:
            pct = (avg - prev_avg) / prev_avg * 100
            cmp = f"NEW BEST ▼{abs(pct):.1f}%" if avg < prev_avg else f"+{pct:.1f}% vs best"
        elif status == "passed" and prev_avg is None:
            cmp = "first data"
        else:
            cmp = ""
        print(f"  {str(list(tile)):<18} {status:<10} {avg_s:>9} {min_s:>9} {pad_s:>9}  {cmp}",
              flush=True)
    if prev_best:
        prev_tile = prev_best.get("size") or prev_best.get("tile")
        pd_val = profiling.get(sk, {}).get(DTYPE)
        cur_best = pd_val.get("Best m,n,k") if isinstance(pd_val, dict) else None
        cur_avg  = (cur_best.get("Best NPU average time") or cur_best.get("avg")) if cur_best else None
        cur_tile = (cur_best.get("size") or cur_best.get("tile")) if cur_best else None
        print(f"  Prior best: {prev_tile} at {prev_avg:.1f}µs", flush=True)
        if cur_best and cur_avg and prev_avg and cur_avg < prev_avg:
            print(f"  New best:   {cur_tile} at {cur_avg:.1f}µs", flush=True)

# New best tiles summary
print("\n" + "─" * 70, flush=True)
if new_bests:
    print("NEW BEST TILES DISCOVERED", flush=True)
    print("─" * 70, flush=True)
    hdr = f"  {'Shape':<22} {'Old best':<14} {'Old avg':>8}  {'New best':<14} {'New avg':>8}  {'Δ':>7}"
    print(hdr, flush=True)
    for sk, ot, oa, nt, na in new_bests:
        delta = f"{(na - oa) / oa * 100:+.1f}%"
        print(f"  {sk:<22} {str(ot):<14} {oa:>7.1f}µs  {str(nt):<14} {na:>7.1f}µs  {delta:>7}",
              flush=True)
else:
    print("NO NEW BEST TILES — all tested tiles matched or were slower than prior bests",
          flush=True)

# Files updated
print("\nFiles updated this session:", flush=True)
print(f"  ✓ npu_execution_profiling.json  — {passed_cnt} new working tiles, {failed_cnt} new failing tiles", flush=True)
print(f"  ✓ profiling_errors.json         — updated", flush=True)
print(f"  ✓ prompts/agent-tiling-profile.json — {run_idx} new run entries appended", flush=True)
print(f"  ✓ references/memory/tiling-logs/{log_name}", flush=True)
