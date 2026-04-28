#!/usr/bin/env python3
"""
Create a compact profile JSON containing only:
- shape key (M_N_K)
- per-dtype best tile (best_mnk)
- per-dtype best execution time (best_exec_us)
"""

import argparse
import json
import os


DTYPES = ["i8", "i16", "bf16"]


def to_shape(key: str):
    m_str, n_str, k_str = key.split("_")
    return int(m_str), int(n_str), int(k_str)


def build_compact(full_profile: dict) -> dict:
    entries = {}
    unique_m, unique_n, unique_k = set(), set(), set()

    for key, shape_entry in full_profile.items():
        try:
            m, n, k = to_shape(key)
        except Exception:
            continue
        compact_shape = {}
        for dtype in DTYPES:
            dtype_entry = shape_entry.get(dtype, {})
            best = dtype_entry.get("Best m,n,k", {})
            tile = best.get("size")
            exec_us = best.get("Best NPU average time")
            if tile is None or exec_us is None:
                continue
            compact_shape[dtype] = {
                "best_mnk": [int(tile[0]), int(tile[1]), int(tile[2])],
                "best_exec_us": float(exec_us),
            }
        if not compact_shape:
            continue
        entries[key] = compact_shape
        unique_m.add(m)
        unique_n.add(n)
        unique_k.add(k)

    return {
        "format": "vla_profile_compact_v1",
        "entries": entries,
        "unique_M": sorted(unique_m),
        "unique_N": sorted(unique_n),
        "unique_K": sorted(unique_k),
        "num_shapes": len(entries),
    }


def main():
    parser = argparse.ArgumentParser(description="Build compact GEMM profile JSON.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/vla_profile.json",
        help="Path to full vla_profile.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/vla_profile_compact.json",
        help="Path to write compact profile",
    )
    args = parser.parse_args()

    base = os.path.dirname(__file__)
    in_path = args.input if os.path.isabs(args.input) else os.path.join(base, args.input)
    out_path = args.output if os.path.isabs(args.output) else os.path.join(base, args.output)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        full_profile = json.load(f) or {}
    compact = build_compact(full_profile)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)

    print(f"Wrote compact profile to: {out_path}")
    print(f"Shapes retained: {compact['num_shapes']}")


if __name__ == "__main__":
    main()
