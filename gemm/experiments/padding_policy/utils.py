#!/usr/bin/env python3
"""Shared utilities for padding-policy experiments."""

from __future__ import annotations

import json
import re
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


Shape3D = Tuple[int, int, int]
Tile3D = Tuple[int, int, int]

DTYPES = ["i8", "i16", "bf16"]
DEFAULT_INPUTS: List[Shape3D] = [
    (33, 47, 45),
    (63, 63, 63),
    (129, 129, 129),
    (257, 257, 257),
    (513, 513, 513),
    (769, 769, 769),
    (511, 65, 257),
    (1023, 129, 513),
    (1537, 257, 769),
    (65, 511, 257),
    (129, 1023, 513),
    (257, 1537, 769),
    (129, 129, 1023),
    (257, 257, 1537),
    (513, 513, 1900),
    (512, 512, 33),
    (1024, 768, 47),
    (1536, 1024, 63),
]


@dataclass(frozen=True)
class Candidate:
    shape: Shape3D
    tile: Tile3D
    execution_us: float


def shape_key(shape: Shape3D, sep: str = ",") -> str:
    return f"{shape[0]}{sep}{shape[1]}{sep}{shape[2]}"


def parse_shape_key(text: str) -> Shape3D:
    parts = [p.strip() for p in re.split(r"[,_]", text.strip()) if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Invalid shape key: {text}")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def parse_inputs_arg(raw: str) -> List[Shape3D]:
    out: List[Shape3D] = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        out.append(parse_shape_key(part))
    if not out:
        raise ValueError("No valid --inputs parsed.")
    return out


def load_inputs_file(path: Path) -> List[Shape3D]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("inputs", [])
    out: List[Shape3D] = []
    for item in raw:
        if isinstance(item, str):
            out.append(parse_shape_key(item))
            continue
        if isinstance(item, (list, tuple)) and len(item) == 3:
            out.append((int(item[0]), int(item[1]), int(item[2])))
            continue
        raise ValueError(f"Unsupported input item in {path}: {item}")
    if not out:
        raise ValueError(f"No valid inputs in file: {path}")
    return out


def dedupe_inputs(inputs: Iterable[Shape3D]) -> List[Shape3D]:
    out: List[Shape3D] = []
    seen = set()
    for shp in inputs:
        if shp in seen:
            continue
        seen.add(shp)
        out.append(shp)
    return out


def input_feature(shape: Shape3D, feature: str) -> int:
    m, n, k = shape
    if feature == "ab_elements":
        # Input tensor sizes only: |A| + |B| = M*K + K*N
        return m * k + k * n
    if feature == "touch_elements":
        return m * k + k * n + m * n
    if feature == "volume":
        return m * n * k
    if feature == "max_dim":
        return max(m, n, k)
    raise ValueError(f"Unsupported feature: {feature}")


class ProfileDB:
    def __init__(self, profile_path: Path):
        with profile_path.open("r", encoding="utf-8") as f:
            raw = json.load(f) or {}

        self._by_dtype: Dict[str, Dict[Shape3D, Candidate]] = {dtype: {} for dtype in DTYPES}
        shapes: List[Shape3D] = []

        for key, shape_entry in raw.items():
            try:
                shape = parse_shape_key(key)
            except Exception:
                continue
            shapes.append(shape)
            for dtype in DTYPES:
                dtype_entry = (shape_entry or {}).get(dtype, {})
                best = (dtype_entry or {}).get("Best m,n,k", {})
                tile = best.get("size")
                exec_us = best.get("Best NPU average time")
                if tile is None or exec_us is None:
                    continue
                try:
                    cand = Candidate(
                        shape=shape,
                        tile=(int(tile[0]), int(tile[1]), int(tile[2])),
                        execution_us=float(exec_us),
                    )
                except (TypeError, ValueError, IndexError):
                    continue
                self._by_dtype[dtype][shape] = cand

        self.shapes = sorted(set(shapes))
        self.unique_M = sorted({m for m, _, _ in self.shapes})
        self.unique_N = sorted({n for _, n, _ in self.shapes})
        self.unique_K = sorted({k for _, _, k in self.shapes})

        self._sorted_by_exec: Dict[str, List[Candidate]] = {}
        for dtype in DTYPES:
            vals = list(self._by_dtype[dtype].values())
            vals.sort(key=lambda c: (c.execution_us, c.shape[0], c.shape[1], c.shape[2]))
            self._sorted_by_exec[dtype] = vals

    @staticmethod
    def _nearest_larger(sorted_vals: List[int], target: int) -> Optional[int]:
        idx = bisect_left(sorted_vals, target)
        if idx >= len(sorted_vals):
            return None
        return sorted_vals[idx]

    def feasible(self, dtype: str, input_shape: Shape3D) -> List[Candidate]:
        m0, n0, k0 = input_shape
        out: List[Candidate] = []
        for cand in self._sorted_by_exec[dtype]:
            m, n, k = cand.shape
            if m >= m0 and n >= n0 and k >= k0:
                out.append(cand)
        return out

    def candidate_for_shape(self, dtype: str, shape: Shape3D) -> Optional[Candidate]:
        return self._by_dtype.get(dtype, {}).get(shape)

    def closest_shape_ge(self, input_shape: Shape3D) -> Optional[Shape3D]:
        m0, n0, k0 = input_shape
        m = self._nearest_larger(self.unique_M, m0)
        n = self._nearest_larger(self.unique_N, n0)
        k = self._nearest_larger(self.unique_K, k0)
        if m is None or n is None or k is None:
            return None
        return (m, n, k)

    def closest_candidate(self, dtype: str, input_shape: Shape3D) -> Candidate:
        m0, n0, k0 = input_shape
        m = self._nearest_larger(self.unique_M, m0)
        n = self._nearest_larger(self.unique_N, n0)
        k = self._nearest_larger(self.unique_K, k0)
        if m is not None and n is not None and k is not None:
            cand = self._by_dtype[dtype].get((m, n, k))
            if cand is not None:
                return cand
        feasible = self.feasible(dtype, input_shape)
        if not feasible:
            raise ValueError(f"No feasible profile candidate for dtype={dtype} input={input_shape}")
        return min(
            feasible,
            key=lambda c: (
                c.shape[0] * c.shape[1] * c.shape[2],
                c.shape[0],
                c.shape[1],
                c.shape[2],
            ),
        )

    def fastest_candidate(self, dtype: str, input_shape: Shape3D) -> Candidate:
        m0, n0, k0 = input_shape
        for cand in self._sorted_by_exec[dtype]:
            m, n, k = cand.shape
            if m >= m0 and n >= n0 and k >= k0:
                return cand
        raise ValueError(f"No feasible profile candidate for dtype={dtype} input={input_shape}")


def parse_gemm_output(raw: str, dtype: str) -> dict:
    pad_match = re.search(rf"\[{dtype}\] Padding time:\s*([\d\.]+)us", raw)
    unpad_match = re.search(rf"\[{dtype}\] Unpadding time:\s*([\d\.]+)us", raw)
    npu_matches = re.findall(r"Avg NPU execution time:\s*([\d\.]+)us", raw)
    return {
        "padding_us": float(pad_match.group(1)) if pad_match else None,
        "unpadding_us": float(unpad_match.group(1)) if unpad_match else None,
        "execution_us": float(npu_matches[-1]) if npu_matches else None,
        "passed": "PASSED!" in raw,
        "mapping_executed": bool(npu_matches),
    }


def load_best_padding_from_sweep(path: Path) -> Dict[Tuple[Shape3D, str], dict]:
    """
    Returns best-per-input map keyed by ((M,N,K), dtype).
    Ranking metric is minimal padding_us (ties break on lower unpadding_us).
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f) or {}

    out: Dict[Tuple[Shape3D, str], dict] = {}
    for in_key, dtype_block in (raw.get("inputs") or {}).items():
        try:
            input_shape = parse_shape_key(in_key)
        except Exception:
            continue
        if not isinstance(dtype_block, dict):
            continue
        for dtype, candidates in dtype_block.items():
            if dtype not in DTYPES or not isinstance(candidates, dict):
                continue
            best = None
            best_score = None
            for padded_key, metrics in candidates.items():
                if not isinstance(metrics, dict):
                    continue
                pad_us = metrics.get("padding_us")
                if pad_us is None:
                    continue
                try:
                    pad = float(pad_us)
                    unpad = float(metrics.get("unpadding_us") or 0.0)
                    padded_shape = parse_shape_key(padded_key)
                except (TypeError, ValueError):
                    continue
                score = (pad, unpad)
                if best_score is None or score < best_score:
                    best_score = score
                    best = {
                        "padding_us": pad,
                        "unpadding_us": unpad,
                        "overhead_us": pad + unpad,
                        "best_padded_shape": list(padded_shape),
                        "source_file": str(path),
                    }
            if best is None:
                continue
            key = (input_shape, dtype)
            prev = out.get(key)
            if prev is None:
                out[key] = best
                continue
            prev_score = (float(prev["padding_us"]), float(prev.get("unpadding_us", 0.0)))
            if best_score is not None and best_score < prev_score:
                out[key] = best
    return out


def load_padding_lookup(path: Path) -> Dict[Tuple[Shape3D, str, Shape3D], dict]:
    """
    Returns exact per-candidate lookup:
      key = ((M,N,K), dtype, (Mp,Np,Kp))
      value = {"padding_us": float, "unpadding_us": float, "overhead_us": float}
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f) or {}

    out: Dict[Tuple[Shape3D, str, Shape3D], dict] = {}
    for in_key, dtype_block in (raw.get("inputs") or {}).items():
        try:
            input_shape = parse_shape_key(in_key)
        except Exception:
            continue
        if not isinstance(dtype_block, dict):
            continue
        for dtype, candidates in dtype_block.items():
            if dtype not in DTYPES or not isinstance(candidates, dict):
                continue
            for padded_key, metrics in candidates.items():
                if not isinstance(metrics, dict):
                    continue
                pad_us = metrics.get("padding_us")
                if pad_us is None:
                    continue
                try:
                    padded_shape = parse_shape_key(padded_key)
                    pad = float(pad_us)
                    unpad = float(metrics.get("unpadding_us") or 0.0)
                except (TypeError, ValueError):
                    continue
                out[(input_shape, dtype, padded_shape)] = {
                    "padding_us": pad,
                    "unpadding_us": unpad,
                    "overhead_us": pad + unpad,
                }
    return out


def fit_threshold_1d(
    rows: List[dict],
    small_strategy: str,
    large_strategy: str,
) -> dict:
    """
    rows: [{"feature": int, "manual_copy_us": float, "numpy_pad_us": float}]
    Evaluate threshold policy:
      if feature <= t -> small_strategy else large_strategy
    and return best threshold minimizing average chosen latency.
    """
    if not rows:
        raise ValueError("Cannot fit threshold with empty rows")

    feature_vals = sorted({int(r["feature"]) for r in rows})
    candidates = [feature_vals[0] - 1] + feature_vals
    best = None

    for threshold in candidates:
        chosen_sum = 0.0
        regret_sum = 0.0
        wins = 0
        for r in rows:
            f = int(r["feature"])
            manual = float(r["manual_copy_us"])
            numpy_ = float(r["numpy_pad_us"])
            if f <= threshold:
                chosen = manual if small_strategy == "manual_copy" else numpy_
            else:
                chosen = manual if large_strategy == "manual_copy" else numpy_
            oracle = min(manual, numpy_)
            chosen_sum += chosen
            regret_sum += (chosen - oracle)
            if abs(chosen - oracle) < 1e-12:
                wins += 1

        num = len(rows)
        avg_chosen = chosen_sum / num
        avg_regret = regret_sum / num
        accuracy = wins / num
        metric = (avg_chosen, avg_regret, -accuracy, threshold)
        if best is None or metric < best["metric"]:
            best = {
                "threshold": threshold,
                "avg_chosen_us": avg_chosen,
                "avg_regret_us": avg_regret,
                "oracle_match_rate": accuracy,
                "metric": metric,
            }

    assert best is not None
    return {
        "threshold": int(best["threshold"]),
        "avg_chosen_us": float(best["avg_chosen_us"]),
        "avg_regret_us": float(best["avg_regret_us"]),
        "oracle_match_rate": float(best["oracle_match_rate"]),
    }
