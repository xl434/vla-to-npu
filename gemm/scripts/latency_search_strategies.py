#!/usr/bin/env python3
"""
Latency search strategies for choosing padded GEMM shape.

This module is intentionally "functions-ready" (library-style), not a full CLI app.
You can import it and call one of the three strategy functions:

1) select_with_hashmap_27(...)
2) select_with_sorted_first_fit(...)
3) select_with_preprocessed_map(...)

All strategies return SearchResult with:
- decision_us: time spent selecting padded shape and finding its best tile
- padding_us: estimated pad+unpad time
- execution_us: best known NPU runtime for chosen padded shape
- total_us: decision_us + padding_us + execution_us
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from bisect import bisect_left
import itertools
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


Shape3D = Tuple[int, int, int]
Tile3D = Tuple[int, int, int]

DTYPES = ["i8", "i16", "bf16"]
STRATEGIES = ["hashmap_27", "sorted_first_fit", "preprocessed_map"]
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
DEFAULT_PROFILE = "../trials/vla_profile_combined.json"


@dataclass(frozen=True)
class CandidateInfo:
    shape: Shape3D
    best_tile: Tile3D
    execution_us: float


@dataclass(frozen=True)
class TilePerf:
    tile: Tile3D
    execution_us: float


@dataclass(frozen=True)
class SearchResult:
    strategy: str
    input_shape: Shape3D
    dtype: str
    chosen_shape: Shape3D
    chosen_tile: Tile3D
    decision_us: float
    padding_us: float
    execution_us: float
    total_us: float
    num_candidates_considered: int


class ProfileDB:
    """
    Lightweight view over profile JSON.

    Expected format:
    - Combined/full profile keyed by shape "M_N_K" with per-dtype entries.
    """

    def __init__(self, profile_path: str):
        with open(profile_path, "r", encoding="utf-8") as f:
            raw = json.load(f) or {}

        if isinstance(raw, dict) and raw.get("format") == "vla_profile_compact_v1":
            raise ValueError(
                "Compact profile format is no longer supported here. "
                "Use a combined/full profile such as vla_profile_combined.json."
            )
        self._entries = raw
        self._best_by_dtype: Dict[str, Dict[Shape3D, CandidateInfo]] = {}
        self._tile_options_by_dtype: Dict[str, Dict[Shape3D, List[TilePerf]]] = {}
        self.shapes: List[Shape3D] = []
        for key, shape_entry in self._entries.items():
            try:
                m, n, k = [int(x) for x in key.split("_")]
            except ValueError:
                continue
            shape = (m, n, k)
            self.shapes.append(shape)

            # Parse best candidate once and keep in tuple-key map for fast lookups.
            for dtype, dtype_entry in shape_entry.items():
                tile_options = self._parse_working_tiles(dtype_entry)
                if tile_options:
                    self._tile_options_by_dtype.setdefault(dtype, {})[shape] = tile_options
                    best = min(tile_options, key=lambda x: x.execution_us)
                    cand = CandidateInfo(shape=shape, best_tile=best.tile, execution_us=best.execution_us)
                else:
                    cand = self._parse_candidate_entry(shape, dtype_entry)
                    if cand is not None:
                        self._tile_options_by_dtype.setdefault(dtype, {})[shape] = [
                            TilePerf(tile=cand.best_tile, execution_us=cand.execution_us)
                        ]
                if cand is None:
                    continue
                self._best_by_dtype.setdefault(dtype, {})[shape] = cand
        self.shapes.sort()

        self.unique_M = sorted({m for m, _, _ in self.shapes})
        self.unique_N = sorted({n for _, n, _ in self.shapes})
        self.unique_K = sorted({k for _, _, k in self.shapes})

        self._candidates_by_dtype: Dict[str, List[CandidateInfo]] = {
            dtype: list(shape_map.values()) for dtype, shape_map in self._best_by_dtype.items()
        }
        for dtype in self._candidates_by_dtype:
            self._candidates_by_dtype[dtype].sort(key=lambda c: c.shape)

    def _parse_candidate_entry(
        self, shape: Shape3D, dtype_entry: dict
    ) -> Optional[CandidateInfo]:
        best = dtype_entry.get("Best m,n,k", {})
        tile = best.get("size")
        exec_us = best.get("Best NPU average time")
        if not tile or exec_us is None:
            return None
        try:
            t = (int(tile[0]), int(tile[1]), int(tile[2]))
            e = float(exec_us)
        except (ValueError, TypeError, IndexError):
            return None
        return CandidateInfo(shape=shape, best_tile=t, execution_us=e)

    def _parse_working_tiles(self, dtype_entry: dict) -> List[TilePerf]:
        working = dtype_entry.get("Working m,n,k", {})
        if not isinstance(working, dict):
            return []
        out: List[TilePerf] = []
        for _, tile_entry in working.items():
            if not isinstance(tile_entry, dict):
                continue
            tile = tile_entry.get("size")
            exec_us = tile_entry.get("avg")
            if tile is None or exec_us is None:
                continue
            try:
                t = (int(tile[0]), int(tile[1]), int(tile[2]))
                e = float(exec_us)
            except (ValueError, TypeError, IndexError):
                continue
            out.append(TilePerf(tile=t, execution_us=e))
        return out

    def feasible_shapes(self, m0: int, n0: int, k0: int) -> List[Shape3D]:
        return [(m, n, k) for (m, n, k) in self.shapes if m >= m0 and n >= n0 and k >= k0]

    def best_candidate(self, shape: Shape3D, dtype: str) -> Optional[CandidateInfo]:
        return self._best_by_dtype.get(dtype, {}).get(shape)

    def find_best_tile_for_shape(self, shape: Shape3D, dtype: str) -> Optional[CandidateInfo]:
        """
        Find best m,n,k for one padded shape by scanning known working tiles.
        This is intentionally query-time work, so strategy timing can include it.
        """
        tile_options = self._tile_options_by_dtype.get(dtype, {}).get(shape, [])
        if not tile_options:
            return self.best_candidate(shape, dtype)
        best = min(tile_options, key=lambda x: x.execution_us)
        return CandidateInfo(shape=shape, best_tile=best.tile, execution_us=best.execution_us)

    def feasible_candidates(self, m0: int, n0: int, k0: int, dtype: str) -> List[CandidateInfo]:
        cands = self._candidates_by_dtype.get(dtype, [])
        return [c for c in cands if c.shape[0] >= m0 and c.shape[1] >= n0 and c.shape[2] >= k0]

    @staticmethod
    def nearest_larger(sorted_vals: List[int], target: int, count: int) -> List[int]:
        i = bisect_left(sorted_vals, target)
        if i >= len(sorted_vals):
            return []
        return sorted_vals[i : i + count]


class PaddingEstimator:
    """
    Pad-time estimator.

    Priority:
    1) exact lookup from padding_sweep_profile.json if provided
    2) simple analytic fallback
    """

    def __init__(self, padding_profile_path: Optional[str] = None):
        self._exact_pad: Dict[Tuple[Shape3D, str, Shape3D], float] = {}
        if padding_profile_path:
            with open(padding_profile_path, "r", encoding="utf-8") as f:
                raw = json.load(f) or {}
            # Flatten nested JSON into tuple-key map for O(1) hot-path lookup.
            for in_key, in_entry in raw.get("inputs", {}).items():
                try:
                    m0, n0, k0 = [int(x) for x in in_key.split(",")]
                except ValueError:
                    continue
                input_shape = (m0, n0, k0)
                for dtype, dtype_entry in in_entry.items():
                    if not isinstance(dtype_entry, dict):
                        continue
                    for p_key, metrics in dtype_entry.items():
                        try:
                            mp, np_, kp = [int(x) for x in p_key.split(",")]
                        except ValueError:
                            continue
                        pad = metrics.get("padding_us")
                        unpad = metrics.get("unpadding_us")
                        if pad is None or unpad is None:
                            continue
                        self._exact_pad[(input_shape, dtype, (mp, np_, kp))] = float(pad) + float(
                            unpad
                        )

    def estimate_us(self, input_shape: Shape3D, padded_shape: Shape3D, dtype: str) -> float:
        exact = self._estimate_from_db(input_shape, padded_shape, dtype)
        if exact is not None:
            return exact
        return self._estimate_analytic(input_shape, padded_shape)

    def _estimate_from_db(
        self, input_shape: Shape3D, padded_shape: Shape3D, dtype: str
    ) -> Optional[float]:
        return self._exact_pad.get((input_shape, dtype, padded_shape))

    @staticmethod
    def _estimate_analytic(input_shape: Shape3D, padded_shape: Shape3D) -> float:
        """
        Cheap fallback model.
        Assumes cost roughly scales with touched elements:
        - zero-init padded A/B/C
        - copy A/B/C-in into padded buffers
        """
        m0, n0, k0 = input_shape
        mp, np_, kp = padded_shape
        zero_elems = mp * kp + kp * np_ + mp * np_
        copy_elems = m0 * k0 + k0 * n0 + m0 * n0
        touched = zero_elems + copy_elems
        # 0.0025us per element is a placeholder for "functions-ready" mode.
        return 0.0025 * float(touched)


def _score(
    decision_us: float, padding_us: float, execution_us: float
) -> float:
    return decision_us + padding_us + execution_us


def _pick_min_by_exec(cands: Iterable[CandidateInfo]) -> CandidateInfo:
    cands = list(cands)
    if not cands:
        raise ValueError("No valid candidates to choose from.")
    return min(cands, key=lambda c: c.execution_us)


def _make_result(
    strategy: str,
    input_shape: Shape3D,
    dtype: str,
    chosen: CandidateInfo,
    decision_us: float,
    padding_us: float,
    considered: int,
) -> SearchResult:
    total_us = _score(decision_us, padding_us, chosen.execution_us)
    return SearchResult(
        strategy=strategy,
        input_shape=input_shape,
        dtype=dtype,
        chosen_shape=chosen.shape,
        chosen_tile=chosen.best_tile,
        decision_us=decision_us,
        padding_us=padding_us,
        execution_us=chosen.execution_us,
        total_us=total_us,
        num_candidates_considered=considered,
    )


def _shape_volume(shape: Shape3D) -> int:
    return shape[0] * shape[1] * shape[2]


def _dominance_prune_candidates(cands: List[CandidateInfo]) -> List[CandidateInfo]:
    kept: List[CandidateInfo] = []
    for a in cands:
        dominated = False
        for b in cands:
            if b is a:
                continue
            if (
                b.shape[0] >= a.shape[0]
                and b.shape[1] >= a.shape[1]
                and b.shape[2] >= a.shape[2]
                and b.execution_us <= a.execution_us
            ):
                dominated = True
                break
        if not dominated:
            kept.append(a)
    return kept


_DB_CACHE: Dict[str, ProfileDB] = {}
_PAD_CACHE: Dict[str, PaddingEstimator] = {}


def get_profile_db_cached(profile_path: str) -> ProfileDB:
    db = _DB_CACHE.get(profile_path)
    if db is None:
        db = ProfileDB(profile_path)
        _DB_CACHE[profile_path] = db
    return db


def get_padding_estimator_cached(padding_profile_path: Optional[str]) -> PaddingEstimator:
    key = padding_profile_path or "__none__"
    est = _PAD_CACHE.get(key)
    if est is None:
        est = PaddingEstimator(padding_profile_path)
        _PAD_CACHE[key] = est
    return est


class FastLatencySelector:
    """
    Persistent in-memory selector.
    Build once, query many times.
    """

    def __init__(
        self,
        db: ProfileDB,
        pad_estimator: PaddingEstimator,
        dtype: str,
    ):
        self.db = db
        self.pad_estimator = pad_estimator
        self.dtype = dtype
        self._all_cands = list(db._candidates_by_dtype.get(dtype, []))
        self._cands_by_shape = {c.shape: c for c in self._all_cands}
        # Sorted-first-fit is intentionally "fastest execution first", then first feasible by shape.
        self._sorted_by_exec = sorted(
            self._all_cands, key=lambda c: (c.execution_us, c.shape[0], c.shape[1], c.shape[2])
        )
        self._pre_map = build_preprocessed_fast_map(db, dtype)

    def query_hashmap_27(
        self, input_shape: Shape3D, per_dim_neighbors: int = 3
    ) -> SearchResult:
        return select_with_hashmap_27(
            self.db, self.pad_estimator, input_shape, self.dtype, per_dim_neighbors
        )

    def query_sorted_first_fit(self, input_shape: Shape3D) -> SearchResult:
        t0 = time.perf_counter()
        m0, n0, k0 = input_shape
        chosen: Optional[CandidateInfo] = None
        considered = 0
        for cand in self._sorted_by_exec:
            considered += 1
            m, n, k = cand.shape
            if m >= m0 and n >= n0 and k >= k0:
                chosen = cand
                break
        if chosen is None:
            raise ValueError("No valid candidate in fast sorted-first-fit selector.")
        decision_us = (time.perf_counter() - t0) * 1e6
        padding_us = self.pad_estimator.estimate_us(input_shape, chosen.shape, self.dtype)
        return _make_result(
            "sorted_first_fit_fast",
            input_shape,
            self.dtype,
            chosen,
            decision_us,
            padding_us,
            considered,
        )

    def query_preprocessed(self, input_shape: Shape3D) -> SearchResult:
        return select_with_preprocessed_map(
            self.db, self.pad_estimator, self._pre_map, input_shape, self.dtype
        )


def select_with_hashmap_27(
    db: ProfileDB,
    pad_estimator: PaddingEstimator,
    input_shape: Shape3D,
    dtype: str,
    per_dim_neighbors: int = 3,
) -> SearchResult:
    """
    Strategy 1: Hashmap-27 style search.
    - Take nearest larger M/N/K values (default 3 each)
    - Cross-product candidates
    - Pick candidate with best execution time in this neighborhood
    """
    t0 = time.perf_counter()
    m0, n0, k0 = input_shape
    m_vals = db.nearest_larger(db.unique_M, m0, per_dim_neighbors)
    n_vals = db.nearest_larger(db.unique_N, n0, per_dim_neighbors)
    k_vals = db.nearest_larger(db.unique_K, k0, per_dim_neighbors)
    shapes = list(itertools.product(m_vals, n_vals, k_vals))

    valid: List[CandidateInfo] = []
    for shp in shapes:
        c = db.best_candidate(shp, dtype)
        if c is not None:
            valid.append(c)
    if not valid:
        raise ValueError("No valid candidates in hashmap-27 neighborhood.")

    # Decision timing measures only shape+tile selection work.
    chosen_shape = min(
        valid,
        key=lambda c: c.execution_us,
    )
    chosen = db.best_candidate(chosen_shape.shape, dtype)
    if chosen is None:
        raise ValueError("No valid best m,n,k for chosen shape in hashmap-27.")
    decision_us = (time.perf_counter() - t0) * 1e6
    padding_us = pad_estimator.estimate_us(input_shape, chosen.shape, dtype)
    return _make_result(
        "hashmap_27", input_shape, dtype, chosen, decision_us, padding_us, len(valid)
    )


def select_with_sorted_first_fit(
    db: ProfileDB,
    pad_estimator: PaddingEstimator,
    input_shape: Shape3D,
    dtype: str,
) -> SearchResult:
    """
    Strategy 2: Sorted first-feasible.
    - Sort all profiled shapes by fastest execution time
    - Choose the first shape in that order that can fit the input
    """
    t0 = time.perf_counter()
    m0, n0, k0 = input_shape
    cands = list(db._candidates_by_dtype.get(dtype, []))
    cands.sort(key=lambda c: (c.execution_us, c.shape[0], c.shape[1], c.shape[2]))
    chosen: Optional[CandidateInfo] = None
    considered = 0
    for cand in cands:
        considered += 1
        m, n, k = cand.shape
        if m >= m0 and n >= n0 and k >= k0:
            chosen = cand
            break
    if chosen is None:
        raise ValueError("No valid candidate in sorted first-fit strategy.")

    decision_us = (time.perf_counter() - t0) * 1e6
    padding_us = pad_estimator.estimate_us(input_shape, chosen.shape, dtype)
    return _make_result(
        "sorted_first_fit", input_shape, dtype, chosen, decision_us, padding_us, considered
    )


def build_preprocessed_fast_map(
    db: ProfileDB, dtype: str, require_not_slower_than_self: bool = True
) -> Dict[Shape3D, CandidateInfo]:
    """
    Strategy 3 preprocessing helper.
    For each profiled shape S, map to the fastest dominating shape F (F >= S in all dims).
    """
    all_valid: List[CandidateInfo] = []
    for shp in db.shapes:
        cand = db.best_candidate(shp, dtype)
        if cand is not None:
            all_valid.append(cand)
    mapping: Dict[Shape3D, CandidateInfo] = {}
    for s in all_valid:
        dominates = [
            f
            for f in all_valid
            if f.shape[0] >= s.shape[0] and f.shape[1] >= s.shape[1] and f.shape[2] >= s.shape[2]
        ]
        strictly_bigger_and_faster = [
            f
            for f in dominates
            if (
                (f.shape[0] > s.shape[0] or f.shape[1] > s.shape[1] or f.shape[2] > s.shape[2])
                and f.execution_us < s.execution_us
            )
        ]
        if strictly_bigger_and_faster:
            mapping[s.shape] = _pick_min_by_exec(strictly_bigger_and_faster)
            continue
        if require_not_slower_than_self:
            dominates = [f for f in dominates if f.execution_us <= s.execution_us]
            if not dominates:
                dominates = [s]
        best = _pick_min_by_exec(dominates)
        mapping[s.shape] = best
    return mapping


def _nearest_profile_shape_ge(db: ProfileDB, input_shape: Shape3D) -> Shape3D:
    """
    Find nearest-larger profiled anchor shape by independent nearest-larger per dimension.
    Falls back to smallest feasible by volume.
    """
    m0, n0, k0 = input_shape
    m_list = db.nearest_larger(db.unique_M, m0, 1)
    n_list = db.nearest_larger(db.unique_N, n0, 1)
    k_list = db.nearest_larger(db.unique_K, k0, 1)
    if m_list and n_list and k_list:
        return (m_list[0], n_list[0], k_list[0])
    feas = db.feasible_shapes(m0, n0, k0)
    if not feas:
        raise ValueError("No feasible profile shape for this input.")
    return min(feas, key=lambda s: (s[0] * s[1] * s[2], s[0], s[1], s[2]))


def select_with_preprocessed_map(
    db: ProfileDB,
    pad_estimator: PaddingEstimator,
    preprocessed_map: Dict[Shape3D, CandidateInfo],
    input_shape: Shape3D,
    dtype: str,
) -> SearchResult:
    """
    Strategy 3 query:
    - Anchor input to nearest-larger profiled shape
    - Use preprocessed map to jump to best dominating fast shape
    """
    t0 = time.perf_counter()
    anchor = _nearest_profile_shape_ge(db, input_shape)
    chosen = preprocessed_map.get(anchor)
    if chosen is None:
        anchor_cand = db.best_candidate(anchor, dtype)
        if anchor_cand is None:
            raise ValueError("No candidate for anchor shape in preprocessed strategy.")
        chosen = anchor_cand
    chosen = db.best_candidate(chosen.shape, dtype)
    if chosen is None:
        raise ValueError("No valid best m,n,k for chosen shape in preprocessed strategy.")

    decision_us = (time.perf_counter() - t0) * 1e6
    padding_us = pad_estimator.estimate_us(input_shape, chosen.shape, dtype)
    return _make_result(
        "preprocessed_map", input_shape, dtype, chosen, decision_us, padding_us, 1
    )


def _parse_inputs(raw: str) -> List[Shape3D]:
    out: List[Shape3D] = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        m_str, n_str, k_str = [x.strip() for x in part.split(",")]
        out.append((int(m_str), int(n_str), int(k_str)))
    if not out:
        raise ValueError("No valid --inputs parsed.")
    return out


def _run_strategy_once(
    strategy: str,
    selector: FastLatencySelector,
    input_shape: Shape3D,
    per_dim_neighbors: int,
) -> SearchResult:
    if strategy == "hashmap_27":
        return selector.query_hashmap_27(input_shape=input_shape, per_dim_neighbors=per_dim_neighbors)
    if strategy == "sorted_first_fit":
        return selector.query_sorted_first_fit(input_shape=input_shape)
    if strategy == "preprocessed_map":
        return selector.query_preprocessed(input_shape=input_shape)
    raise ValueError(f"Unknown strategy: {strategy}")


def _build_per_input_payload(
    strategy: str,
    inputs: List[Shape3D],
    dtypes: List[str],
    per_strategy_dtype_results: Dict[str, dict],
) -> dict:
    by_input: Dict[Shape3D, Dict[str, dict]] = {}
    failures: List[dict] = []

    for dtype in dtypes:
        dtype_block = per_strategy_dtype_results.get(dtype, {})
        for row in dtype_block.get("results", []):
            key = tuple(int(x) for x in row["input_shape"])
            if key not in by_input:
                by_input[key] = {}
            by_input[key][dtype] = {
                "chosen_mnk": row["chosen_tile"],
                "decision_us": row["decision_us"],
            }
        for fail in dtype_block.get("failures", []):
            failures.append(
                {
                    "input_shape": fail.get("input_shape"),
                    "dtype": dtype,
                    "error": fail.get("error"),
                }
            )

    inputs_payload: Dict[str, dict] = {}
    for input_shape in inputs:
        input_key = f"{input_shape[0]},{input_shape[1]},{input_shape[2]}"
        dtype_times = {}
        for dtype in dtypes:
            dtype_times[dtype] = by_input.get(input_shape, {}).get(
                dtype,
                {"chosen_mnk": None, "decision_us": None},
            )
        inputs_payload[input_key] = dtype_times

    return {
        "strategy": strategy,
        "num_inputs": len(inputs),
        "inputs": inputs_payload,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark search decision time for selecting padded M,N,K and best m,n,k."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=STRATEGIES + ["all"],
        default="hashmap_27",
        help="Search strategy to benchmark, or 'all' for all three.",
    )
    parser.add_argument(
        "--inputs",
        type=str,
        default=None,
        help='Semicolon-separated M0,N0,K0 triplets, e.g. "300,500,700;777,801,333".',
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=DEFAULT_PROFILE,
        help="Path to combined/full profile JSON (e.g. vla_profile_combined.json).",
    )
    parser.add_argument(
        "--padding-profile",
        type=str,
        default=None,
        help="Optional path to padding_sweep_profile.json for exact pad+unpad lookup.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["all", "i8", "i16", "bf16"],
        default="all",
        help="Run one dtype or all dtypes.",
    )
    parser.add_argument(
        "--per-dim-neighbors",
        type=int,
        default=3,
        help="Neighbor count per dimension for hashmap_27.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write JSON summary.",
    )
    parser.add_argument(
        "--minimal-out",
        type=str,
        default=None,
        help=(
            "Optional path to write per-input JSON for one strategy. "
            "Each input row includes per-dtype chosen_mnk and decision_us."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Optional directory for per-strategy per-input JSON files. "
            "Each file has per-input/per-dtype chosen_mnk and decision_us."
        ),
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    profile_path = Path(args.profile) if os.path.isabs(args.profile) else (base / args.profile)
    pad_profile_path: Optional[Path]
    if args.padding_profile:
        pad_profile_path = (
            Path(args.padding_profile)
            if os.path.isabs(args.padding_profile)
            else (base / args.padding_profile)
        )
    else:
        pad_profile_path = None
    out_path = (
        None
        if not args.out
        else (Path(args.out) if os.path.isabs(args.out) else (base / args.out))
    )
    minimal_out_path = (
        None
        if not args.minimal_out
        else (
            Path(args.minimal_out)
            if os.path.isabs(args.minimal_out)
            else (base / args.minimal_out)
        )
    )
    out_dir_path = (
        None
        if not args.out_dir
        else (Path(args.out_dir) if os.path.isabs(args.out_dir) else (base / args.out_dir))
    )

    inputs = _parse_inputs(args.inputs) if args.inputs else list(DEFAULT_INPUTS)
    dtypes = DTYPES if args.dtype == "all" else [args.dtype]
    db = get_profile_db_cached(str(profile_path))
    pad_estimator = get_padding_estimator_cached(str(pad_profile_path) if pad_profile_path else None)
    strategies = STRATEGIES if args.strategy == "all" else [args.strategy]

    if minimal_out_path and len(strategies) != 1:
        raise ValueError("--minimal-out requires a single --strategy (not --strategy all).")

    summary = {
        "strategy": args.strategy if len(strategies) == 1 else "all",
        "profile_path": str(profile_path),
        "padding_profile_path": str(pad_profile_path) if pad_profile_path else None,
        "num_inputs": len(inputs),
        "dtype_results": {},
        "strategies": {},
    }

    for strategy in strategies:
        per_strategy_dtype_results = {}
        for dtype in dtypes:
            selector = FastLatencySelector(
                db=db,
                pad_estimator=pad_estimator,
                dtype=dtype,
            )
            total_decision_us = 0.0
            failures = []
            results = []

            for input_shape in inputs:
                try:
                    result = _run_strategy_once(
                        strategy=strategy,
                        selector=selector,
                        input_shape=input_shape,
                        per_dim_neighbors=args.per_dim_neighbors,
                    )
                except Exception as exc:
                    failures.append({"input_shape": list(input_shape), "error": str(exc)})
                    continue

                total_decision_us += result.decision_us
                result_row = {
                    "input_shape": list(result.input_shape),
                    "chosen_shape": list(result.chosen_shape),
                    "chosen_tile": list(result.chosen_tile),
                    "decision_us": result.decision_us,
                    "num_candidates_considered": result.num_candidates_considered,
                }
                results.append(result_row)

            num_success = len(results)
            avg_decision_us = (total_decision_us / num_success) if num_success else None
            per_strategy_dtype_results[dtype] = {
                "num_success": num_success,
                "num_failures": len(failures),
                "total_decision_us": total_decision_us,
                "total_decision_ms": total_decision_us / 1000.0,
                "avg_decision_us": avg_decision_us,
                "results": results,
                "failures": failures,
            }
            print(
                f"[{dtype}] strategy={strategy} "
                f"inputs={len(inputs)} success={num_success} failures={len(failures)} "
                f"total_decision_us={total_decision_us:.3f} "
                f"total_decision_ms={total_decision_us/1000.0:.6f}"
            )

        if len(strategies) == 1:
            summary["dtype_results"] = per_strategy_dtype_results
        summary["strategies"][strategy] = {"dtype_results": per_strategy_dtype_results}

        per_input_payload = _build_per_input_payload(
            strategy=strategy,
            inputs=inputs,
            dtypes=dtypes,
            per_strategy_dtype_results=per_strategy_dtype_results,
        )

        if out_dir_path:
            out_dir_path.mkdir(parents=True, exist_ok=True)
            minimal_path = out_dir_path / f"{strategy}.json"
            with minimal_path.open("w", encoding="utf-8") as f:
                json.dump(per_input_payload, f, indent=2)
            print(f"[DONE] Wrote minimal strategy file to {minimal_path}")
        if minimal_out_path:
            minimal_out_path.parent.mkdir(parents=True, exist_ok=True)
            with minimal_out_path.open("w", encoding="utf-8") as f:
                json.dump(per_input_payload, f, indent=2)
            print(f"[DONE] Wrote minimal strategy file to {minimal_out_path}")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[DONE] Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
