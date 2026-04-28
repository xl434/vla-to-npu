# Padding Policy Experiments

Minimal workflow for:

1. Padding-only timing sweeps (`manual_copy` vs `numpy_pad`).
2. Fitting per-dtype transition threshold (when to switch pad implementation).
3. Running baseline policies (Option 1/2/3) with end-to-end timing.

## Active scripts

- `run_padding_only_async.py`
  - Pure padding benchmark (CPU/GPU host only, no NPU run).
- `fit_transition_threshold.py`
  - Builds merged DB + threshold JSON from padding sweep outputs.
- `run_baseline_policies_async.py`
  - Runs Option 1/2/3 and writes friendly `results.json`.
- `plot_padding_transition_lines.py`
  - Visualizes manual vs numpy padding timing and fitted lines.
- `plot_baseline1_time_chunks.py`
  - Baseline timing chunk plots from baseline results JSON.

## Typical workflow

1. Padding-only sweep:

```bash
cd /home/ec935/vla-to-npu/gemm/experiments/padding_policy
python run_padding_only_async.py \
  --include-default-inputs \
  --random-count 1000 \
  --dtypes i8,i16,bf16 \
  --repeats 20 \
  --warmup 5 \
  --jobs 1
```

2. Fit transition DB:

```bash
python fit_transition_threshold.py \
  --manual-dir runs/padding/padding_only_primary/padding_sweeps \
  --numpy-dir runs/padding/padding_only_primary/padding_sweeps \
  --out-dir runs/padding/analysis_latest
```

3. Run baselines (Option 3 included automatically):

```bash
python run_baseline_policies_async.py \
  --include-default-inputs \
  --random-count 100 \
  --dtypes i8,i16,bf16 \
  --jobs 1
```

## Clean directory layout

- `runs/padding/padding_only_primary/`
  - Canonical padding-only dataset.
- `runs/padding/analysis_primary/`
  - Canonical transition DB + thresholds.
- `runs/baseline/baseline1/baseline1_primary/`
  - Canonical baseline1 output (`manifest.json`, `results.json`).
- `runs/baseline/baseline3/`
  - Baseline3 run outputs (`baseline3_<timestamp>/...`).
