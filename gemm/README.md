# GEMM Layout

This folder is organized by workflow:

- `experiments/padding_policy/`: active padding-policy experiment harnesses and organized run outputs.
- `scripts/`: core runtime/search scripts.
  - `v2_test_mapping_large_gemm.py`
  - `latency_search_strategies.py`
- `profiling/`: profiling runners and profile builders.
  - `data/`: generated profile JSON artifacts.
- `padding/`: padding sweep and padding analysis scripts.
  - `plots/`: generated padding plot images.
- `plots/`: general plotting scripts and generated non-padding plot images.
- `archive/`: older conversion scripts and archived datasets.
- `trials/`: trial profile/error snapshots.
- `top.prj/`, `llama/`: AIE project artifacts.
