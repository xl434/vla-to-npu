import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# -----------------------------
# 1. Configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "padding_sweep_profile.json"
OUTPUT_DIR = BASE_DIR / "plots"
sns.set_theme(style="whitegrid")

# -----------------------------
# 2. Data Loading
# -----------------------------
def load_data(filepath):
    if not filepath.exists():
        print("File not found.")
        return pd.DataFrame()

    with open(filepath, 'r') as f:
        data = json.load(f)

    records = []
    if "inputs" in data:
        for orig_dims, dtype_dict in data["inputs"].items():
            for dtype, padded_configs in dtype_dict.items():
                for padded_dims, metrics in padded_configs.items():
                    p_m, p_n, p_k = map(int, padded_dims.split(','))
                    records.append({
                        'Input': orig_dims,
                        'p_m': p_m, 'p_n': p_n, 'p_k': p_k,
                        'padding_us': metrics.get('padding_us', 0)
                    })
    return pd.DataFrame(records)

df = load_data(INPUT_FILE)
if df.empty:
    print("No data loaded. Exiting.")
    raise SystemExit(1)

# -----------------------------
# 3. Visualization: Isolation Slices
# -----------------------------
# We only look at the first input to keep it clean
unique_inputs = df['Input'].unique()
if len(unique_inputs) > 0:
    target_input = unique_inputs[0]
    df_sub = df[df['Input'] == target_input]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    # --- Slice 1: Vary K (Fix M, N) ---
    # We need to find a pair of (M, N) that has multiple K values
    common_mn = df_sub.groupby(['p_m', 'p_n']).size().nlargest(3).index
    
    for (m, n) in common_mn:
        slice_data = df_sub[(df_sub['p_m'] == m) & (df_sub['p_n'] == n)].sort_values('p_k')
        if len(slice_data) > 1:
            axes[0].plot(slice_data['p_k'], slice_data['padding_us'], marker='o', label=f'M={m}, N={n}')
    
    axes[0].set_title('Effect of Varying K (Rows of B / Cols of A)', fontsize=12, weight='bold')
    axes[0].set_xlabel('K Dimension', fontsize=10)
    axes[0].set_ylabel('Padding Time (us)', fontsize=10)
    axes[0].legend()
    axes[0].grid(True)

    # --- Slice 2: Vary M (Fix N, K) ---
    common_nk = df_sub.groupby(['p_n', 'p_k']).size().nlargest(3).index
    
    for (n, k) in common_nk:
        slice_data = df_sub[(df_sub['p_n'] == n) & (df_sub['p_k'] == k)].sort_values('p_m')
        if len(slice_data) > 1:
            axes[1].plot(slice_data['p_m'], slice_data['padding_us'], marker='o', label=f'N={n}, K={k}')

    axes[1].set_title('Effect of Varying M (Rows of A)', fontsize=12, weight='bold')
    axes[1].set_xlabel('M Dimension', fontsize=10)
    axes[1].legend()
    axes[1].grid(True)

    # --- Slice 3: Vary N (Fix M, K) ---
    common_mk = df_sub.groupby(['p_m', 'p_k']).size().nlargest(3).index
    
    for (m, k) in common_mk:
        slice_data = df_sub[(df_sub['p_m'] == m) & (df_sub['p_k'] == k)].sort_values('p_n')
        if len(slice_data) > 1:
            axes[2].plot(slice_data['p_n'], slice_data['padding_us'], marker='o', label=f'M={m}, K={k}')

    axes[2].set_title('Effect of Varying N (Cols of B)', fontsize=12, weight='bold')
    axes[2].set_xlabel('N Dimension', fontsize=10)
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle(f'Isolation Analysis for Input: {target_input}', fontsize=16)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "padding_isolation_analysis.png")
    print("Saved 'padding_isolation_analysis.png'")
