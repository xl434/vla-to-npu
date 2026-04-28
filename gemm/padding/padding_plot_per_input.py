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
def load_overhead_data(filepath):
    if not filepath.exists():
        print(f"Error: File '{filepath}' not found.")
        return pd.DataFrame()

    with open(filepath, 'r') as f:
        data = json.load(f)

    records = []
    if "inputs" not in data: 
        print("Error: 'inputs' key missing in JSON.")
        return pd.DataFrame()

    for orig_dims, dtype_dict in data["inputs"].items():
        for dtype, padded_configs in dtype_dict.items():
            for padded_dims, metrics in padded_configs.items():
                
                # We ONLY care about the Padding and Unpadding time
                pad_us = metrics.get('padding_us', 0)
                unpad_us = metrics.get('unpadding_us', 0)
                total_overhead = pad_us + unpad_us
                
                records.append({
                    'Input': orig_dims,  
                    'Strategy': padded_dims, 
                    'Padding': pad_us,
                    'Unpadding': unpad_us,
                    'Total Overhead': total_overhead
                })

    return pd.DataFrame(records)

df = load_overhead_data(INPUT_FILE)

if df.empty:
    print("No data loaded. Exiting.")
    raise SystemExit(1)

# -----------------------------
# 3. Visualization: The "Overhead Race"
# -----------------------------
unique_inputs = df['Input'].unique()

for ui in unique_inputs:
    # Filter for this specific input and sort by Total Overhead (lowest to highest)
    df_subset = df[df['Input'] == ui].sort_values('Total Overhead')
    
    # Take the top 15 padding strategies to avoid overcrowding the x-axis
    df_subset = df_subset.head(15) 

    # Prepare Stacked Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df_subset))
    width = 0.6
    
    # Stack Padding (Bottom) and Unpadding (Top)
    p1 = ax.bar(x, df_subset['Padding'], width, label='Padding Time', color='#e67e22', edgecolor='white')
    p2 = ax.bar(x, df_subset['Unpadding'], width, bottom=df_subset['Padding'], label='Unpadding Time', color='#2ecc71', edgecolor='white')
    
    ax.set_title(f'Padding Overhead for Input: {ui}\n(Ranked Fastest to Slowest)', fontsize=14, weight='bold')
    ax.set_xlabel('Padding Strategy (Target Dimensions)', fontsize=12)
    ax.set_ylabel('Overhead Time (us)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df_subset['Strategy'], rotation=45, ha='right', fontsize=10)
    ax.legend()
    
    # Add text labels on top of the bars showing the total overhead
    for i, v in enumerate(df_subset['Total Overhead']):
        ax.text(i, v + 0.2, f"{v:.1f}", ha='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    
    # Save image
    safe_filename = ui.replace(',', '_')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / f"padding_overhead_only_{safe_filename}.png", dpi=300)
    print(f"Saved padding_overhead_only_{safe_filename}.png")
    plt.close()
