import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

# Load the json data
file_path = 'profile_M1024.json'
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract data
extracted_data = []

for key, content in data.items():
    # Key format: M_N_K (e.g., "1024_32_64")
    parts = key.split('_')
    if len(parts) != 3:
        continue
    
    try:
        m = int(parts[0])
        n = int(parts[1])
        k = int(parts[2])
    except ValueError:
        continue
        
    # Filter for M=1024
    if m != 1024:
        continue
    
    for dtype in ['i8', 'i16', 'bf16']:
        if dtype in content:
            best_stats = content[dtype].get('Best m,n,k', {})
            time_val = best_stats.get('Best NPU average time')
            
            if time_val is not None:
                extracted_data.append({
                    'N': n,
                    'K': k,
                    'dtype': dtype,
                    'Time': time_val
                })

df = pd.DataFrame(extracted_data)

# Setup for plots
sns.set_style("whitegrid")
dtypes = ['i8', 'i16', 'bf16']

# Function to create plots
def create_plots(df, x_col, hue_col, dtypes, filename_prefix):
    for dtype in dtypes:
        plt.figure(figsize=(10, 12)) # Tall figure size
        
        subset = df[df['dtype'] == dtype].copy()
        
        if subset.empty:
            continue
            
        # Sort values for smooth lines
        subset = subset.sort_values(by=[x_col, hue_col])
        
        # Ensure Hue is sorted for legend
        hue_order = sorted(subset[hue_col].unique())
        
        sns.lineplot(
            data=subset, 
            x=x_col, 
            y='Time', 
            hue=hue_col, 
            marker='o',
            palette='Paired', 
            hue_order=hue_order
        )
        
        # Log scale for x axis (Base 2)
        plt.xscale('log', base=2)
        
        # Formatting X ticks to show specific values (32, 64, etc.)
        unique_xs = sorted(subset[x_col].unique())
        plt.xticks(unique_xs, unique_xs, rotation=90)
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        
        plt.title(f'Best NPU Average Time vs {x_col} (M=1024, {hue_col} varies) - {dtype}')
        plt.ylabel('Best NPU Average Time (ms)')
        plt.xlabel(f'{x_col} (Log Scale)')
        
        # Legend outside
        plt.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save and show
        plt.savefig(f'plots/{filename_prefix}_{dtype}.png')
        plt.show()

# 1. Plots with N on x-axis (K is the curve)
create_plots(df, x_col='N', hue_col='K', dtypes=dtypes, filename_prefix='plot_n')

# 2. Plots with K on x-axis (N is the curve)
create_plots(df, x_col='K', hue_col='N', dtypes=dtypes, filename_prefix='plot_k')