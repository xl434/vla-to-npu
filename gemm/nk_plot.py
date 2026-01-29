import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the json data
file_path = 'profile_NK.json'
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract data into a list of dictionaries
extracted_data = []

for key, content in data.items():
    # Parse M, N, K from the key (e.g., "32_64_64")
    parts = key.split('_')
    if len(parts) != 3:
        continue
    
    m = int(parts[0])
    n = int(parts[1])
    k = int(parts[2])
    
    # Iterate through data types
    for dtype in ['i8', 'i16', 'bf16']:
        if dtype in content:
            best_stats = content[dtype].get('Best m,n,k', {})
            time_val = best_stats.get('Best NPU average time')
            
            # Only add valid numerical data
            if time_val is not None:
                extracted_data.append({
                    'M': m,
                    'N': n,
                    'dtype': dtype,
                    'Time': time_val
                })

# Create DataFrame
df = pd.DataFrame(extracted_data)

# Set up the plotting style
sns.set_style("whitegrid")

# List of data types to plot
dtypes = ['i8', 'i16', 'bf16']

for dtype in dtypes:
    # Increased height to 12 inches
    plt.figure(figsize=(10, 12))
    
    # Filter data for current dtype
    subset = df[df['dtype'] == dtype].sort_values(by=['M', 'N'])
    
    # Plot curves
    sns.lineplot(
        data=subset, 
        x='M', 
        y='Time', 
        hue='N', 
        marker='o',
        palette='Paired'
    )
    
    # Ensure X-axis ticks match the M values exactly
    plt.xscale('log', base=2)
    unique_ms = sorted(df['M'].unique())
    plt.xticks(unique_ms, unique_ms, rotation = 90)
    
    plt.title(f'Best NPU Average Time vs M (Data Type: {dtype})')
    plt.xlabel('M')
    plt.ylabel('Best NPU Average Time (ms)')
    
    # Place legend outside
    plt.legend(title='N (N=K)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save/Show the plot
    plt.savefig(f'plots/n_k_{dtype}.png')
    plt.show()