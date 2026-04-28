import csv
import json
from collections import defaultdict
from pathlib import Path

def is_number(s):
    try:
        float(s)
        return True
    except:
        return False

# Data structure keyed by "M_N_K"
data = defaultdict(lambda: {
    'i8': defaultdict(list),
    'i16': defaultdict(list),
    'bf16': defaultdict(list),
    'i8_fail': set(),
    'i16_fail': set(),
    'bf16_fail': set()
})

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "vla_profile.csv"
with csv_path.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    # Strip whitespace from header names
    reader.fieldnames = [name.strip() for name in reader.fieldnames]
    for raw_row in reader:
        # Normalize keys, ensure no None values
        row = {k.strip(): (v or '').strip() for k, v in raw_row.items()}
        M = row.get('M', '')
        N = row.get('N', '')
        K = row.get('K', '')
        if not (M and N and K):
            continue
        key = f"{M}_{N}_{K}"
        try:
            m = int(row.get('m', '0'))
            n = int(row.get('n', '0'))
            k = int(row.get('k', '0'))
        except ValueError:
            continue

        for dtype, col in [('i8', 'i8_avg'), ('i16', 'i16_avg'), ('bf16', 'bf16_avg')]:
            val = row.get(col, '')
            if is_number(val):
                data[key][dtype][(m, n, k)].append(float(val))
            else:
                data[key][f"{dtype}_fail"].add((m, n, k))

# Build final JSON structure
final = {}
for key, metrics in data.items():
    final[key] = {}
    for dtype in ['i8', 'i16', 'bf16']:
        groups = metrics[dtype]
        # Compute average for each (m,n,k)
        work_groups = {mnk: sum(vals) / len(vals) for mnk, vals in groups.items() if vals}
        working_list = list(work_groups.keys())
        if work_groups:
            best_mnk = min(work_groups, key=work_groups.get)
            best_time = work_groups[best_mnk]
        else:
            best_mnk = None
            best_time = None

        failing_list = list(metrics[f"{dtype}_fail"])
        final[key][dtype] = {
            "Best m,n,k": {
                "size": list(best_mnk) if best_mnk else None,
                "Best NPU average time": best_time
            },
            "Working m,n,k": [list(x) for x in working_list],
            "Failing m,n,k": [list(x) for x in failing_list]
        }

# Write JSON to file
out_path = BASE_DIR.parent / "profiling" / "data" / "vla_profile.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open('w', encoding='utf-8') as jf:
    json.dump(final, jf, indent=2)

print(f"Wrote JSON data to {out_path}")
