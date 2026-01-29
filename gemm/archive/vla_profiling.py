import ast
import subprocess
import re
import os
import json
import argparse


def get_tile_combinations(M, N, K):
   pass


def run_experiment(script_path, M, N, K, m, n, k):
   # Run the real test script and capture its entire output
   env = os.environ.copy()
   cmd = [
       "python3", script_path,
       "--M", str(M), "--N", str(N), "--K", str(K),
       "--m", str(m), "--n", str(n), "--k", str(k),
   ]
   proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
   out_bytes, _ = proc.communicate()
   output = out_bytes.decode("utf-8", errors="ignore")
   print(output, end="")
   return output


def extract_timings(raw):
   # returns list of three average times
   avg = re.findall(r"Avg NPU execution time:\s*([\d\.]+)us", raw)
   if len(avg) < 3:
       raise ValueError("Failed to parse timings")
   return [float(x) for x in avg]


def update_json(base, M, N, K, m, n, k, dtype, avg_time, error_type=None):
   path = os.path.join(base, "vla_profile.json")
   try:
       with open(path) as jf:
           profile = json.load(jf) or {}
   except (FileNotFoundError, json.JSONDecodeError):
       profile = {}
   key = f"{M}_{N}_{K}"
   # Prepopulate new key
   if key not in profile:
       profile[key] = {
           dt: {
               "Best m,n,k": {"size": None, "Best NPU average time": None},
               "Working m,n,k": {},
               "Failing m,n,k": []
           } for dt in ["i8","i16","bf16"]
       }
   sect = profile[key][dtype]
   size = [m, n, k]
   # record failure or success
   if error_type or avg_time is None:
       if size not in sect["Failing m,n,k"]:
           sect["Failing m,n,k"].append(size)
   else:
       # update running average for this tile size
       ws = sect["Working m,n,k"]
       tile_key = f"{m}_{n}_{k}"
       if tile_key not in ws:
           ws[tile_key] = {"size": size, "avg": avg_time, "count": 1}
       else:
           entry = ws[tile_key]
           entry["count"] += 1
           entry["avg"] = (entry["avg"] * (entry["count"] - 1) + avg_time) / entry["count"]
       # recompute best across all tile averages
       best = sect["Best m,n,k"]
       min_key, min_entry = min(ws.items(), key=lambda item: item[1]["avg"])
       best["size"] = min_entry["size"]
       best["Best NPU average time"] = min_entry["avg"]
   with open(path, "w") as jf:
       json.dump(profile, jf, indent=2)


def main():
   parser = argparse.ArgumentParser(description="Run VLA profiling for a single GEMM configuration")
   parser.add_argument("--M", type=int, required=True)
   parser.add_argument("--N", type=int, required=True)
   parser.add_argument("--K", type=int, required=True)
   parser.add_argument("--m", type=int, required=True)
   parser.add_argument("--n", type=int, required=True)
   parser.add_argument("--k", type=int, required=True)
   args = parser.parse_args()
   base = os.path.dirname(__file__)
   script = os.path.join(base, "v2_test_mapping_large_gemm.py")
   raw = run_experiment(script, args.M, args.N, args.K, args.m, args.n, args.k)
   try:
       avgs = extract_timings(raw)
       error = None
   except Exception as e:
       avgs = [None, None, None]
       error = str(e)
   for dtype, avg in zip(["i8","i16","bf16"], avgs):
       update_json(base, args.M, args.N, args.K, args.m, args.n, args.k, dtype, avg, error)


if __name__ == "__main__":
   main()
