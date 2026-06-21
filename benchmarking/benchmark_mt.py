import os
import time
import json
import argparse
import numpy as np
import torch

def run_python_mt_sweeps(out_path):
    print(" ========== Starting Python Multithreaded Benchmarking ========= ")
    step = 10
    NUM_RUNS = 10
    
    with open(out_path, "w") as f:
        N = 10
        while N <= 2000:
            # 1. NumPy MT Benchmark
            A_np = np.ones((N, N), dtype=np.float32)
            B_np = np.ones((N, N), dtype=np.float32) * 2.0
            
            # Warmup
            _ = np.matmul(A_np, B_np)
            
            total_time_np = 0.0
            min_time_np = float('inf')
            
            for r in range(NUM_RUNS):
                start = time.perf_counter()
                _ = np.matmul(A_np, B_np)
                elapsed = time.perf_counter() - start
                total_time_np += elapsed
                if elapsed < min_time_np: min_time_np = elapsed
                
            avg_time_np = total_time_np / NUM_RUNS
            f.write(json.dumps({
                "benchmark": "matmul_mt", "N": N, "kernel": "simd_mt", "lang": "numpy",
                "avg_time": avg_time_np, "min_time": min_time_np
            }) + "\n")
            
            # 2. PyTorch MT Benchmark
            A_pt = torch.ones((N, N), dtype=torch.float32)
            B_pt = torch.ones((N, N), dtype=torch.float32) * 2.0
            
            # Warmup
            _ = torch.matmul(A_pt, B_pt)
            
            total_time_pt = 0.0
            min_time_pt = float('inf')
            
            for r in range(NUM_RUNS):
                start = time.perf_counter()
                _ = torch.matmul(A_pt, B_pt)
                elapsed = time.perf_counter() - start
                total_time_pt += elapsed
                if elapsed < min_time_pt: min_time_pt = elapsed
                
            avg_time_pt = total_time_pt / NUM_RUNS
            f.write(json.dumps({
                "benchmark": "matmul_mt", "N": N, "kernel": "simd_mt", "lang": "pytorch",
                "avg_time": avg_time_pt, "min_time": min_time_pt
            }) + "\n")
            
            f.flush()
            
            if step == 10 and N >= 100:
                step = 100
                print("N stepped up to 100")
            elif step == 100 and N >= 1000:
                step = 1000
                print("N stepped up to 1000")
                
            if N == 2000:
                break
            N += step

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="./py-data/mt_benchmark_results.jsonl")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    run_python_mt_sweeps(args.out)
    print("Done!")
