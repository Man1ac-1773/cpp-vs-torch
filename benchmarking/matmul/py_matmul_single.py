import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import numpy as np
import time
import json
import sys

# Ensure torch is single threaded
torch.set_num_threads(1)

def run_single_thread_sweep(filepath):
    print(" ========== Starting Single-threaded PyTorch & Numpy Benchmarking ========= ")
    step = 10
    NUM_RUNS = 10
    N = 10
    
    with open(filepath, "w") as f:
        while N <= 2000:
            # --- PyTorch ---
            A_pt = torch.ones((N, N), dtype=torch.float32)
            B_pt = torch.ones((N, N), dtype=torch.float32)
            C_warmup = torch.matmul(A_pt, B_pt)
            
            total_time = 0.0
            min_time = 1e9
            for _ in range(NUM_RUNS):
                start = time.time()
                C = torch.matmul(A_pt, B_pt)
                elapsed = time.time() - start
                total_time += elapsed
                min_time = min(min_time, elapsed)
            
            f.write(json.dumps({
                "benchmark": "matmul", "N": N, "kernel": "aten", 
                "lang": "pytorch", "threads": 1, 
                "avg_time": total_time/NUM_RUNS, "min_time": min_time
            }) + "\n")
            
            # --- Numpy ---
            A_np = np.ones((N, N), dtype=np.float32)
            B_np = np.ones((N, N), dtype=np.float32)
            C_warmup_np = np.matmul(A_np, B_np)
            
            total_time = 0.0
            min_time = 1e9
            for _ in range(NUM_RUNS):
                start = time.time()
                C = np.matmul(A_np, B_np)
                elapsed = time.time() - start
                total_time += elapsed
                min_time = min(min_time, elapsed)
            
            f.write(json.dumps({
                "benchmark": "matmul", "N": N, "kernel": "openblas/mkl", 
                "lang": "numpy", "threads": 1, 
                "avg_time": total_time/NUM_RUNS, "min_time": min_time
            }) + "\n")

            if step == 10 and N >= 100: step = 100
            if step == 100 and N >= 1000: step = 1000
            N += step

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "performance-plugged"
    filepath = f"./data/py_single_{mode}.jsonl"
    run_single_thread_sweep(filepath)
    print(f"Finished! Results saved to {filepath}")
