import torch
import numpy as np
import time
import json
import sys
import multiprocessing

def run_multi_thread_sweep(filepath):
    # pytorch and numpy will use all available cores by default.
    threads = multiprocessing.cpu_count()
    print(f" ========== Starting Multi-threaded PyTorch & Numpy Benchmarking ({threads} threads) ========= ")
    step = 10
    NUM_RUNS = 10
    N = 10
    
    with open(filepath, "w") as f:
        while N <= 2000:
            # --- pytorch ---
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
                "lang": "pytorch", "threads": threads, 
                "avg_time": total_time/NUM_RUNS, "min_time": min_time,
                "avg_cycles": 0, "min_cycles": 0
            }) + "\n")
            
            # --- numpy ---
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
                "lang": "numpy", "threads": threads, 
                "avg_time": total_time/NUM_RUNS, "min_time": min_time,
                "avg_cycles": 0, "min_cycles": 0
            }) + "\n")

            if step == 10 and N >= 100: step = 100
            if step == 100 and N >= 1000: step = 1000
            N += step

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "performance-plugged"
    filepath = f"./data/py_multi_{mode}.jsonl"
    run_multi_thread_sweep(filepath)
    print(f"Finished! Results saved to {filepath}")
