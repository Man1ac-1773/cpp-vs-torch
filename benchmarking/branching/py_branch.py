import torch
import sys
import time
import ctypes
import os
import json

if len(sys.argv) != 2:
    print("Usage: python py_branch.py <N>")
    sys.exit(1)

N = int(sys.argv[1])

A = torch.ones((N, N), dtype=torch.float32)
B = torch.ones((N, N), dtype=torch.float32)

_ = torch.matmul(A, B)

# Load the C wrapper
lib = ctypes.CDLL("./libperf.so")
lib.perf_init_branches.restype = ctypes.c_void_p
lib.perf_init_misses.restype = ctypes.c_void_p
lib.perf_start_c.argtypes = [ctypes.c_void_p]
lib.perf_read_c.argtypes = [ctypes.c_void_p]
lib.perf_read_c.restype = ctypes.c_longlong

pc_branches = lib.perf_init_branches()
pc_misses = lib.perf_init_misses()

start = time.time()

lib.perf_start_c(pc_branches)
lib.perf_start_c(pc_misses)

C = torch.matmul(A, B)

branches = lib.perf_read_c(pc_branches)
misses = lib.perf_read_c(pc_misses)

end = time.time()

miss_rate = (misses / branches * 100.0) if branches > 0 else 0.0

print(json.dumps({
    "N": N,
    "backend": "py",
    "branches": branches,
    "branch_misses": misses,
    "miss_rate_percent": miss_rate,
    "time": end - start
}))
