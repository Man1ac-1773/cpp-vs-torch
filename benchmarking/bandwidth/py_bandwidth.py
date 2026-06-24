import torch
import sys
import time

if len(sys.argv) != 2:
    print("Usage: python py_bandwidth.py <N>")
    sys.exit(1)

N = int(sys.argv[1])

A = torch.ones((N, N), dtype=torch.float32)
B = torch.ones((N, N), dtype=torch.float32)

_ = torch.matmul(A, B)

start = time.time()
C = torch.matmul(A, B)
end = time.time()

print(f"{end - start}")
