import torch
import sys
import time

if len(sys.argv) != 3:
    print("Usage: python py_scaling.py <N> <THREADS>")
    sys.exit(1)

N = int(sys.argv[1])
THREADS = int(sys.argv[2])

torch.set_num_threads(THREADS)

A = torch.ones((N, N), dtype=torch.float32)
B = torch.ones((N, N), dtype=torch.float32)

_ = torch.matmul(A, B)

start = time.time()
C = torch.matmul(A, B)
end = time.time()

print(f"{end - start}")
