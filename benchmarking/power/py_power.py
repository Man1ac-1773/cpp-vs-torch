import torch
import numpy as np
import time
import json
import os
import sys

def get_rapl_energy_joules():
    try:
        with open("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", "r") as f:
            return float(f.read().strip()) / 1e6
    except Exception as e:
        print("Failed to read RAPL:", e)
        return 0.0

def power_bench_py(N):
    A = torch.ones((N, N), dtype=torch.float32)
    B = torch.ones((N, N), dtype=torch.float32)
    
    iterations = 1000 if N < 100 else (50 if N < 500 else 5)
    
    start_energy = get_rapl_energy_joules()
    for _ in range(iterations):
        C = torch.matmul(A, B)
    end_energy = get_rapl_energy_joules()
    
    return (end_energy - start_energy) / iterations

def power_bench_numpy(N):
    A = np.ones((N, N), dtype=np.float32)
    B = np.ones((N, N), dtype=np.float32)
    
    iterations = 1000 if N < 100 else (50 if N < 500 else 5)
    
    start_energy = get_rapl_energy_joules()
    for _ in range(iterations):
        C = np.matmul(A, B)
    end_energy = get_rapl_energy_joules()
    
    return (end_energy - start_energy) / iterations

def run_mnist_power_bench(f):
    print("Running Power Benchmark for PyTorch MNIST (5 Epochs)...")
    INPUT_DIM = 784
    HIDDEN_DIM = 256
    OUTPUT_DIM = 10
    BATCH_SIZE = 60000
    
    X_np = np.fromfile("../../resources/data/train_images.bin", dtype=np.float32).reshape(BATCH_SIZE, INPUT_DIM)
    Y_np = np.fromfile("../../resources/data/train_labels.bin", dtype=np.float32).reshape(BATCH_SIZE, OUTPUT_DIM)
    
    X = torch.tensor(X_np, dtype=torch.float32)
    Y = torch.tensor(Y_np, dtype=torch.float32)
    
    W1 = torch.full((INPUT_DIM, HIDDEN_DIM), 0.01, requires_grad=True)
    W2 = torch.full((HIDDEN_DIM, OUTPUT_DIM), -0.01, requires_grad=True)
    
    optimizer = torch.optim.SGD([W1, W2], lr=0.01)
    loss_fn = torch.nn.MSELoss()
    
    start_energy = get_rapl_energy_joules()
    for epoch in range(5):
        h1 = torch.matmul(X, W1)
        a1 = torch.relu(h1)
        pred = torch.matmul(a1, W2)
        loss = loss_fn(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end_energy = get_rapl_energy_joules()
    
    joules = (end_energy - start_energy) / 5.0
    print(f"PyTorch MNIST Energy: {joules} Joules")
    f.write(json.dumps({"task": "mnist_epoch", "engine": "pytorch", "energy_joules": joules}) + "\n")

mode = sys.argv[1] if len(sys.argv) > 1 else "performance_plugged"
filepath = f"./data/py_{mode}.jsonl"

print(f"Sweeping Power metrics for PyTorch and Numpy ({mode}) (N=10 to N=1000)...")
with open(filepath, "a") as f:
    step = 10
    N = 10
    while N <= 1000:
        py_energy = power_bench_py(N)
        numpy_energy = power_bench_numpy(N)
        f.write(json.dumps({"task": "matmul", "N": N, "engine": "pytorch", "energy_joules": py_energy}) + "\n")
        f.write(json.dumps({"task": "matmul", "N": N, "engine": "numpy", "energy_joules": numpy_energy}) + "\n")
        if step == 10 and N >= 100:
            step = 100
        N += step
        
    run_mnist_power_bench(f)

print(f"Done! Results saved to {filepath}")
