import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import json
import os

def get_rapl_energy_joules():
    try:
        with open("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", "r") as f:
            return float(f.read().strip()) / 1e6
    except Exception as e:
        print("Failed to read RAPL:", e)
        return 0.0

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "performance-plugged"
    filepath = f"./data/py_train_{mode}.jsonl"

    BATCH_SIZE = 1024
    INPUT_DIM = 128
    HIDDEN_DIM = 256
    OUTPUT_DIM = 64
    EPOCHS = 100
    LR = 0.01

    print(f"Training 2-Layer MLP on PyTorch Engine (Batch={BATCH_SIZE})...")

    # Use uniform random to break symmetry
    X = (torch.rand((BATCH_SIZE, INPUT_DIM)) * 2.0) - 1.0
    Y = (torch.rand((BATCH_SIZE, OUTPUT_DIM)) * 2.0) - 1.0

    model = nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN_DIM, bias=False),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, OUTPUT_DIM, bias=False)
    )

    # Manual weight init to match C/C++ magnitude
    with torch.no_grad():
        model[0].weight.uniform_(-0.1, 0.1)
        model[2].weight.uniform_(-0.1, 0.1)

    # PyTorch's Linear applies matmul as X @ W^T, our C engines do X @ W.
    # To be mathematically identical, we don't care, it's just dummy sweeps.

    optimizer = optim.SGD(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    total_time = 0.0
    start_energy = get_rapl_energy_joules()

    for epoch in range(EPOCHS):
        start_time = time.time()

        pred = model(X)
        loss = criterion(pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_time = time.time() - start_time
        total_time += epoch_time

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f} | Time: {epoch_time:.6f}s")

    end_energy = get_rapl_energy_joules()
    energy_joules = end_energy - start_energy

    print(f"Total Training Time (PyTorch): {total_time:.4f}s | Energy: {energy_joules:.4f}J")

    with open(filepath, "a") as f:
        f.write(json.dumps({
            "task": "dummy_mlp",
            "engine": "pytorch",
            "backend": "aten",
            "total_time": total_time,
            "avg_epoch_time": total_time / EPOCHS,
            "energy_joules": energy_joules
        }) + "\n")

if __name__ == "__main__":
    main()
