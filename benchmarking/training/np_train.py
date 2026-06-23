import numpy as np
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
    filepath = f"./data/np_train_{mode}.jsonl"

    BATCH_SIZE = 1024
    INPUT_DIM = 128
    HIDDEN_DIM = 256
    OUTPUT_DIM = 64
    EPOCHS = 100
    LR = 0.01

    print(f"Training 2-Layer MLP on Numpy Engine (Batch={BATCH_SIZE})...")

    # Use uniform random
    np.random.seed(42)
    X = (np.random.rand(BATCH_SIZE, INPUT_DIM).astype(np.float32) * 2.0) - 1.0
    Y = (np.random.rand(BATCH_SIZE, OUTPUT_DIM).astype(np.float32) * 2.0) - 1.0

    W1 = (np.random.rand(INPUT_DIM, HIDDEN_DIM).astype(np.float32) * 0.2) - 0.1
    W2 = (np.random.rand(HIDDEN_DIM, OUTPUT_DIM).astype(np.float32) * 0.2) - 0.1

    total_time = 0.0
    start_energy = get_rapl_energy_joules()

    for epoch in range(EPOCHS):
        start_time = time.time()

        # Forward Pass
        h1 = np.matmul(X, W1)
        a1 = np.maximum(0, h1) # ReLU
        pred = np.matmul(a1, W2)

        # Loss (MSE)
        diff = pred - Y
        loss = np.mean(diff ** 2)

        # Backward Pass
        # dL/dPred = 2 * diff / (BATCH_SIZE * OUTPUT_DIM)
        grad_pred = 2.0 * diff / (BATCH_SIZE * OUTPUT_DIM)
        
        # grad_W2 = a1.T @ grad_pred
        grad_W2 = np.matmul(a1.T, grad_pred)
        
        # grad_a1 = grad_pred @ W2.T
        grad_a1 = np.matmul(grad_pred, W2.T)
        
        # grad_h1 = grad_a1 * relu_deriv
        grad_h1 = grad_a1 * (h1 > 0).astype(np.float32)
        
        # grad_W1 = X.T @ grad_h1
        grad_W1 = np.matmul(X.T, grad_h1)

        # Optimizer step
        W1 -= LR * grad_W1
        W2 -= LR * grad_W2

        epoch_time = time.time() - start_time
        total_time += epoch_time

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.6f} | Time: {epoch_time:.6f}s")

    end_energy = get_rapl_energy_joules()
    energy_joules = end_energy - start_energy

    print(f"Total Training Time (Numpy): {total_time:.4f}s | Energy: {energy_joules:.4f}J")

    with open(filepath, "a") as f:
        f.write(json.dumps({
            "task": "dummy_mlp",
            "engine": "numpy",
            "backend": "openblas",
            "total_time": total_time,
            "avg_epoch_time": total_time / EPOCHS,
            "energy_joules": energy_joules
        }) + "\n")

if __name__ == "__main__":
    main()
