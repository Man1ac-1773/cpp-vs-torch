import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import json
import os
import struct
import numpy as np

def get_rapl_energy_joules():
    try:
        with open("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", "r") as f:
            return float(f.read().strip()) / 1e6
    except Exception as e:
        print("Failed to read RAPL:", e)
        return 0.0

def load_mnist_bin(image_path, label_path, num_samples, input_dim=784, output_dim=10):
    with open(image_path, 'rb') as f:
        img_data = f.read()
    X = np.frombuffer(img_data, dtype=np.float32).copy().reshape(num_samples, input_dim)
    
    with open(label_path, 'rb') as f:
        lbl_data = f.read()
    Y = np.frombuffer(lbl_data, dtype=np.float32).copy().reshape(num_samples, output_dim)
    
    return torch.from_numpy(X), torch.from_numpy(Y)

def compute_accuracy(pred, target):
    pred_classes = torch.argmax(pred, dim=1)
    target_classes = torch.argmax(target, dim=1)
    correct = (pred_classes == target_classes).sum().item()
    return correct / target.shape[0]

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "performance-plugged"
    filepath = f"./data/py_mnist_{mode}.jsonl"

    TRAIN_SIZE = 60000
    TEST_SIZE = 10000
    INPUT_DIM = 784
    HIDDEN1 = 64
    HIDDEN2 = 32
    OUTPUT_DIM = 10
    EPOCHS = 10
    BATCH_SIZE = 128
    LR = 0.01

    print("Training MNIST 3-Layer MLP on PyTorch Engine...")

    X_train, Y_train = load_mnist_bin("../../resources/data/train_images.bin", "../../resources/data/train_labels.bin", TRAIN_SIZE, INPUT_DIM, OUTPUT_DIM)
    X_test, Y_test = load_mnist_bin("../../resources/data/test_images.bin", "../../resources/data/test_labels.bin", TEST_SIZE, INPUT_DIM, OUTPUT_DIM)

    model = nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN1, bias=False),
        nn.ReLU(),
        nn.Linear(HIDDEN1, HIDDEN2, bias=False),
        nn.ReLU(),
        nn.Linear(HIDDEN2, OUTPUT_DIM, bias=False)
    )

    with torch.no_grad():
        model[0].weight.uniform_(-0.1, 0.1)
        model[2].weight.uniform_(-0.1, 0.1)
        model[4].weight.uniform_(-0.1, 0.1)

    optimizer = optim.SGD(model.parameters(), lr=LR)
    
    # We are using CrossEntropyLoss, but our labels are one-hot encoded.
    # PyTorch's CrossEntropyLoss expects class indices if target is 1D, or probabilities if target is 2D.
    # We will pass the probabilities (or logits) to CrossEntropyLoss if PyTorch supports it, 
    # but the easiest is to pass the argmax.
    criterion = nn.CrossEntropyLoss()
    Y_train_idx = torch.argmax(Y_train, dim=1)
    Y_test_idx = torch.argmax(Y_test, dim=1)

    total_time = 0.0
    start_energy = get_rapl_energy_joules()

    NUM_BATCHES = TRAIN_SIZE // BATCH_SIZE

    for epoch in range(EPOCHS):
        epoch_time = 0.0
        total_loss = 0.0

        for b in range(NUM_BATCHES):
            start_idx = b * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            X_batch = X_train[start_idx:end_idx]
            Y_batch_idx = Y_train_idx[start_idx:end_idx]

            start_time = time.time()

            pred = model(X_batch)
            loss = criterion(pred, Y_batch_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_time += time.time() - start_time
            total_loss += loss.item()

        print(f"Epoch {epoch} | Avg Loss: {total_loss / NUM_BATCHES:.6f} | Compute Time: {epoch_time:.6f}s")
        total_time += epoch_time

    with torch.no_grad():
        test_pred = model(X_test)
        test_acc = compute_accuracy(test_pred, Y_test)
    
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    end_energy = get_rapl_energy_joules()
    energy_joules = end_energy - start_energy

    print(f"Total Training Time (PyTorch): {total_time:.4f}s | Energy: {energy_joules:.4f}J")

    with open(filepath, "a") as f:
        f.write(json.dumps({
            "task": "mnist_mlp",
            "engine": "pytorch",
            "backend": "aten",
            "total_time": total_time,
            "avg_epoch_time": total_time / EPOCHS,
            "energy_joules": energy_joules,
            "test_acc": test_acc
        }) + "\n")

if __name__ == "__main__":
    main()
