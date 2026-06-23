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

def load_mnist_bin(image_path, label_path, num_samples, input_dim=784, output_dim=10):
    with open(image_path, 'rb') as f:
        img_data = f.read()
    X = np.frombuffer(img_data, dtype=np.float32).copy().reshape(num_samples, input_dim)
    
    with open(label_path, 'rb') as f:
        lbl_data = f.read()
    Y = np.frombuffer(lbl_data, dtype=np.float32).copy().reshape(num_samples, output_dim)
    
    return X, Y

def compute_accuracy(pred, target):
    pred_classes = np.argmax(pred, axis=1)
    target_classes = np.argmax(target, axis=1)
    correct = np.sum(pred_classes == target_classes)
    return correct / target.shape[0]

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "performance-plugged"
    filepath = f"./data/np_mnist_{mode}.jsonl"

    TRAIN_SIZE = 60000
    TEST_SIZE = 10000
    INPUT_DIM = 784
    HIDDEN1 = 64
    HIDDEN2 = 32
    OUTPUT_DIM = 10
    EPOCHS = 5
    LR = 0.5

    print("Training MNIST 3-Layer MLP on Numpy Engine...")

    X_train, Y_train = load_mnist_bin("../../resources/data/train_images.bin", "../../resources/data/train_labels.bin", TRAIN_SIZE, INPUT_DIM, OUTPUT_DIM)
    X_test, Y_test = load_mnist_bin("../../resources/data/test_images.bin", "../../resources/data/test_labels.bin", TEST_SIZE, INPUT_DIM, OUTPUT_DIM)

    np.random.seed(42)
    W1 = (np.random.rand(INPUT_DIM, HIDDEN1).astype(np.float32) * 0.2) - 0.1
    W2 = (np.random.rand(HIDDEN1, HIDDEN2).astype(np.float32) * 0.2) - 0.1
    W3 = (np.random.rand(HIDDEN2, OUTPUT_DIM).astype(np.float32) * 0.2) - 0.1

    total_time = 0.0
    start_energy = get_rapl_energy_joules()

    for epoch in range(EPOCHS):
        start_time = time.time()

        # Forward Pass
        h1 = np.matmul(X_train, W1)
        a1 = np.maximum(0, h1)
        h2 = np.matmul(a1, W2)
        a2 = np.maximum(0, h2)
        logits = np.matmul(a2, W3)

        # Softmax + Cross Entropy
        # Shift logits for numerical stability
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Loss
        loss = -np.mean(np.sum(Y_train * np.log(probs + 1e-8), axis=1))

        # Backward Pass
        # dL/dLogits = (probs - Y_train) / BATCH_SIZE
        grad_logits = (probs - Y_train) / TRAIN_SIZE
        
        grad_W3 = np.matmul(a2.T, grad_logits)
        grad_a2 = np.matmul(grad_logits, W3.T)
        
        grad_h2 = grad_a2 * (h2 > 0).astype(np.float32)
        grad_W2 = np.matmul(a1.T, grad_h2)
        grad_a1 = np.matmul(grad_h2, W2.T)
        
        grad_h1 = grad_a1 * (h1 > 0).astype(np.float32)
        grad_W1 = np.matmul(X_train.T, grad_h1)

        # Optimizer step
        W1 -= LR * grad_W1
        W2 -= LR * grad_W2
        W3 -= LR * grad_W3

        epoch_time = time.time() - start_time
        total_time += epoch_time

        train_acc = compute_accuracy(probs, Y_train)
        print(f"Epoch {epoch} | Loss: {loss:.6f} | Train Acc: {train_acc * 100:.2f}% | Time: {epoch_time:.6f}s")

    # Test Evaluation
    h1_test = np.matmul(X_test, W1)
    a1_test = np.maximum(0, h1_test)
    h2_test = np.matmul(a1_test, W2)
    a2_test = np.maximum(0, h2_test)
    logits_test = np.matmul(a2_test, W3)
    test_acc = compute_accuracy(logits_test, Y_test)
    
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    end_energy = get_rapl_energy_joules()
    energy_joules = end_energy - start_energy

    print(f"Total Training Time (Numpy): {total_time:.4f}s | Energy: {energy_joules:.4f}J")

    with open(filepath, "a") as f:
        f.write(json.dumps({
            "task": "mnist_mlp",
            "engine": "numpy",
            "backend": "openblas",
            "total_time": total_time,
            "avg_epoch_time": total_time / EPOCHS,
            "energy_joules": energy_joules,
            "test_acc": test_acc
        }) + "\n")

if __name__ == "__main__":
    main()
