import torch
import time
import json

BATCH_SIZE = 1024
INPUT_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 64
EPOCHS = 100
LR = 0.01

print(f"Training 2-Layer MLP on PyTorch (Batch={BATCH_SIZE})...")

X = torch.full((BATCH_SIZE, INPUT_DIM), 0.5)
Y = torch.full((BATCH_SIZE, OUTPUT_DIM), 0.1)

W1 = torch.full((INPUT_DIM, HIDDEN_DIM), 0.01, requires_grad=True)
W2 = torch.full((HIDDEN_DIM, OUTPUT_DIM), -0.01, requires_grad=True)

optimizer = torch.optim.SGD([W1, W2], lr=LR)
loss_fn = torch.nn.MSELoss()

total_time = 0

with open('./py-data/train_results.jsonl', 'w') as f:
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # 1. Forward Pass
        h1 = torch.matmul(X, W1)
        a1 = torch.relu(h1)
        pred = torch.matmul(a1, W2)
        
        # 2. Loss
        loss = loss_fn(pred, Y)
        
        # 3. Backward Pass
        optimizer.zero_grad()
        loss.backward()
        
        # 4. Optimizer Step
        optimizer.step()
        
        epoch_time = time.time() - start_time
        total_time += epoch_time
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item()} | Time: {epoch_time}s")
            
        f.write(json.dumps({
            "engine": "PyTorch",
            "epoch": epoch,
            "loss": loss.item(),
            "time": epoch_time
        }) + "\n")

print(f"Total Training Time (PyTorch): {total_time}s")
