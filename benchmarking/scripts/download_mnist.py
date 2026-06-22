import os
import numpy as np
from torchvision import datasets

os.makedirs('../data', exist_ok=True)

print("Downloading MNIST...")
dataset = datasets.MNIST('../data', download=True, train=True)

print("Flattening and normalizing...")
X = dataset.data.numpy().reshape(-1, 28*28).astype(np.float32) / 255.0
Y_labels = dataset.targets.numpy()

print("One-hot encoding labels...")
Y = np.zeros((len(Y_labels), 10), dtype=np.float32)
Y[np.arange(len(Y_labels)), Y_labels] = 1.0

print("Dumping to binary files for C/C++ engines...")
X.tofile('../data/train_images.bin')
Y.tofile('../data/train_labels.bin')

print(f"Successfully dumped {X.shape} images and {Y.shape} labels to ../data/")
