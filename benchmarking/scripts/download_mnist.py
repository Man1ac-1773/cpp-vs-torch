import os
import numpy as np
from torchvision import datasets

import pathlib

# Get the absolute path to the repo root
repo_root = pathlib.Path(__file__).parent.parent.parent.absolute()
data_dir = repo_root / "resources" / "data"

os.makedirs(data_dir, exist_ok=True)

print("Downloading MNIST...")
dataset = datasets.MNIST(str(data_dir), download=True, train=True)

print("Flattening and normalizing...")
X = dataset.data.numpy().reshape(-1, 28*28).astype(np.float32) / 255.0
Y_labels = dataset.targets.numpy()

print("One-hot encoding labels...")
Y = np.zeros((len(Y_labels), 10), dtype=np.float32)
Y[np.arange(len(Y_labels)), Y_labels] = 1.0

print("Dumping to binary files for C/C++ engines...")
X.tofile(data_dir / 'train_images.bin')
Y.tofile(data_dir / 'train_labels.bin')

print(f"Successfully dumped {X.shape} images and {Y.shape} labels to {data_dir}/")
