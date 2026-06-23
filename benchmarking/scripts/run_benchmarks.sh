#!/bin/bash
# run_benchmarks.sh
# Automates the execution of all dummy MLP and MNIST MLP benchmarks

set -e

# Change directory to the root of the benchmarking folder
cd "$(dirname "$0")/.."

echo "========================================="
echo " Compiling Benchmarking Scripts... "
echo "========================================="

mkdir -p data

# Compile Dummy MLP Scripts
echo "Compiling training/c_train.cpp..."
g++ -O3 -march=native training/c_train.cpp -o training/c_train -fopenmp

echo "Compiling training/cpp_train.cpp..."
g++ -O3 -march=native training/cpp_train.cpp -o training/cpp_train -fopenmp

# Compile MNIST MLP Scripts
echo "Compiling training/c_mnist.cpp..."
g++ -O3 -march=native training/c_mnist.cpp -o training/c_mnist -fopenmp

echo "Compiling training/cpp_mnist.cpp..."
g++ -O3 -march=native training/cpp_mnist.cpp -o training/cpp_mnist -fopenmp

echo "Compilation successful!"
echo "========================================="
echo " Running Dummy MLP Sweeps... "
echo "========================================="

echo "Running C Dummy MLP..."
./training/c_train full_run

echo "Running C++ Dummy MLP..."
./training/cpp_train full_run

echo "Running PyTorch Dummy MLP..."
python3 training/py_train.py full_run

echo "Running NumPy Dummy MLP..."
python3 training/np_train.py full_run

echo "========================================="
echo " Running MNIST MLP Sweeps... "
echo "========================================="

echo "Running C MNIST MLP..."
./training/c_mnist full_run

echo "Running C++ MNIST MLP..."
./training/cpp_mnist full_run

echo "Running PyTorch MNIST MLP..."
python3 training/py_mnist.py full_run

echo "Running NumPy MNIST MLP..."
python3 training/np_mnist.py full_run

echo "========================================="
echo " All Benchmarks Completed! "
echo " Results are stored in the benchmarking/data/ directory."
echo "========================================="
