# 04. Scaling to MNIST: The Framework Overhead Reversal

After optimizing the fundamental matrix multiplication kernels, I needed to test if my custom C and C++ Automatic Differentiation engines could actually train a neural network faster than PyTorch. 

My hypothesis was that on small-scale networks, my lightweight custom engines would beat PyTorch by bypassing its massive overhead. But as the network scaled up, the sheer mathematical superiority of PyTorch's Intel MKL backend would eventually overwhelm my custom OpenMP AVX2 loops.

I conducted the benchmarks across two workloads:
1. **Dummy MLP:** Small-scale matrix multiplication (`1024` batch size, `128 -> 256 -> 64`).
2. **MNIST MLP:** Large-scale full-batch matrix multiplication (`60000` batch size, `784 -> 64 -> 32 -> 10`).

### The Empirical Data

All tests were strictly run on the `performance-plugged` OS power profile.

**Small-Scale (Dummy MLP)**
| Engine | Avg Epoch Time | Total Energy |
|--------|----------------|--------------|
| **NumPy (OpenBLAS)** | `1.27 ms` | `14.5 J` |
| **C++ (SIMD - OpenMP)** | `3.71 ms` | `43.1 J` |
| **PyTorch (ATen)** | `19.40 ms` | `38.4 J` |
| **C (SIMD - Pthreads)** | `34.54 ms` | `239.3 J` |

On the small dataset, my **C++ SIMD Engine completely crushed PyTorch (5.2x faster)!** However, notice the massive energy anomaly with the C SIMD engine utilizing Pthreads.

**Large-Scale (MNIST MLP)**
| Engine | Avg Epoch Time | Total Energy | Test Accuracy |
|--------|----------------|--------------|---------------|
| **PyTorch (ATen)** | `61.7 ms` | `725.4 J` | `79.44%` |
| **NumPy (OpenBLAS)** | `94.4 ms` | `1416.5 J` | `79.40%` |
| **C++ (SIMD - OpenMP)** | `366.4 ms` | `5249.1 J` | `73.07%` |
| **C++ (Naive)** | `5592.8 ms` | `12579 J` | `72.23%` |

When scaled up to the `60000 x 784` MNIST dataset, the narrative violently flipped. **PyTorch decimated my custom engines, running almost 6x faster than C++ SIMD.**

### The Mechanism: PyTorch Framework Overhead
Why did PyTorch take 19.4ms on a workload that C++ finished in 3.7ms? The actual math takes fractions of a millisecond. The remaining milliseconds in PyTorch are pure "Framework Overhead." This consists of:
1. **The GIL:** Releasing the Python Global Interpreter Lock.
2. **Pybind11 Translation:** Marshaling data across the Python/C++ boundary into libtorch.
3. **ATen Dynamic Graph:** Allocating `torch::Tensor` objects, instantiating `std::shared_ptr` nodes for the Autograd graph, and executing the C++ dispatcher.
4. **MKL Setup:** Preparing the Intel MKL BLAS library for execution.

My C++ engine is just a single compiled binary executing pure pointers, bypassing all of this. NumPy bypasses the Autograd graph entirely because we manually calculate gradients, which is why it runs in just 1.27ms.

However, on the massive MNIST dataset, the math itself takes tens of milliseconds. The framework overhead is diluted. ATen's highly-tuned Intel MKL (Math Kernel Library) operations utilize cache-blocking, register tiling, and assembly micro-kernels that vastly outperform my raw `#pragma omp parallel for` AVX intrinsic loops. 

### The Pthreads Energy Spike Anomaly
Look closely at the Dummy MLP data. The C SIMD (Pthreads) implementation burned a staggering **239.3 Joules** while the C++ SIMD implementation doing the exact same math with OpenMP burned only **43.1 Joules** (a 5.5x difference!). 

Initially, I hypothesized this was due to aggressive busy-wait spinlocks in Pthreads. However, deeper analysis across the raw Matmul benchmarks shows this isn't true; Pthreads actually burns *less* energy than OpenMP in raw, sustained matrix multiplications. 

The energy anomaly is specifically isolated to the *training loop logic* itself, possibly tied to how the OS scheduler handles rapid sequential kernel launches across different power modes. I conduct a full deep-dive into this exact anomaly in **[A8. OpenMP vs Pthreads](A8_openmp_vs_pthreads.md)**.

### 5. The Verdict
Do not use heavy deep learning frameworks for tiny environments (like Reinforcement Learning with small MLPs); custom C/C++ engines are strictly better. But for massive datasets, you cannot easily beat decades of BLAS micro-kernel optimization.

---
### Scripts & Raw Data
- **C Engine Script (MNIST)**: [`c_mnist.cpp`](../benchmarking/training/c_mnist.cpp)
- **C++ Engine Script (MNIST)**: [`cpp_mnist.cpp`](../benchmarking/training/cpp_mnist.cpp)
- **PyTorch Script (MNIST)**: [`py_mnist.py`](../benchmarking/training/py_mnist.py)
- **NumPy Script (MNIST)**: [`np_mnist.py`](../benchmarking/training/np_mnist.py)
- **Raw Data Log**: [`benchmarking/training/data/`](../benchmarking/training/data/)
