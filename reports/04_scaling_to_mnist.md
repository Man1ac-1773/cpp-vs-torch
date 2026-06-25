# 04. Scaling to MNIST: The Framework Overhead Reversal

### 1. The Hypothesis
After optimizing the fundamental matrix multiplication kernels, I needed to test if my custom C and C++ Automatic Differentiation engines could actually train a neural network faster than PyTorch. 

My hypothesis was that on small-scale networks, my lightweight custom engines would beat PyTorch by bypassing its massive overhead. But as the network scaled up, the sheer mathematical superiority of PyTorch's Intel MKL backend would eventually overwhelm my custom OpenMP AVX2 loops.

I conducted the benchmarks across two workloads:
1. **Dummy MLP:** Small-scale matrix multiplication (`1024` batch size, `128 -> 256 -> 64`).
2. **MNIST MLP:** Large-scale full-batch matrix multiplication (`60000` batch size, `784 -> 64 -> 32 -> 10`).

### 2. The Empirical Data

**Small-Scale (Dummy MLP)**
| Engine | Avg Epoch Time | Total Energy |
|--------|----------------|--------------|
| **NumPy (OpenBLAS)** | `1.27 ms` | `14.5 J` |
| **C++ (SIMD)** | `3.71 ms` | `43.1 J` |
| **PyTorch (ATen)** | `19.40 ms` | `38.4 J` |
| **C (SIMD - Pthreads)** | `34.54 ms` | `239.3 J` |

On the small dataset, my **C++ SIMD Engine completely crushed PyTorch (5.2x faster)!** However, notice the massive anomaly with the C SIMD engine utilizing Pthreads.

**Large-Scale (MNIST MLP)**
| Engine | Avg Epoch Time | Total Energy | Test Accuracy |
|--------|----------------|--------------|---------------|
| **PyTorch (ATen)** | `56.7 ms` | `884.4 J` | `79.44%` |
| **NumPy (OpenBLAS)** | `94.4 ms` | `1416.5 J` | `79.40%` |
| **C++ (SIMD)** | `366.4 ms` | `5249.1 J` | `73.07%` |
| **C++ (Naive)** | `5592.8 ms` | `12579 J` | `72.23%` |

When scaled up to the `60000 x 784` MNIST dataset, the narrative violently flipped. **PyTorch decimated my custom engines, running 6.4x faster than C++ SIMD.**

### 3. The Mechanism: PyTorch Framework Overhead
Why did PyTorch take 19.4ms on a workload that C++ finished in 3.7ms? The actual math takes fractions of a millisecond. The remaining 19 milliseconds in PyTorch is pure "Framework Overhead." This consists of four specific systems-level bottlenecks:
1. **The GIL:** Releasing the Python Global Interpreter Lock.
2. **Pybind11 Translation:** Marshaling data across the Python/C++ boundary into libtorch.
3. **ATen Dynamic Graph:** Allocating `torch::Tensor` objects, instantiating `std::shared_ptr` nodes for the Autograd graph, and executing the C++ dispatcher.
4. **MKL Setup:** Preparing the Intel MKL BLAS library for execution.

My C++ engine is just a single compiled binary executing pure pointers, bypassing all of this.

However, on the massive MNIST dataset, the math itself takes tens of milliseconds. The 19ms framework overhead is diluted. ATen's highly-tuned Intel MKL (Math Kernel Library) operations utilize cache-blocking, register tiling, and assembly micro-kernels that vastly outperform my raw `#pragma omp parallel for` AVX intrinsic loops. 

### 4. Anomaly for Future Investigation: The Pthreads Energy Spike
Look closely at the Dummy MLP data. The C SIMD (Pthreads) implementation burned a staggering **239.3 Joules** while the C++ SIMD implementation doing the exact same math burned only **43.1 Joules** (a 5.5x difference!). 

I do not currently have the definitive answer for why this occurred. My leading hypothesis is that the POSIX Threads (`pthreads`) implementation I wrote used an aggressive busy-wait spinlock that pinned the CPU frequency at maximum turbo across multiple cores, whereas the C++ OpenMP implementation managed thread-sleeping more efficiently via condition variables. Later in the project, I migrated the C engine from Pthreads to OpenMP, which may provide comparative insights. I will write this down as future investigative work to dig into.

### 5. The Verdict
I successfully built a functional Automatic Differentiation engine from scratch in C/C++ that computes numerically stable gradients matching PyTorch. Do not use heavy deep learning frameworks for tiny environments (like Reinforcement Learning with small MLPs); custom C/C++ engines are strictly better. But for massive datasets, you cannot easily beat decades of BLAS micro-kernel optimization.
