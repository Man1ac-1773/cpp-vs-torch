# 06. The Limits of Custom SIMD: Hitting the BLAS Wall

Throughout my benchmarking, stripping away high-level abstractions like Python dispatch overhead, heap allocations, and page faults yielded massive performance gains. My custom C engine with an $O(1)$ bump arena consistently beat PyTorch on smaller workloads where framework overhead is the primary bottleneck.

However, when scaling up to the full 60,000-image MNIST dataset using Mini-Batch Stochastic Gradient Descent (SGD), the performance dynamics shifted entirely. 

Here are the steady-state execution times per epoch for the four engines running Mini-Batch SGD (Batch Size: 128):

| Engine / Backend | Steady-State Epoch Time |
| :--- | :--- |
| **NumPy (OpenBLAS)** | `~0.115s` |
| **PyTorch (ATen/MKL)** | `~0.119s` |
| **My C Engine (SIMD)** | `~0.250s` |
| **My C++ Engine (SIMD)** | `~0.250s` |

*(Note: PyTorch's `Epoch 0` took a massive `1.97s` to initialize its thread pools and ATen context, but immediately settled into `0.119s` for all subsequent epochs).*

My heavily optimized SIMD engine, utilizing `#pragma omp parallel for`, cache tiling, and `_mm256_fmadd_ps` AVX2 intrinsics, is exactly **2x slower** than the Python frameworks.

Why didn't I beat Python here? Because Python isn't actually doing the math. At scale, framework overhead approaches 0%, and execution time is entirely dominated by the underlying Basic Linear Algebra Subprograms (BLAS) library. I am no longer competing against Python; I am competing against **OpenBLAS** and **Intel MKL**.

Here is why my custom SIMD implementation cannot beat them:

## 1. Hand-Written Assembly Micro-Kernels
My SIMD engine relies on the `g++` compiler to translate C++ intrinsics into machine code. While modern compilers are incredibly smart, they cannot match the efficiency of the **assembly micro-kernels** used by MKL and OpenBLAS. These micro-kernels are meticulously hand-written by hardware engineers to perfectly utilize all available AVX registers, meticulously schedule instructions to hide CPU pipeline latency, and perfectly prefetch data into L1 cache before it is needed.

## 2. Multi-Level Cache Packing (GotoBLAS)
While I implemented basic loop tiling to keep chunks of the matrix inside the L3 cache, OpenBLAS utilizes highly advanced algorithms (like the GotoBLAS algorithm). Before multiplying, it dynamically **packs** blocks of the $A$ and $B$ matrices into temporary, contiguous memory arrays. This completely eliminates Translation Lookaside Buffer (TLB) misses and ensures perfect alignment for the L1 and L2 caches. Because I do not perform memory packing, my engine still suffers from minor cache-bank conflicts.

## 3. Dynamic Architecture Targeting
NumPy and PyTorch ship with pre-compiled binaries containing kernels for multiple CPU architectures. At runtime, they query the exact CPU they are running on and load the most optimal assembly path (e.g., utilizing AVX-512 if available). My engine was compiled statically with `-march=native` and relies entirely on AVX2 instructions.

## Conclusion
Building an engine from scratch teaches you exactly where the bottlenecks in Deep Learning lie. I successfully proved that I can eliminate memory and dispatch overhead to build a faster, leaner Autograd system. However, for the dense, large-scale matrix multiplications that form the core compute of ML, it is virtually impossible to outperform the decades of hand-tuned assembly optimizations present in industry-standard BLAS libraries.
