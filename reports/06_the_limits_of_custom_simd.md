# 06. The Limits of Custom SIMD: Hitting the BLAS Wall

Throughout my benchmarking, stripping away high-level abstractions like Python dispatch overhead, heap allocations, and page faults yielded massive performance gains. My custom engines consistently beat PyTorch on smaller workloads where framework overhead is the primary bottleneck.

However, when scaling up to the full 60,000-image MNIST dataset using Mini-Batch Stochastic Gradient Descent (SGD), the performance dynamics shifted entirely. 

Here are the steady-state execution times per epoch for the four engines running Mini-Batch SGD (Batch Size: 128) on the `performance-plugged` power profile:

| Engine / Backend | Steady-State Epoch Time | Memory Management Strategy |
| :--- | :--- | :--- |
| **NumPy (OpenBLAS)** | `~0.115s` | Pre-allocated arrays |
| **PyTorch (ATen/MKL)** | `~0.119s` | Caching Allocator |
| **My C Engine (SIMD)** | `~0.211s` | $O(1)$ Arena Checkpointing |
| **My C++ Engine (SIMD)** | `~0.252s` | `std::vector` Heap Allocations |

*(Note: PyTorch's `Epoch 0` took a massive `1.97s` to initialize its thread pools and ATen context, but immediately settled into `0.119s` for all subsequent epochs).*

My heavily optimized SIMD engine, utilizing `#pragma omp parallel for`, cache tiling, and `_mm256_fmadd_ps` AVX2 intrinsics, is roughly **2x slower** than the Python frameworks.

### Why is the C engine ~20% faster than C++?
Notice that the C engine (`0.211s`) outperformed the C++ engine (`0.252s`) by nearly 20%. Why did they not hit the exact same execution bottleneck? 

The answer lies in the SGD inner loop. The C++ engine relies heavily on `std::vector` to hold intermediate activations. Even with compiler optimizations, the OS still has to actively manage heap fragmentation and invoke destructors. The C engine, however, leverages the $O(1)$ Arena Checkpoint. During the thousands of mini-batch iterations, it completely bypasses the OS memory allocator by simply resetting a pointer. At high iteration counts, this lack of allocation overhead translates to a direct 20% speedup.

### The BLAS Wall
Even with the Arena Checkpoint, C is still twice as slow as PyTorch/NumPy. Why didn't I beat Python here? Because Python isn't actually doing the math. At scale, framework overhead approaches 0%, and execution time is entirely dominated by the underlying Basic Linear Algebra Subprograms (BLAS) library. I am no longer competing against Python; I am competing against **OpenBLAS** and **Intel MKL**.

Here is why my custom SIMD implementation cannot beat them:

1. **Hand-Written Assembly Micro-Kernels:** My SIMD engine relies on the `g++` compiler to translate C++ intrinsics into machine code. These cannot match the meticulously hand-written assembly micro-kernels used by MKL and OpenBLAS, which perfectly schedule instructions to hide CPU pipeline latency.
2. **Multi-Level Cache Packing (GotoBLAS):** OpenBLAS dynamically packs blocks of the $A$ and $B$ matrices into temporary, contiguous memory arrays. This completely eliminates Translation Lookaside Buffer (TLB) misses and ensures perfect alignment for the L1 and L2 caches. Because I do not perform memory packing, my engine still suffers from minor cache-bank conflicts.

### Conclusion
Building an engine from scratch teaches you exactly where the bottlenecks in Deep Learning lie. I successfully proved that I can eliminate memory and dispatch overhead to build a faster, leaner Autograd system. However, for the dense, large-scale matrix multiplications that form the core compute of ML, it is virtually impossible to outperform the decades of hand-tuned assembly optimizations present in industry-standard BLAS libraries.
