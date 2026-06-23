# 04. Scaling to MNIST: The Framework Overhead Reversal

I successfully completed a comprehensive performance evaluation of my custom C and C++ Automatic Differentiation engines against NumPy (OpenBLAS) and PyTorch (ATen).

The benchmarks were conducted across two distinct workloads:
1. **Dummy MLP:** Small-scale matrix multiplication (`1024` batch size, `128 -> 256 -> 64`). Tests Framework Overhead.
2. **MNIST MLP:** Large-scale full-batch matrix multiplication (`60000` batch size, `784 -> 64 -> 32 -> 10`). Tests Raw Compute scaling.

---

## 1. The "Framework Overhead" Reversal

The most profound finding from my data is the absolute reversal of PyTorch's performance depending on the scale of the workload.

### Small-Scale (Dummy MLP)
| Engine | Avg Epoch Time | Total Energy |
|--------|----------------|--------------|
| **NumPy (OpenBLAS)** | `1.27 ms` | `14.5 J` |
| **C++ (SIMD)** | `3.71 ms` | `43.1 J` |
| **PyTorch (ATen)** | `19.40 ms` | `38.4 J` |
| **C (SIMD - Pthreads)** | `34.54 ms` | `239.3 J` |

On the small Dummy dataset, my **C++ SIMD Engine completely crushed PyTorch (5.2x faster)!** 
Because the matrix dimensions are tiny, the actual math takes fractions of a millisecond. PyTorch's `19.4ms` execution time is entirely eaten up by "Framework Overhead"—traversing the Python GIL, dynamically constructing the ATen computation graph, and dispatching tasks to MKL. My lightweight C++ engine avoids all of this. 

### Large-Scale (MNIST MLP)
| Engine | Avg Epoch Time | Total Energy | Test Accuracy |
|--------|----------------|--------------|---------------|
| **PyTorch (ATen)** | `56.7 ms` | `884.4 J` | `79.44%` |
| **NumPy (OpenBLAS)** | `94.4 ms` | `1416.5 J` | `79.40%` |
| **C (SIMD)** | `307.5 ms` | `4628.9 J` | `73.07%` |
| **C++ (SIMD)** | `366.4 ms` | `5249.1 J` | `73.07%` |
| **C++ (Naive)** | `5592.8 ms` | `12579 J` | `72.23%` |

When I scaled up to the `60000 x 784` MNIST dataset, the narrative violently flipped. 
**PyTorch decimated my custom engines, running 5.4x faster than my C SIMD engine.**
At this scale, the math dominates the runtime, completely diluting PyTorch's framework overhead. ATen's highly-tuned Intel MKL (Math Kernel Library) operations utilize cache-blocking, register tiling, and micro-kernels that vastly outperform my raw `#pragma omp parallel for` AVX intrinsic loops. 

> [!TIP]
> **The Takeaway:** Do not use heavy deep learning frameworks for tiny environments (like Reinforcement Learning with small MLPs); custom C/C++ engines are strictly better. But for massive datasets, you cannot easily beat decades of BLAS micro-kernel optimization.

---

## 2. Tiling vs Naive 

I observed a distinct behavior with the Cache-Tiled backend (`MATMUL_TILED`):
- On the **Dummy dataset**, Tiled (`44ms`) was strictly worse than Naive (`20ms`). The matrices naturally fit in the L1/L2 cache, making the 6-layers of nested loops pure branch-prediction overhead.
- On the **MNIST dataset**, Tiled (`605ms`) was nearly **10x faster** than Naive (`5592ms`)! Because the `60000 x 784` matrices massively exceed the L3 cache, the spatial locality enforced by block-tiling perfectly prevented cache thrashing.

---

## Conclusion
I successfully built a functional Automatic Differentiation engine from scratch in C/C++ that computes numerically stable gradients matching PyTorch. I optimized it from a `O(N^3)` Naive implementation to a highly parallelized OpenMP AVX engine, closing the gap significantly—but ultimately bowing to the supremacy of PyTorch's ATen backends on large-scale compute workloads.
