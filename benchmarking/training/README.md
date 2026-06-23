# Autograd Engine Benchmarking Analysis

This document serves as a comprehensive analysis of the performance characteristics observed when benchmarking our custom C and C++ automatic differentiation engines against NumPy and PyTorch. 

We trained a small 2-layer Multi-Layer Perceptron (MLP) mapping `256 -> 128 -> 64` dimensions over 100 epochs with a batch size of `1024`.

## Architectural Findings

### 1. The Pthread Spawning Bottleneck
Our initial C Engine SIMD implementation utilized `pthread_create` and `pthread_join` inside the core `tensor_matmul_simd_mt` function.

> [!WARNING]
> Because this function is called multiple times per epoch, the engine was spawning and destroying 8 POSIX threads *thousands of times a second*. The OS-level context switching overhead was astronomical.

**The Fix:** We replaced the explicit POSIX thread logic with OpenMP (`#pragma omp parallel for`). OpenMP utilizes an implicit, persistent thread-pool which entirely bypasses the thread creation overhead. 
**The Result:** The C Engine's epoch execution time dropped from **34.5ms** to **~4.5ms**, yielding a near **8x performance multiplier** with a single line of code.

### 2. PyTorch Overhead vs. Bare-Metal Compute
PyTorch clocked in at **19.4ms** per epoch, making it nearly **4.5x slower** than our custom C++ SIMD engine (which averaged **4.5ms**). 

> [!NOTE]
> Why does our tiny engine beat PyTorch? 
> Because the matrices in our dummy MLP are very small (`1024x256`, `256x128`). At this scale, the actual floating-point math takes mere microseconds. PyTorch's execution time is entirely dominated by "Framework Overhead"—the time spent traversing the Python GIL, building the dynamic computation graph in ATen, and dispatching to MKL backend libraries.

Conversely, **NumPy** averaged **1.2ms** per epoch. NumPy executes extremely tight, pre-compiled C-bindings to OpenBLAS, avoiding PyTorch's heavy autograd DAG construction.

### 3. Tiling is Harmful for Small Matrices
Across both C and C++ engines, the `tiled` matrix multiplication backend performed *strictly worse* than the `naive` `O(N^3)` backend.

> [!TIP]
> Tiling divides matrices into `32x32` spatial blocks to enforce temporal data locality in the L1/L2 caches. However, since our Dummy MLP dimensions (`256`, `128`, `64`) are very small, the matrices *already fit perfectly* into the cache. 
> 
> Enforcing 6 layers of deeply nested loops over small matrices introduces massive branch-prediction latency and loop-counter overhead without any spatial benefits. Tiling should only be used dynamically when matrices exceed cache capacities.

## Final Performance & Energy Rankings

Below are the final runtime and energy efficiency rankings for the Dummy MLP test (lower is better):

| Rank | Engine / Backend | Avg Epoch Time | Total Energy |
|------|------------------|----------------|--------------|
| 1 | **NumPy (OpenBLAS)** | `1.27 ms` | `14.5 J` |
| 2 | **C++ (SIMD)** | `4.40 ms` | `43.1 J` |
| 3 | **C (SIMD)** | `5.20 ms` | `52.3 J` |
| 4 | **PyTorch (ATen)** | `19.4 ms` | `38.4 J` |
| 5 | **C++ (Naive)** | `20.6 ms` | `203.5 J` |
| 6 | **C (Naive)** | `28.0 ms` | `272.2 J` |
| 7 | **C (Tiled)** | `42.1 ms` | `326.1 J` |
| 8 | **C++ (Tiled)** | `44.1 ms` | `301.9 J` |

*(Note: PyTorch ranks 4th in speed but 2nd in energy, likely due to highly optimized CPU throttling and multi-core thread sleeping between Python dispatches.)*
