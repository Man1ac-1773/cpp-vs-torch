# 02. "Beating" PyTorch

Before tackling PyTorch, I needed to ensure my single-threaded math was actually optimal. I implemented two massive low-level optimizations:
1. **Cache Tiling**: Blocking matrices into 32x32 chunks to perfectly fit the CPU's L1 cache. *(Deep Dive: [A1. Cache Misses & Tiling](A1_cache_misses_and_tiling.md))*
2. **AVX SIMD Intrinsics**: Bypassing the compiler to execute 8 math operations per clock cycle. *(Deep Dive: [A2. OS Jitter & Allocators](A2_os_jitter_and_allocators.md))*

Even after perfectly tuning memory access with Tiling and utilizing AVX SIMD intrinsics, a fully multi-threaded PyTorch was still destroying my custom engines. 

PyTorch finished the N=2000 benchmark in 0.026 seconds, while my highly-optimized single-threaded C and C++ SIMD kernels took 0.87 seconds and 1.12 seconds, respectively. The reason was simple: PyTorch leverages all 20 hardware threads of my CPU by default, while my engine was restricted to a single core. 

It was time to level the playing field and unlock hardware threading.

I implemented multithreading in two different ways to compare threading paradigms:
1. **C Engine (`pthreads`)**: I hand-rolled thread spawning, deploying 20 worker threads and manually partitioning the matrix rows into rigid, equal chunks.
2. **C++ Engine (`OpenMP`)**: I used compiler directives (`#pragma omp parallel for`), allowing the C++ compiler to dynamically distribute loop iterations across logical cores.

I ran the N=2000 benchmark again, exclusively on the `performance-plugged` power profile.

### The Multithreading Showdown (N=2000)

| Engine | Threading Model | Time (Seconds) | PyTorch Status |
|---|---|---|---|
| **PyTorch** | `ATen` (Built-in MT) | **0.026s** | **Undisputed Champion** |
| **NumPy** | `OpenBLAS` (Built-in MT) | 0.046s | Fast |
| **C++ (`minigrad`)**| Dynamic `OpenMP` | 0.136s | Solid Attempt |
| **C (`macrograd`)**| Manual `pthreads` | 0.157s | Solid Attempt |

### The Verdict

Okay, so I didn't actually dethrone PyTorch's multithreaded ATen backend. PyTorch is heavily optimized and leverages elite BLAS libraries that are virtually impossible to beat with a weekend of C code. 

However, my multi-threaded C++ OpenMP kernel finished in 0.136s. Remember that single-threaded PyTorch took 0.115s? That means I *almost* beat PyTorch single thread though lmfao!

It was incredibly interesting to see C++ beat C in this specific phase (0.136s vs 0.157s). My manual C `pthread` implementation statically partitioned the matrix into 20 equal chunks. Because the OS is noisy (background tasks, interrupts), if one core gets throttled, that specific thread finishes last and the entire `pthread_join` blocks waiting for it. This is known as the classic "straggler" problem.

OpenMP, however, uses dynamic work-stealing beneath the hood. As cores finish their chunks early, they dynamically pick up the remaining matrix rows from the slower cores. This completely eliminated the straggler problem, allowing the C++ engine to scale much better.

**Key Learning:** Hardware threading is where the real leaps in performance happen, but static work allocation is vulnerable to OS noise. Dynamic thread pools are mandatory for maximum CPU utilization.

### Scripts and Raw Data

The multithreaded benchmarks were run using [`c_matmul_multi.cpp`](../benchmarking/matmul/c_matmul_multi.cpp), [`cpp_matmul_multi.cpp`](../benchmarking/matmul/cpp_matmul_multi.cpp), and [`py_matmul_multi.py`](../benchmarking/matmul/py_matmul_multi.py). The full performance curves from N=10 to N=2000 are recorded in the [`benchmarking/matmul/data/`](../benchmarking/matmul/data/) directory.
