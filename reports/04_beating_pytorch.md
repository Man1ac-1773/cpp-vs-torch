# 04. Beating PyTorch

Even after perfectly tuning memory access with Tiling and bypassing the compiler with AVX SIMD intrinsics, PyTorch was still destroying my custom engines. 

PyTorch finished the N=2000 benchmark in 0.23 seconds, while my highly-optimized single-threaded C kernel took 1.78 seconds. The reason was simple: PyTorch leverages all 16 hardware cores of my laptop by default, while my engine was restricted to a single core. 

It was time to level the playing field and unlock hardware threading.

I implemented multithreading in two different ways to compare threading paradigms:
1. **C Engine (`pthreads`)**: I hand-rolled thread spawning, deploying 8 worker threads and manually partitioning the matrix rows into rigid, equal chunks.
2. **C++ Engine (`OpenMP`)**: I used compiler directives (`#pragma omp parallel for`), allowing the C++ compiler to dynamically distribute loop iterations across logical cores.

I ran the N=2000 benchmark again.

### The Multithreading Showdown (N=2000)

| Engine | Threading Model | Time (Seconds) | Scaling Speedup | PyTorch Status |
|---|---|---|---|---|
| **PyTorch** | `ATen` (Built-in MT) | 0.23s | N/A | Defending Champion |
| **C (`macrograd`)**| Manual `pthreads` | 0.25s | 6.07x | Close, but no cigar |
| **C++ (`minigrad`)**| Dynamic `OpenMP` | **0.20s** | **9.38x** | **Dethroned!** |

### The Verdict

I actually managed to dethrone PyTorch lmao. By unlocking hardware threading, the C++ OpenMP kernel finished in 0.20s, officially outpacing the PyTorch ATen baseline.

It was incredibly interesting to see C++ beat C in this specific phase. My manual C `pthread` implementation statically partitioned the matrix into 8 equal chunks. Because the OS is noisy (background tasks, interrupts), if one core gets throttled, that specific thread finishes last and the entire `pthread_join` blocks waiting for it. This is known as the classic "straggler" problem.

OpenMP, however, uses dynamic work-stealing beneath the hood. As cores finish their chunks early, they dynamically pick up the remaining matrix rows from the slower cores. This completely eliminated the straggler problem, allowing the C++ engine to achieve a near-perfect 9.4x linear speedup and secure the ultimate win.

**Key Learning:** Hardware threading is where the real leaps in performance happen, but static work allocation is vulnerable to OS noise. Dynamic thread pools are mandatory for maximum CPU utilization.

### Scripts and Raw Data

The multithreaded benchmarks were run using `../benchmarking/benchmark_mt_c.cpp`, `../benchmarking/benchmark_mt_cpp.cpp`, and `../benchmarking/benchmark_mt.py`. The full performance curves from N=10 to N=2000 are recorded in the respective `mt_benchmark_results.jsonl` files located inside the `c-data`, `cpp-data`, and `py-data` directories.
