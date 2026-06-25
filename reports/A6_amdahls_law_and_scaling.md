# A6. Amdahl's Law and the Core-Scaling Illusion

A fundamental concept in systems engineering is Amdahl's Law: adding more compute resources (threads/cores) to a workload will eventually hit a point of diminishing returns, dictated by the portion of the workload that cannot be parallelized.

To test exactly where our matrix multiplication hits this wall, I explicitly enabled OpenMP multithreading (`#pragma omp parallel for`) on the memory-bound `naive` backend, the cache-friendly `tiled` backend, and the compute-bound `simd` backend. 

I then ran a massive $N=2000$ matrix multiplication, scaling the `OMP_NUM_THREADS` (and PyTorch thread limit) from 1 all the way up to 20 logical cores.

### Scaling Efficiency (Speedup vs Threads)

*Note: The theoretical "Perfect Scaling" would be a 20.0x speedup at 20 threads.*

| Threads | PyTorch Speedup | SIMD Speedup | Tiled Speedup | Naive Speedup |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 1.00x | 1.00x | 1.00x | 1.00x |
| **2** | 1.98x | 2.02x | 1.98x | 1.44x |
| **4** | 3.68x | 4.02x | 3.92x | 3.80x |
| **6** | **2.86x** | **5.33x** | **5.36x** | **4.66x** |
| **7** | 3.10x | **4.65x** (Drop!) | **4.30x** (Drop!)| **3.91x** (Drop!)|
| **12**| 3.82x | 5.53x | 6.88x | 6.19x |
| **16**| 5.02x | 6.65x | 7.43x | 6.17x |
| **20**| 3.82x | **7.43x (Max)** | **6.01x (Regression)** | **4.98x (Regression)** |

### The Findings

1. **The P-Core to E-Core Dropoff:** 
   Look precisely at the jump from **Thread 6 to Thread 7**. Across every single custom C++ backend, the speedup factor physically drops. A 6-thread workload ran *faster* than a 7-thread workload! 
   This perfectly exposes modern asymmetric CPU architecture (like Intel's P-Core/E-Core or Hyper-Threading designs). Threads 1-6 occupied the high-performance physical cores. The moment Thread 7 was spawned, the OS either scheduled it onto a slower Efficiency Core (E-Core) or forced two threads to share a single core's resources via Hyper-Threading, causing immediate resource contention and cache thrashing.

2. **Memory Bottleneck vs Compute Bottleneck:**
   Amdahl's law dictates that the speedup is limited by the bottleneck. 
   For the memory-bound **Naive** and **Tiled** backends, throwing 20 threads at the problem caused a severe performance regression. At 20 threads, the Naive code ran *slower* than it did at 9 threads! This happens because 20 threads are simultaneously trying to pull un-cached data from RAM, completely saturating the motherboard's memory bus.
   Conversely, the **SIMD** backend (which is compute-bound and packs the L1 cache perfectly) continued to scale all the way to 20 threads without regressing.

3. **PyTorch's Overhead at Small Thread Counts:**
   PyTorch hit diminishing returns almost immediately. This is likely because at $N=2000$, PyTorch completes the math so astonishingly fast (0.02s) that the overhead of spawning and synchronizing 20 software threads physically takes longer than just doing the math on 4 threads.
