# A1. Cache Misses and Tiling

To fix the 17-second disaster from the naive implementation, I implemented Matrix Tiling (also known as Loop Blocking). 

Instead of iterating over the entire row and column sequentially, I broke the large matrices down into small 32x32 sub-matrices (blocks). By multiplying these small blocks together one at a time, I ensured that all the required data fit perfectly inside the CPU's tiny, ultra-fast 32KB L1 Data Cache.

This dropped the execution time by roughly 2x in both my C and C++ engines. But I wanted hard data to prove exactly why this happened at the hardware level.

### Hooking into the Linux Kernel

I tapped directly into the CPU's Performance Monitoring Unit (PMU) using the Linux `perf_event_open` syscall. I instructed the CPU to track every single time it tried to fetch data from the L1 Data Cache and failed, forcing a CPU stall.

I ran the profiler for an N=1000 matrix. 

### L1 Data Cache Misses (N=1000)

| Algorithm | Paradigm | L1 Cache Misses | Execution Time | Speedup |
|---|---|---|---|---|
| **Naive Matmul** | Sequential Columns | 996,030,754 | ~1.90s | 1.0x |
| **Tiled Matmul** | 32x32 Sub-blocks | 3,409,932 | ~1.16s | 1.6x |

### The Verdict

The naive algorithm triggered nearly one billion cache misses. Tiling reduced this by 292x down to just 3.4 million. 

**Key Learning:** Memory access patterns are often more important than the actual math. By making sure the 32x32 blocks fit perfectly inside the L1 cache, I completely bypassed the main memory bottleneck. The CPU spent its time actually doing math instead of sitting idle waiting for RAM to respond.

It is actually crazy how much performance you lose just by reading memory in the wrong order lmao. But even with a 2x speedup, I was still miles behind PyTorch. It was time to bypass the compiler entirely.

### Scripts and Raw Data

The PMU profiling logic for this experiment is isolated in `../benchmarking/cache/cpp_cache.cpp` and `c_cache.cpp`, which interfaces directly with the Linux syscalls via `../benchmarking/common/perf_profiler.h`.
