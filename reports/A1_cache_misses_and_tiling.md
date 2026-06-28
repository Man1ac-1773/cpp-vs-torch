# A1. Cache Misses and Tiling

### The Hypothesis
In my baseline benchmarks, the naive $O(N^3)$ matrix multiplication algorithm took a disastrous 8 to 9 seconds to multiply two $2000 \times 2000$ matrices on the highest performance profile. My hypothesis was that the CPU was not bottlenecked by the actual arithmetic, but rather by the latency of fetching data from the main memory. Specifically, I suspected that iterating over the columns of the second matrix was destroying spatial locality, leading to catastrophic Cache Thrashing. 

To fix this, I implemented Matrix Tiling (Loop Blocking), breaking the matrices into $32 \times 32$ sub-blocks designed to fit perfectly inside the CPU's ultra-fast L1 Data Cache. 

### The Empirical Data
To prove this hypothesis, I hooked directly into the CPU's Performance Monitoring Unit (PMU) using the Linux `perf_event_open` syscall. I instructed the CPU to track every single time it attempted to fetch data from the L1 Data Cache and failed.

Here is the hardware counter data for an $N=1000$ matrix multiplication using my C Engine on the `performance-plugged` profile:

| Algorithm | Paradigm | L1 Cache Misses | Execution Time (`performance-plugged`) |
|---|---|---|---|
| **Naive Matmul** | Sequential Columns | `996,030,754` | ~0.93s |
| **Tiled Matmul** | $32 \times 32$ Sub-blocks | `3,409,932` | ~0.57s |

Tiling reduced L1 Cache Misses by a staggering **292x**, resulting in an immediate speedup.

> [!NOTE]  
> **The Battery Profile Anomaly**: In earlier testing, I frequently saw N=1000 execution times of **~1.90s** (Naive) and **~1.16s** (Tiled). It turns out these were captured while the laptop was running under the `saver` and `saver-plugged` power profiles. The OS actively schedules threads onto slower E-Cores and artificially caps clock speeds to save energy. When doing bare-metal benchmarking, failing to disclose or lock the battery profile can lead to wildly inaccurate conclusions!

### The Hardware Mechanism: Cache Lines
Why did the naive algorithm trigger nearly one billion cache misses? The answer lies in how modern CPUs fetch memory. 

A CPU does not fetch a single 4-byte `float` from RAM. To optimize for sequential access, the memory controller fetches an entire **64-byte Cache Line** at once. Since $64 \text{ bytes} / 4 \text{ bytes} = 16$, every single memory fetch pulls exactly **16 continuous floats** into the L1 cache.

In C and C++, matrices are stored in **Row-Major Order** (a flat 1D array where rows are laid out end-to-end). 
- When iterating through a matrix **row-wise** (Matrix A), I read `index 0`, then `index 1`, etc. The CPU fetches a cache line containing indices 0 through 15. The first read is a Cache Miss, but the next 15 reads are guaranteed Cache Hits.
- When iterating through a matrix **column-wise** (Matrix B), I read `index 0`, then the next element is at `index 0 + N`. If $N=1000$, the next element is 4,000 bytes away. The CPU fetches a 64-byte cache line, I use exactly 1 float, and then I jump to a completely different memory address, throwing away the remaining 15 floats.

By reading column-wise, I was mathematically wasting **93.75%** ($15/16$) of the physical memory bus bandwidth. The CPU was spending its time waiting for RAM to respond, doing absolutely zero math. 

### The Code Proof
Here is the inner loop of the Naive algorithm. Notice the memory access on `B`:

```c
// Naive: column-wise access on B is disastrous
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            // A is accessed sequentially (A[i*N + k])
            // B is accessed strided (B[k*N + j]) -> CACHE THRASHING
            sum += A[i*N + k] * B[k*N + j];
        }
        C[i*N + j] = sum;
    }
}
```

By adding three outer loops to process the matrix in $32 \times 32$ blocks, the `Tiled` algorithm ensures that a chunk of `B` is loaded into the L1 cache once and then reused 32 times by different rows of `A` before it is ever evicted.

### The Verdict
Memory access patterns are often vastly more important than the actual math being executed. Tiling successfully bypassed the main memory bottleneck by enforcing spatial locality. However, even with the cache fixed, I was still miles behind PyTorch. My CPU was fed, but the math was too slow. It was time to bypass the compiler entirely using SIMD.
