# A2. OS Jitter and Allocators

### 1. The Hypothesis
After optimizing the CPU cache via Tiling, I bypassed the compiler entirely and wrote custom AVX2 SIMD intrinsics. This dropped the execution time from seconds down to about 1.78s for an $N=2000$ matrix. However, an anomaly emerged: my C engine (`macrograd`) was consistently ~20% faster than my C++ engine (`minigrad`), despite both executing the exact same mathematical SIMD instructions. 

My hypothesis was that the standard C++ memory management paradigm—dynamically allocating matrices on the heap using `std::vector`—was introducing severe Operating System overhead compared to the C engine's custom Bump Arena Allocator.

### 2. The Empirical Data
To prove this, I hooked into the Linux `getrusage()` system call to track the number of Minor Page Faults triggered by both engines during a simulated machine learning training loop (repeated matrix allocations).

Here is the data for a single $N=2000$ matrix multiplication pass:

| Engine | Memory Manager | SIMD Time | Minor Page Faults |
|---|---|---|---|
| **C (`macrograd`)** | Bump Allocator | `1.78s` | **0** |
| **C++ (`minigrad`)** | `std::vector` | `2.13s` | **7,813** |

### 3. The Hardware Mechanism: Demand Paging
The common misconception is that the `std::vector` destructor or `malloc()` calls are intrinsically slow. While true, that is not what caused 7,813 page faults. The true culprit is **Demand Paging**.

When the C++ engine calls `new` or `std::vector` to instantiate a matrix, the Linux kernel does not actually map physical RAM to the process. It merely reserves Virtual Address space. The physical RAM is mapped *lazily*. 
When my matrix multiplication code attempts to write to the very first float of that newly allocated vector, the CPU throws a Minor Page Fault. The execution is instantly trapped, control is handed back to the Linux Kernel, the OS finds a physical page in RAM, maps it to the virtual address, and hands control back to my code. This trap-and-map cycle occurs thousands of times as the algorithm touches new pages of the matrix.

In contrast, my C engine uses a Bump Arena Allocator. It pre-allocates a massive 1GB chunk of memory up front and pre-faults it (by using `calloc` or `memset`). During the actual training loop, allocating a new matrix simply means moving a pointer forward. The memory has already been physically mapped to the CPU, resulting in absolutely zero kernel traps.

### 4. The Code Proof
Here is the catastrophic overhead of C++ RAII (Resource Acquisition Is Initialization) vs C Arena Allocation:

**The C++ Engine (OS Jitter):**
```cpp
// Inside the loop: allocates virtual memory. 
// Faults occur upon writing. Destroys at scope exit.
std::vector<float> output(N * N, 0.0f); 
```

**The C Engine (Zero Jitter):**
```c
// Pre-mapped memory. Allocation is just a pointer increment.
float* output = (float*)g_arena.top;
g_arena.top += (N * N * sizeof(float));

// Freeing memory at the end of the pass is instant:
g_arena.top = checkpoint;
```

### 5. The Verdict
Over a multi-epoch training run, the C++ engine suffered millions of microscopic OS context switches, adding massive overhead. High-level abstractions like `std::vector` are incredibly convenient, but putting dynamic heap allocations inside the inner loops of high-performance compute engines is fatal. By bypassing the kernel entirely with an Arena Allocator, I successfully squeezed an extra 19% of raw speed out of the exact same SIMD instructions.
