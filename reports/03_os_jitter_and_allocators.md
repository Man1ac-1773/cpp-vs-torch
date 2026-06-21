# 03. OS Jitter and Allocators

After tiling, I went full low-level and replaced the standard math operations with Advanced Vector Extensions (AVX) SIMD intrinsics. 

By using the `_mm256_fmadd_ps` hardware instruction, I forced the CPU to execute 8 floating-point operations in a single clock cycle. This optimization dropped the execution time from seconds down to about 1.78s for an N=2000 matrix. 

But as I was analyzing the sweep data, I noticed a very weird anomaly. 

### The SIMD Anomaly (N=2000)

| Engine | Memory Manager | SIMD Time | Relative Speed |
|---|---|---|---|
| **C (`macrograd`)** | Bump Arena Allocator | 1.78s | 1.0x |
| **C++ (`minigrad`)** | `std::vector` (Heap) | 2.13s | 1.19x Slower |

My C engine was consistently nearly 20% faster than my C++ engine, even though both were running the exact same AVX instructions and mathematical loops. The only difference between the two engines was memory management. 

The C engine uses a custom Bump Arena Allocator, which pre-allocates a massive 1GB chunk of memory up front. The C++ engine relies on the standard library `std::vector` to dynamically allocate heap memory. 

To prove that C++ was suffering from OS-level jitter, I hooked into the Linux `getrusage()` system call to track Minor Page Faults during repeated matrix allocations (simulating a standard machine learning training loop).

### Minor Page Faults (Second Allocation Pass)

| Engine | Memory Manager | Minor Page Faults |
|---|---|---|
| **C (`macrograd`)** | Bump Allocator | 0 |
| **C++ (`minigrad`)** | `std::vector` | 7,813 |

### The Verdict

The data perfectly explained the anomaly. Because the C engine uses a bump allocator, the memory remains physically mapped to the process even when it is "freed" (by simply resetting a pointer). The subsequent matrix allocations happen instantly. 

The C++ engine, however, triggers the `std::vector` destructor when the matrix goes out of scope, returning the memory to the OS. On the very next iteration, the OS has to pause the CPU, trap the process, and remap the physical RAM pages, triggering thousands of page faults per matrix. 

**Key Learning:** Over a 10-run sweep, the C++ engine suffered millions of microscopic OS context switches, adding a massive 0.35s overhead. High-level abstractions like `std::vector` are incredibly convenient, but they are absolutely not free lmao.
