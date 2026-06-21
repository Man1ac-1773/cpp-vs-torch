# 03. OS Jitter and Allocators

After tiling, I went full low-level and replaced the standard math operations with AVX SIMD intrinsics. By using `_mm256_fmadd_ps`, I forced the CPU to do 8 floating-point operations per clock cycle. 

This dropped the execution time from seconds down to about 1.78s for an N=2000 matrix. 

But I noticed a very weird anomaly in the data. My C engine was consistently 16% faster than my C++ engine, even though they were running the exact same AVX instructions. 

The only difference between the two engines was memory management. The C engine uses a custom Bump Arena Allocator with a pre-allocated 1GB page. The C++ engine uses `std::vector` for dynamic heap allocations. 

To prove that C++ was suffering from OS-level jitter, I hooked into the Linux `getrusage()` syscall to track Minor Page Faults during repeated matrix allocations, simulating a standard training loop.

Here is the data for the second allocation pass:
- C (Bump Allocator): 0 page faults
- C++ (`std::vector`): 7,813 page faults

The data was pretty clear. Because the C engine uses a bump allocator, the memory remains physically mapped to the process even when it is "freed". The next allocation happens instantly. 

The C++ engine triggers the `std::vector` destructor when the matrix goes out of scope, returning the memory to the OS. On the next iteration, the OS has to pause the CPU and remap the physical RAM pages, triggering thousands of page faults per matrix. Over a 10-run sweep, this resulted in millions of microscopic OS context switches, adding a massive 0.35s overhead. 

It really goes to show that high-level abstractions like `std::vector` are not exactly free lmao.
