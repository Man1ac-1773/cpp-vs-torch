# cpptorch-vs-torch

I built this project to answer a very simple question: how hard is it to build a matrix multiplication engine from scratch that can beat PyTorch? 

Spoiler alert: I actually beat it lmao.

This repository contains two deep learning engines built entirely from scratch:
- `macrograd`: A pure C backend using a custom Bump Arena Allocator.
- `minigrad`: A C++ backend using `std::vector` and RAII principles.

I benchmarked both of these engines against NumPy and PyTorch across varying matrix sizes (N=10 to N=2000), implementing optimizations one by one to see how deep the rabbit hole goes. 

### The journey
If you want to read about the deep systems-level profiling I did, check out the `reports/` directory. I documented the entire optimization ladder:

1. **The Naive Baseline**: Writing the standard O(N^3) triple for loop.
2. **Cache Misses & Tiling**: Hooking into the Linux `perf_event_open` syscall to prove that matrix tiling reduces L1 cache misses by nearly 300x.
3. **OS Jitter & Allocators**: Bypassing the compiler with AVX SIMD intrinsics, and using `getrusage()` to prove that C++ `std::vector` causes thousands of OS-level page faults compared to my custom C bump allocator.
4. **Beating PyTorch**: Implementing pthreads and OpenMP to achieve dynamic work-stealing, finally pushing the execution time down to 0.20s and dethroning PyTorch's 0.23s baseline.

### Directory structure
- `cpp-vs-torch/macrograd/`: The C engine
- `cpp-vs-torch/minigrad/`: The C++ engine
- `cpp-vs-torch/benchmarking/`: The raw C++ and Python benchmarking scripts, Chrome profiler trace generators, and JSONL data dumps
- `reports/`: My technical writeups on the profiling discoveries
