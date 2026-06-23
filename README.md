# CPP-vs-Torch: Outpacing PyTorch from scratch

Everyone uses PyTorch and NumPy, but very few people actually know what happens under the hood when you multiply two tensors. I decided to build two deep learning engines completely from scratch just to see how deep the rabbit hole goes, and to answer a single question.

Can a solo dev build a matrix multiplication engine in raw C/C++ that beats PyTorch? 

Spoiler alert: I actually pulled it off lmao.

This repository contains two backend engines built entirely from the ground up:
- `macrograd`: A pure C backend utilizing a custom Bump Arena Allocator.
- `minigrad`: A C++ backend relying on `std::vector` and RAII principles.

I benchmarked both engines against PyTorch and NumPy across varying matrix sizes (N=10 to N=2000), stacking optimizations one by one. The performance journey was insane, dropping execution time from an abysmal 17 seconds down to just 0.20 seconds, officially dethroning PyTorch.

### The optimization rabbit hole

I documented the entire process. If you want to see exactly how you strip away abstractions to achieve bare metal performance, check out my technical deep dives in the `reports/` directory. Here is a teaser of what you will find:

- **[01. The Naive Baseline](reports/01_the_naive_baseline.md)**: Writing the standard O(N^3) triple for loop. Finding out that my code was roughly 73x slower than PyTorch.
- **[02. Cache Misses & Tiling](reports/02_cache_misses_and_tiling.md)**: Hooking into the Linux `perf_event_open` syscall to prove that simply reading memory in the wrong order caused 1 billion cache misses, and how matrix tiling fixed it.
- **[03. OS Jitter & Allocators](reports/03_os_jitter_and_allocators.md)**: Bypassing the compiler with AVX SIMD intrinsics, and using `getrusage()` to expose the hidden cost of high-level abstractions like `std::vector` compared to a raw C bump allocator.
- **[04. Beating PyTorch](reports/04_beating_pytorch.md)**: Unlocking hardware threading via pthreads and OpenMP. Witnessing dynamic work-stealing finally push execution time down to 0.20s and secure the win.
- **[05. End-to-End ML Training](reports/05_end_to_end_ml_training.md)**: Proving that our optimized C engine can train a Multi-Layer Perceptron up to 2x faster than PyTorch's ATen backend by utilizing an $O(1)$ Arena memory checkpoint to eliminate page faults.
- **[06. Diagnosing Erratic Execution Times](reports/06_diagnosing_erratic_times.md)**: An analysis into why single-threaded CPU intensive workloads experience massive time jumps due to OS Thread Migration (E-Cores), Thermal Throttling, and Subnormal Floating Point microcode traps.

### Project structure
- `/macrograd/`: The C engine source code
- `/minigrad/`: The C++ engine source code
- `/benchmarking/`: The raw C++ and Python benchmarking scripts, Chrome profiler trace generators, and JSONL data dumps
- `/reports/`: My technical writeups on the profiling discoveries
