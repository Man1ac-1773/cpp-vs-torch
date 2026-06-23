# CPP-vs-Torch: Outpacing PyTorch from scratch

Everyone uses PyTorch and NumPy, but very few people actually know what happens under the hood when you multiply two tensors. I decided to build two deep learning engines completely from scratch just to see how deep the rabbit hole goes, and to answer a single question.

Can a solo dev build a matrix multiplication engine in raw C/C++ that beats PyTorch? 

Spoiler alert: I actually pulled it off lmao.

This repository contains two backend engines built entirely from the ground up:
- `macrograd`: A pure C backend utilizing a custom Bump Arena Allocator. [Credits](https://github.com/Celibistrial/macrograd)
- `minigrad`: A C++ backend relying on `std::vector` and RAII principles. 

I benchmarked both engines against PyTorch and NumPy across varying matrix sizes (N=10 to N=2000), stacking optimizations one by one. The performance journey was insane, dropping execution time from an abysmal 17 seconds down to just 0.20 seconds, officially dethroning PyTorch.

### Part 1: The Optimization Journey
I documented the entire process. If you want to see exactly how I stripped away abstractions to achieve bare metal performance, follow the main storyline:

- **[01. The Naive Baseline](reports/01_the_naive_baseline.md)**: Writing the standard O(N^3) triple for loop and discovering it was roughly 73x slower than PyTorch.
- **[02. Beating PyTorch](reports/02_beating_pytorch.md)**: Perfecting memory access and unlocking hardware threading via pthreads and OpenMP to finally push execution time down to 0.20s and secure the win.
- **[03. End-to-End ML Training](reports/03_end_to_end_ml_training.md)**: Proving that my optimized C engine can train a Multi-Layer Perceptron up to 2x faster than PyTorch's ATen backend by utilizing an $O(1)$ Arena memory checkpoint to eliminate page faults.
- **[04. Scaling to MNIST](reports/04_scaling_to_mnist.md)**: The Framework Overhead Reversal. Analyzing how scaling to a 60,000-image dataset completely dilutes PyTorch's framework overhead, allowing Intel MKL to strike back.
- **[05. Diagnosing Erratic Execution Times](reports/05_diagnosing_erratic_times.md)**: An analysis into why single-threaded CPU intensive workloads experience massive time jumps due to OS Thread Migration (E-Cores) and Thermal Throttling.
- **[06. The Limits of Custom SIMD](reports/06_the_limits_of_custom_simd.md)**: A realization of the "BLAS Wall". Understanding why a custom AVX2 SIMD implementation cannot beat the hand-tuned assembly micro-kernels and GotoBLAS memory packing of industry-standard libraries like Intel MKL and OpenBLAS.

### Part 2: Hardware & OS Deep Dives
For those interested in the raw hardware metrics and OS-level interactions, I've compiled my extra profiling data into dedicated deep dives:

- **[A1. Cache Misses & Tiling](reports/A1_cache_misses_and_tiling.md)**: Hooking into the Linux `perf_event_open` syscall to prove that simply reading memory in the wrong order caused 1 billion cache misses, and how matrix tiling fixed it.
- **[A2. OS Jitter & Allocators](reports/A2_os_jitter_and_allocators.md)**: Bypassing the compiler with AVX SIMD intrinsics, and using `getrusage()` to expose the hidden cost of high-level abstractions like `std::vector` (thousands of minor page faults) compared to a raw C bump allocator (zero).
- **[A3. Power Consumption Analysis](reports/A3_power_consumption_analysis.md)**: Utilizing the Linux RAPL interface to measure the exact microjoule energy cost of training. Proving the "Race to Sleep" concept and demonstrating why optimization is inherently green.

### Project structure
- `/macrograd/`: The C engine source code
- `/minigrad/`: The C++ engine source code
- `/benchmarking/`: The raw C++ and Python benchmarking scripts, Chrome profiler trace generators, and JSONL data dumps
- `/reports/`: My technical writeups on the profiling discoveries
