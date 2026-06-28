# CPP-vs-Torch: Outpacing PyTorch from scratch

**[Check out the live deployment](https://cpp-vs-torch.vercel.app/)**


Everyone uses PyTorch and NumPy, but very few people actually know what happens under the hood when you multiply two tensors. I decided to build two deep learning engines completely from scratch just to see how deep the rabbit hole goes, and to answer a single question.

### Can a solo dev build a matrix multiplication engine in raw C/C++ that beats PyTorch? 

Spoiler alert: I did beat PyTorch. But only on a very specific task lmao. Read more to find out!

This repository contains two backend engines built entirely from the ground up:
- `macrograd`: A pure C backend utilizing a custom Bump Arena Allocator. [Credits](https://github.com/Celibistrial/macrograd)
- `minigrad`: A C++ backend relying on `std::vector` and RAII principles. 

I benchmarked both engines against PyTorch and NumPy across varying matrix sizes (N=10 to N=2000), stacking optimizations one by one. The performance journey was insane, dropping execution time from an abysmal 9.7 seconds down to just 0.136 seconds, proving the incredible power of bare metal optimization.

### Part 1: The Optimization Journey
I documented the entire process. If you want to see exactly how I stripped away abstractions to achieve bare metal performance, follow the main storyline:

- **[00. Methodology](reports/00_methodology.md)**: Defining the exact hardware environment (i7-13650HX, AVX2) and mathematically calculating the absolute physical compute limits of my silicon.
- **[01. The Naive Baseline](reports/01_the_naive_baseline.md)**: Writing the standard O(N^3) triple for loop and discovering it was roughly 84x slower than PyTorch.
- **[02. Beating PyTorch](reports/02_beating_pytorch.md)**: Perfecting memory access and unlocking hardware threading via pthreads and OpenMP to finally push execution time down to 0.136s (almost beating single-threaded PyTorch).
- **[03. End-to-End ML Training](reports/03_end_to_end_ml_training.md)**: Proving that massive ML frameworks carry significant baseline overhead, allowing NumPy and my custom C++ engine to drastically outperform PyTorch on tiny Multi-Layer Perceptrons.
- **[04. Scaling to MNIST](reports/04_scaling_to_mnist.md)**: The Framework Overhead Reversal. Analyzing how scaling to a 60,000-image dataset completely dilutes PyTorch's framework overhead (GIL, dynamic ATen graphs, dispatcher), allowing Intel MKL to strike back.
- **[05. Diagnosing Erratic Execution Times](reports/05_diagnosing_erratic_times.md)**: Laying out an experimental pipeline to diagnose why single-threaded CPU intensive workloads experience 4x latency spikes (investigating Thermal Throttling, E-Cores thread migration, and OS Jitter).
- **[06. The Limits of Custom SIMD](reports/06_the_limits_of_custom_simd.md)**: A realization of the "BLAS Wall". Understanding why a custom AVX2 SIMD implementation cannot beat the hand-tuned assembly micro-kernels and GotoBLAS memory packing of industry-standard libraries like Intel MKL.

### Part 2: Hardware & OS Deep Dives
For those interested in the raw hardware mechanisms, Cache Lines, and OS-level virtual memory interactions, I've compiled my extra profiling data into dedicated systems-engineering deep dives:

- **[A1. Cache Misses & Tiling](reports/A1_cache_misses_and_tiling.md)**: Hooking into the Linux `perf_event_open` syscall to mathematically prove how column-major traversal wastes 93.75% of cache line bandwidth, causing 1 billion cache misses, and how matrix tiling fixed it.
- **[A2. OS Jitter & Allocators](reports/A2_os_jitter_and_allocators.md)**: Exposing the hidden cost of Demand Paging. Proving why `std::vector` (which maps virtual pages lazily and faults on write) triggers 15,593 minor page faults compared to a pre-faulted C bump allocator (zero).
- **[A3. Power Consumption Analysis](reports/A3_power_consumption_analysis.md)**: Utilizing the Linux RAPL interface to measure the exact microjoule energy cost of training. Proving the "Race to Sleep" concept and demonstrating why optimization is inherently green.
- **[A4. The Branch Predictor & Loop Unrolling](reports/A4_branch_predictor_and_loop_unrolling.md)**: Proving that PyTorch executes **8.5 Billion fewer branches** than standard C++ loops at scale, while explaining the dangerous tradeoff of overflowing the Instruction Cache (L1i) with massive loop unrolling.
- **[A5. Compiler Optimization Flags](reports/A5_compiler_optimization_flags.md)**: An A/B test of GCC flags (`-O0` through `-Ofast`), proving how relaxing strict IEEE math compliance triggers auto-vectorization for a massive 5.0x speedup, and exposing the `-march=native` hardware trap.
- **[A6. Amdahl's Law and Scaling](reports/A6_amdahls_law_and_scaling.md)**: A thread-scaling sweep from 1 to 20 cores, proving how asymmetric CPU architectures (P-cores vs E-cores) cause immediate performance drops, and why memory-bound code physically cannot scale across threads.
- **[A7. The Roofline Model](reports/A7_memory_bandwidth_roofline.md)**: Calculating the physical GB/s limits of the hardware during execution, proving mathematically that PyTorch's 606 GFLOPS achieved ~74% of the absolute physical compute limit of the silicon.
- **[A8. OpenMP vs Pthreads](reports/A8_openmp_vs_pthreads.md)**: A deep dive into a bizarre 239 Joule energy spike, comparing the aggressive busy-wait spinlocks of Pthreads against OpenMP's efficient thread-sleeping condition variables during ML training.

### Project structure
- `/macrograd/`: The C engine source code
- `/minigrad/`: The C++ engine source code
- `/benchmarking/`: The raw C++ and Python benchmarking scripts, Chrome profiler trace generators, and JSONL data dumps
- `/reports/`: My technical writeups and systems-level proofs on the profiling discoveries
