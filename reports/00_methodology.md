# 00. Methodology & Hardware Environment

Before diving into the low-level optimizations and performance anomalies, it is critical to define the exact hardware and software environment used to generate these results. System-level benchmarking is deeply intertwined with the specific silicon it runs on. A "memory bottleneck" on one architecture might be a "compute bottleneck" on another. 

To ensure full transparency and reproducibility, here are the exact specifications of the machine I used for this project.

### Hardware Specifications

| Component | Specification |
| :--- | :--- |
| **CPU Architecture** | x86_64 |
| **CPU Model** | 13th Gen Intel(R) Core(TM) i7-13650HX |
| **Topology** | 14 Cores (6 Performance-Cores, 8 Efficiency-Cores), 20 Threads |
| **Max Turbo Frequency** | 4.90 GHz (P-Cores) |
| **SIMD Support** | AVX, AVX2, FMA (256-bit registers) |
| **L1 Data Cache (L1d)** | 544 KiB total |
| **L1 Instruction Cache (L1i)**| 704 KiB total |
| **L2 Cache** | 11.5 MiB total |
| **L3 Cache** | 24 MiB |

### Software & Measurement Tooling

- **Operating System:** Linux (Ubuntu/Debian-based environment)
- **Compilers:** Standard `gcc` and `g++` (utilizing `-O3 -march=native` for release builds, unless otherwise specified in the Compiler Flags appendix).
- **Execution Time Measurement:** All wall-clock execution times were measured using high-resolution monotonic clocks (`std::chrono::high_resolution_clock` or `clock_gettime(CLOCK_MONOTONIC)`).
- **Hardware Counter Profiling:** To extract exact hardware events (like Cache Misses, Branch Misses, and CPU Instructions retired), I wrote a custom C profiler using the Linux `perf_event_open` syscall. This allowed me to read the exact CPU Performance Monitoring Unit (PMU) registers programmatically during execution, bypassing the need for external `perf` CLI wrappers and ensuring my Python and C/C++ benchmarks were measured identically.
- **Memory Jitter Tracking:** To track minor page faults and OS-level memory jitter, I utilized the Linux `getrusage()` syscall (`RUSAGE_SELF`).

### Theoretical Hardware Limits

To properly understand the "Roofline Model" (covered in Appendix A7), we must calculate the theoretical absolute peak performance of this specific CPU.

For an Intel i7-13650HX utilizing **AVX2** (256-bit registers) and **FMA** (Fused Multiply-Add):
- A 256-bit register holds **8 single-precision floats**.
- FMA performs a multiply and an add simultaneously, yielding **2 floating-point operations (FLOPs) per cycle** per lane.
- Therefore, each core can execute $8 \times 2 = 16$ FLOPs per clock cycle (assuming 1 vector execution port).

If we assume a workload is scheduled purely on the 6 Performance-Cores boosting to an average of ~4.5 GHz during a heavy sustained all-core AVX workload:
$\text{Theoretical Max GFLOPS} \approx 6 \text{ Cores} \times 4.5 \text{ GHz} \times 16 \text{ FLOPs/cycle} \approx \mathbf{432 \text{ GFLOPS}}$

If the OS scheduler perfectly utilizes all 14 cores (including the 8 E-cores running at a lower sustained frequency, ~3.0 GHz):
$\text{E-Core GFLOPS} \approx 8 \text{ Cores} \times 3.0 \text{ GHz} \times 16 \text{ FLOPs/cycle} \approx \mathbf{384 \text{ GFLOPS}}$

**Combined Theoretical Peak:** $\approx \mathbf{816 \text{ GFLOPS}}$

Throughout these reports, I will compare my custom C/C++ engines and PyTorch against this theoretical physical limitation.
