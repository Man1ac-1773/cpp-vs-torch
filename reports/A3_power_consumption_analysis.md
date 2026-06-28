# A3. Power Consumption Analysis

While execution speed is the primary metric in machine learning benchmarks, power consumption is equally critical, especially when scaling deployments to data centers. 

To measure the true energy cost of these different implementations, I utilized the Linux **Running Average Power Limit (RAPL)** interface (`/sys/class/powercap/intel-rapl`), which provides microjoule-accurate energy readings directly from the CPU package.

I tracked the total energy consumed across the full 10-epoch training run of the MNIST dataset using Mini-Batch SGD.

### Energy Consumption (MNIST, 10 Epochs)
*(Data recorded using the `performance-plugged` battery profile).*

| Engine | Backend | Total Energy Consumed (Joules) |
|---|---|---|
| **C (`macrograd`)** | Naive | `4,764.9 J` |
| **C (`macrograd`)** | Tiled | `5,133.1 J` |
| **C++ (`minigrad`)** | SIMD (OpenMP) | `175.1 J` |
| **C (`macrograd`)** | SIMD (OpenMP) | `144.7 J` |
| **PyTorch (ATen)** | MKL | `132.0 J` |
| **NumPy (OpenBLAS)** | OpenBLAS | `122.0 J` |

### The Energy Verdict

1. **The Cost of Slowness:** The `naive` and `tiled` backends consumed massive amounts of energy (roughly 5,000 Joules). Because they lacked SIMD and multi-threading, they forced the CPU to stay active for exceptionally long periods. 
2. **The Efficiency of SIMD & BLAS:** The multithreaded `simd` implementations engaged all cores (drawing high peak power), but finished the epochs so fast that total energy plummeted to ~150 Joules. This perfectly illustrates the concept of **Race to Sleep**: it is far more energy-efficient to blast the CPU with power to finish quickly than to sip power and run forever.
3. **The BLAS Wall Strikes Again:** NumPy and PyTorch utilized highly optimized BLAS libraries, finishing in fractions of a second. They are undeniably the most energy-efficient implementations tested.

### The Impact of OS Power Profiles
A massive mistake in early benchmarking is assuming energy consumption is solely dictated by the code. The Operating System's power governor plays a massive role. 

Let's look at the raw energy consumed by PyTorch for a single $N=1000$ matrix multiplication across different laptop battery profiles:

| OS Battery Profile | Execution Time | Energy Consumed |
|---|---|---|
| **`performance-plugged`** | ~0.002s | `0.86 Joules` |
| **`balanced`** | ~0.003s | `0.35 Joules` |

By switching from `performance` to `balanced`, the OS actively parks the P-Cores, throttles maximum turbo frequencies, and shifts background work to E-Cores. The execution time slowed down slightly (from 2ms to 3ms), but the total energy consumed dropped by **nearly 60%**! 

When running benchmarks, failing to lock or disclose the active battery profile completely invalidates energy comparisons, as the OS can drastically alter the hardware's electrical behavior beneath your code.

*(Note: There is also a major energy anomaly specifically tied to the threading paradigms used in the training loop. This is explored fully in **[A8. OpenMP vs Pthreads](A8_openmp_vs_pthreads.md)**).*
