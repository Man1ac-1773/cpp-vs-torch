# A3. Power Consumption Analysis

While execution speed is the primary metric in machine learning benchmarks, power consumption is equally critical, especially when scaling deployments to data centers. 

To measure the true energy cost of these different implementations, I utilized the Linux **Running Average Power Limit (RAPL)** interface (`/sys/class/powercap/intel-rapl`), which provides microjoule-accurate energy readings directly from the CPU package.

I tracked the total energy consumed across the full 10-epoch training run of the MNIST dataset using Mini-Batch SGD.

### Energy Consumption (MNIST, 10 Epochs)

| Engine | Backend | Total Energy Consumed (Joules) | Avg Power Draw (Watts) |
|---|---|---|---|
| **C (`macrograd`)** | Naive | `~12,823 J` | `~9.0 W` |
| **C (`macrograd`)** | Tiled | `10,934 J` | `~60.0 W` |
| **C (`macrograd`)** | SIMD (OpenMP) | `175 J` | `~69.0 W` |
| **PyTorch (ATen)** | MKL | `132 J` | `~43.0 W` |
| **NumPy (OpenBLAS)** | OpenBLAS | `122 J` | `~106.0 W` |

*(Note: The Average Power Draw is calculated as Total Energy / Execution Time. C++ numbers were nearly identical to C).*

### The Energy Verdict

1. **The Cost of Slowness:** The `naive` backend consumed a massive `12,823 Joules`. Even though its average power draw was incredibly low (`~9.0 Watts`) due to running strictly on a single core (and often being throttled or migrated to an E-core), it ran for over 23 minutes. In contrast, the `tiled` backend spiked the power draw to `~60 Watts` but finished in 3 minutes, saving roughly 2,000 Joules! This proves the concept of **Race to Sleep**: it is far more energy-efficient to blast the CPU with power and finish quickly than to sip power and run forever.
2. **The Efficiency of SIMD & BLAS:** My multithreaded `simd` implementation drew `~69 Watts` (engaging all cores) but finished so fast that it only consumed `175 J` in total.
3. **The BLAS Wall Strikes Again:** NumPy and PyTorch utilized highly optimized BLAS libraries, finishing in just 1 to 3 seconds. NumPy drew an astonishing `~106 Watts` (maximizing the entire CPU package's power delivery), but finished the run consuming only `122 J`, making it the most energy-efficient implementation tested!

**Key Learning:** Optimization is inherently green. Every abstraction I stripped away and every cache miss I eliminated directly translated into massive power savings. Writing efficient code doesn't just save time; it saves the battery.
