# Diagnosing Erratic Single-Threaded Execution Times

During our final deep learning benchmarks, we observed a bizarre phenomenon in the `naive` and `tiled` C engine backends. Even though the mathematical workload per epoch was perfectly identical, the execution time would wildly fluctuate:

```text
--- Backend: naive ---
Epoch 0 | Avg Loss: 2.27924 | Compute Time: 1.12779s
Epoch 1 | Avg Loss: 2.01334 | Compute Time: 1.071s
Epoch 2 | Avg Loss: 1.15308 | Compute Time: 18.5183s  <-- Massive slowdown!
Epoch 3 | Avg Loss: 0.722964 | Compute Time: 1.07889s <-- Fast again!
Epoch 4 | Avg Loss: 0.569836 | Compute Time: 12.2163s
Epoch 5 | Avg Loss: 0.489635 | Compute Time: 7.75477s
```

Meanwhile, the OpenMP `simd` backend ran at a perfectly consistent `~0.21s` per epoch.

Why did our single-threaded C engine randomly take **18x longer** on some epochs? In systems programming on modern laptops, this is almost always caused by three hardware/OS factors:

## 1. The Thermal Throttling Cycle
Modern laptop CPUs have extremely aggressive thermal envelopes. 
1. **The Boost**: During Epochs 0 and 1, the CPU boosted a single core to maximum turbo frequency (e.g. 5.0GHz), drawing massive power and finishing the epoch in `1.0s`.
2. **The Throttle**: By Epoch 2, the silicon hit its critical thermal limit (e.g., 100°C). The firmware panicked and dropped the CPU into a lower power state (PL1), stripping the core clock down to base speeds (e.g. 800MHz) to cool off. The epoch took `18.5s`.
3. **The Hysteresis**: Because it took 18 seconds running at a low clock speed, the die cooled down. The firmware removed the throttle, and Epoch 3 boosted back up to 5.0GHz and finished in `1.0s`. The cycle repeats.

## 2. P-Cores vs E-Cores (Thread Migration)
Modern architectures (like Intel 12th/13th Gen and Apple Silicon) use hybrid designs with high-performance cores (P-cores) and high-efficiency cores (E-cores).
The OS scheduler often detects long-running, CPU-intensive single threads and classifies them as "background tasks", violently migrating them to an E-core to keep the laptop UI responsive. 
E-cores are significantly weaker at raw math. When migrated, the epoch took 18 seconds. When migrated back to a P-core, it took 1 second. 

## 3. Subnormal (Denormal) Floating Point Values
In deep learning architectures with ReLUs, weights and gradients can become exceptionally close to zero (e.g., `1e-40`). Standard Floating Point Units (FPUs) cannot natively handle these "subnormals". Instead of executing in 1 CPU cycle, the FPU traps to a software microcode interrupt, which takes 10x to 100x longer to compute. 

## Why didn't `simd` suffer from this?
The `simd` backend uses OpenMP, which spawns threads across *all* available CPU cores. Because it utilizes the entire CPU package simultaneously, the OS scheduler is forced to engage the P-cores, and the workload finishes in just `0.21s`—far too quickly to saturate the thermal block and trigger aggressive thermal throttling! Additionally, SIMD vector registers usually have "Flush to Zero" (FTZ) flags inherently enabled, protecting them from the denormal microcode trap.
