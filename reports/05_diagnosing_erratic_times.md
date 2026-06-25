# 05. Diagnosing Erratic Single-Threaded Execution Times

### 1. The Anomaly
During my final deep learning benchmarks, I observed a bizarre phenomenon in both my C and C++ single-threaded backends. Even though the mathematical workload per epoch was perfectly identical, the execution time would wildly fluctuate:

```text
--- C++ Backend: naive ---
Epoch 0 | Compute Time: 6.48281s
Epoch 1 | Compute Time: 20.5384s <-- 20s jump!
Epoch 2 | Compute Time: 3.45021s
Epoch 3 | Compute Time: 12.6251s
Epoch 7 | Compute Time: 1.01255s <-- Back down to 1s
Epoch 8 | Compute Time: 1.0073s
```

Meanwhile, the OpenMP `simd` multi-threaded backend ran at a perfectly consistent `~0.21s` per epoch. Why did my single-threaded engine randomly take **18x longer** on some epochs? 

### 2. The Hypotheses
In systems programming on modern laptops, severe single-threaded jitter like this is almost always caused by three hardware/OS factors:

**Hypothesis A: The Thermal Throttling Cycle**
Modern laptop CPUs have extremely aggressive thermal envelopes. 
1. **The Boost**: During Epochs 0 and 1, the CPU boosted a single core to maximum turbo frequency (e.g. 4.9 GHz), finishing the epoch fast.
2. **The Throttle**: By Epoch 2, the silicon hit its critical thermal limit (e.g., 100°C). The firmware panicked, dropped the CPU into a lower power state (PL1), and stripped the core clock down to base speeds (e.g. 800MHz) to cool off. The epoch took 20s.
3. **The Hysteresis**: Running at a low clock speed for 20 seconds allowed the die to cool. The firmware removed the throttle, and subsequent epochs boosted back to 4.9 GHz.

**Hypothesis B: P-Cores vs E-Cores (Thread Migration)**
My Intel i7-13650HX uses a hybrid design with high-performance cores (P-cores) and high-efficiency cores (E-cores). The OS scheduler often detects long-running, CPU-intensive single threads and classifies them as "background tasks," violently migrating them to an E-core to keep the laptop UI responsive. E-cores are significantly weaker at raw math. When migrated, the epoch took 20 seconds. When migrated back to a P-core, it took 1 second. 

**Hypothesis C: Subnormal (Denormal) Floating Point Values**
In deep learning architectures, weights and gradients can become exceptionally close to zero (e.g., `1e-40`). Standard Floating Point Units (FPUs) cannot natively handle these "subnormals". Instead of executing in 1 CPU cycle, the FPU traps to a software microcode interrupt, which takes 10x to 100x longer to compute. 

*(Note: The OpenMP `simd` backend likely avoided this because it spawned threads across all CPU cores—forcing the OS to engage the P-cores and finishing too quickly to trigger thermal throttling—and AVX registers often have Flush-To-Zero flags enabled by default).*

### 3. Future Investigative Work
I have not yet definitively proven which of these three hypotheses is the true root cause. To do so, I have outlined the following rigorous experimental pipeline for future work:

1. **Testing Thread Migration:** I will pin the single-threaded execution strictly to P-Core 0 using the Linux `taskset` command (or `pthread_setaffinity_np`). If the 20-second jumps still occur, the OS scheduler is exonerated.
2. **Testing Subnormals:** I will recompile the `naive` backend using the `-ffast-math` GCC flag. This enables Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ) in the CPU control registers. If the jumps disappear, FPU microcode traps were the culprit.
3. **Testing Thermal Throttling:** I will run the benchmark while simultaneously polling the Linux `turbostat` utility or `dmesg` to monitor real-time core frequencies and temperature limit flags.

I will come back to this investigation in a future update to definitively solve the mystery of the erratic epoch times.
