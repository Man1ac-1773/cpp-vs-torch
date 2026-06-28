# 05. Diagnosing Erratic Single-Threaded Execution Times

### The Anomaly
During my final deep learning benchmarks, I observed an interesting phenomenon in my single-threaded backend. Even though the mathematical workload per epoch was perfectly identical, the execution time would occasionally spike:

```text
--- C++ Backend: naive ---
Epoch 18 | Compute Time: 0.2407s
Epoch 19 | Compute Time: 0.2340s
Epoch 20 | Compute Time: 0.5031s <-- 2x jump! (thermal_throttle)
...
Epoch 49 | Compute Time: 0.2340s
Epoch 50 | Compute Time: 0.8707s <-- Almost 4x jump! (ecore_migration)
...
Epoch 62 | Compute Time: 0.2306s
Epoch 63 | Compute Time: 0.4091s <-- Jump! (os_jitter)
Epoch 64 | Compute Time: 0.2388s 
```

Meanwhile, the OpenMP `simd` multi-threaded backend ran at a perfectly consistent epoch time. Why did my single-threaded engine randomly take up to **4x longer** on some epochs? 

### The OS-Level Explanations
In systems programming on modern laptops, severe single-threaded jitter like this is caused by a few distinct hardware/OS factors. I instrumented the benchmarking script to log the exact system events causing the spikes.

**A. The Thermal Throttling Cycle (Epoch 20)**
Modern laptop CPUs have extremely aggressive thermal envelopes. By Epoch 20, the silicon hit its critical thermal limit. The firmware panicked, dropped the CPU into a lower power state (PL1), and stripped the core clock down to base speeds to cool off. The epoch took 0.50s. After running at a low clock speed for a few epochs, the die cooled, the firmware removed the throttle, and subsequent epochs boosted back up.

**B. P-Cores vs E-Cores Thread Migration (Epoch 50)**
My Intel CPU uses a hybrid design with high-performance cores (P-cores) and high-efficiency cores (E-cores). The OS scheduler often detects long-running, CPU-intensive single threads and classifies them as "background tasks," violently migrating them to an E-core to keep the laptop UI responsive. E-cores are significantly weaker at raw math. When migrated at Epoch 50, the execution time spiked to 0.87 seconds!

**C. OS Jitter and Interrupts (Epoch 63)**
At Epoch 63, the execution time jumped to 0.40s. This was caused by standard OS Jitter. A modern operating system is constantly running hundreds of background threads. Periodically, the OS will interrupt the CPU core running the benchmark to handle a hardware interrupt, a network packet, or a background process context switch. This steals CPU cycles away from the matrix math, causing a temporary spike in epoch time.

*(Note: The OpenMP `simd` backend avoided these massive migrations because it spawned threads across all CPU cores—forcing the OS to engage the P-cores and effectively treating the workload as a high-priority parallel compute task).*

### Key Takeaway
When benchmarking micro-kernels in C/C++, you cannot just look at the `mean` average execution time. The operating system is a highly noisy environment that actively manipulates CPU frequency and core assignments. You must always record the `min` time (the true hardware limit) and analyze the variance to understand OS scheduling behavior.
