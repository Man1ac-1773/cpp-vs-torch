# A7. The Roofline Model: Memory vs Compute

### 1. The Hypothesis
In hardware profiling, the "Roofline Model" is a framework used to determine if a workload is bottlenecked by the CPU's memory bus (Memory Bound) or the CPU's arithmetic registers (Compute Bound). 

Matrix multiplication has an "Arithmetic Intensity" of roughly $N/6$ (FLOPs per byte transferred). At small matrix sizes, you perform very little math per byte loaded. As $N$ scales, the arithmetic required scales exponentially faster than the memory required. My hypothesis was that I could map the transition point where PyTorch shifts from Memory Bound to Compute Bound, and prove mathematically why the Naive engine never gets there.

### 2. The Empirical Data
I explicitly calculated the achieved memory throughput (useful bytes transferred / execution time) and achieved compute throughput (FLOPs / execution time) for a multithreaded sweep at $N=2048$. 

This table perfectly maps the exact journey of optimization—from being completely strangled by the memory wall, to incrementally unlocking the compute capabilities of the CPU:

**The Roofline Progression (N=2048, Multi-threaded):**
| Engine | Paradigm | Achieved Memory Bandwidth | Achieved Compute |
| :--- | :--- | :---: | :---: |
| **C++** | Naive | `0.01 GB/s` | `6.2 GFLOPS` |
| **C++** | Tiled | `0.11 GB/s` | `37.4 GFLOPS` |
| **C++** | SIMD (AVX2) | `0.50 GB/s` | `171.8 GFLOPS` |
| **PyTorch** | Intel MKL | **`1.65 GB/s`** | **`606.0 GFLOPS` (Ceiling)** |

*(Note: PyTorch's scaling from N=512 up to N=2048 shows its memory bandwidth requirement dropping from 5.66 GB/s down to 1.65 GB/s as it successfully achieves peak arithmetic intensity).*

### 3. The Hardware Mechanism: Theoretical Absolute Ceilings
To prove the roofline, we must establish the physical limits of the test silicon (Intel Core i7-13650HX). 

As defined in the Methodology report, an AVX2 FMA instruction calculates 16 FLOPs per cycle. Across all 14 physical cores running at sustained boost frequencies, the **Absolute Theoretical Peak Compute** of this specific die is roughly **~816 GFLOPS**. 

Looking at the PyTorch data, as the matrix scales from $N=512$ to $N=2048$, the achieved Memory Bandwidth physically *drops* (from 5.66 GB/s down to 1.65 GB/s). The data chunks fit efficiently inside the cache, the memory bus goes quiet, and the CPU's arithmetic logic units hit their physical limit. At 606 GFLOPS, PyTorch is achieving an incredible **74% of the absolute theoretical silicon limit**.

The Naive algorithm, however, destroys its cache lines (as proven in A1). It is desperately pulling data across the memory bus for every single iteration. As $N$ scales, the memory throughput completely collapses to `0.01 GB/s` due to extreme bus contention, and the compute stalls out at a miserable `6.2 GFLOPS` (almost exactly 100x slower than PyTorch).

### 4. The Code Proof (Arithmetic Intensity)
The roofline shift is dictated by the Arithmetic Intensity formula:
$$ \text{Total FLOPs} = 2 \times N^3 $$
$$ \text{Bytes Transferred} = 3 \times N^2 \times 4 \text{ bytes (floats)} $$
$$ \text{Arithmetic Intensity} = \frac{2N^3}{12N^2} = \frac{N}{6} \text{ FLOPs/Byte} $$

When $N$ is small, the intensity is low (Memory Bound). When $N$ is large, the intensity is massive (Compute Bound).

### 5. The Verdict
This perfectly illustrates that scaling hardware (adding threads or utilizing wider vector registers) is utterly useless if the algorithm cannot satisfy the Roofline Model's arithmetic intensity requirements. The naive algorithm never even touches the compute ceiling; it is strangled by the memory wall. PyTorch's MKL bypasses the memory wall through optimal blocking, allowing it to extract 74% of the silicon's physical potential.
