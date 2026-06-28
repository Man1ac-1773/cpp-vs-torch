# 01. The Naive Baseline

I started this project by implementing the classic O(N^3) triple `for` loop matrix multiplication in both C and C++. I wanted to see just how slow unoptimized code really is compared to industry titans like PyTorch and NumPy.

The baseline algorithm is simple: for every element in the output matrix, compute the dot product of the corresponding row from matrix A and column from matrix B. 

I ran a sweep from N=10 to N=2000 and dumped the data. 

### The N=2000 Execution Times

*(Note: All execution times below were captured on the `performance-plugged` battery profile to ensure the CPU is running at its maximum frequency without OS-level power throttling.)*

| Engine | Language | Paradigm | Time (Seconds) | Relative Speed |
|---|---|---|---|---|
| **PyTorch** | C++ (ATen) | Highly Optimized | 0.115s | 1.0x (Baseline) |
| **NumPy** | C (OpenBLAS) | Highly Optimized | 0.112s | 1.02x |
| **My C++ Engine** | C++ | Naive O(N^3) | 8.48s | ~73x Slower |
| **My C Engine** | C | Naive O(N^3) | 9.73s | ~84x Slower |

It took roughly 8.5 to 9.7 seconds to multiply two 2000x2000 matrices using the naive implementation. The gap between my unoptimized C++ code and PyTorch was massive—roughly 73x slower! 

### Why is it so slow?

The primary reason for this massive delay is not just the sheer number of math operations ($2000^3$ multiplications), it is how the CPU interacts with memory.

When the naive nested loop iterates over the columns of the second matrix, the CPU is constantly jumping around in physical memory. Since matrices are stored in row-major order, reading down a column means skipping thousands of memory addresses for every single element. 

**Key Learning:** Modern CPUs are heavily optimized for sequential memory access. When you skip around, the CPU's pre-fetcher cannot predict what memory you need next. This causes a phenomenon known as "Cache Thrashing", where the CPU constantly waits for data to be fetched from slow main memory instead of having it readily available in the ultra-fast L1 cache.

To fix this disaster, I needed to step in and restructure the memory access patterns manually.

### Scripts and Raw Data

The benchmarking logic for this phase can be found in [`c_matmul_single.cpp`](../benchmarking/matmul/c_matmul_single.cpp), [`cpp_matmul_single.cpp`](../benchmarking/matmul/cpp_matmul_single.cpp), and [`py_matmul_single.py`](../benchmarking/matmul/py_matmul_single.py). The raw sweeping data used to generate these insights is logged in the [`benchmarking/matmul/data/`](../benchmarking/matmul/data/) directory.
