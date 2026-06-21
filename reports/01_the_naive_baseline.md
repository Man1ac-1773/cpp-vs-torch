# 01. The Naive Baseline

I started this project by implementing the classic O(N^3) triple `for` loop matrix multiplication in both C and C++. I wanted to see just how slow unoptimized code really is compared to industry titans like PyTorch and NumPy.

The baseline algorithm is simple: for every element in the output matrix, compute the dot product of the corresponding row from matrix A and column from matrix B. 

I ran a sweep from N=10 to N=2000 and dumped the data. 

### The N=2000 Execution Times

| Engine | Language | Paradigm | Time (Seconds) | Relative Speed |
|---|---|---|---|---|
| **PyTorch** | C++ (ATen) | Highly Optimized | 0.23s | 1.0x (Baseline) |
| **NumPy** | C (OpenBLAS) | Highly Optimized | 0.22s | 1.04x |
| **My C Engine** | C | Naive O(N^3) | 16.48s | ~71x Slower |
| **My C++ Engine** | C++ | Naive O(N^3) | 16.92s | ~73x Slower |

It took nearly 17 seconds to multiply two 2000x2000 matrices lmao. The gap between my unoptimized code and PyTorch was roughly 73x. 

### Why is it so slow?

The primary reason for this massive delay is not just the sheer number of math operations ($2000^3$ multiplications), it is how the CPU interacts with memory.

When the naive nested loop iterates over the columns of the second matrix, the CPU is constantly jumping around in physical memory. Since matrices are stored in row-major order, reading down a column means skipping thousands of memory addresses for every single element. 

**Key Learning:** Modern CPUs are heavily optimized for sequential memory access. When you skip around, the CPU's pre-fetcher cannot predict what memory you need next. This causes a phenomenon known as "Cache Thrashing", where the CPU constantly waits for data to be fetched from slow main memory instead of having it readily available in the ultra-fast L1 cache.

To fix this 17-second disaster, I needed to step in and restructure the memory access patterns manually.

### Scripts and Raw Data

The benchmarking logic for this phase can be found in `../benchmarking/benchmark.cpp` and `../benchmarking/benchmark.py`. The raw sweeping data used to generate these insights is logged in `../benchmarking/c-data/`, `../benchmarking/cpp-data/`, and `../benchmarking/py-data/`.
