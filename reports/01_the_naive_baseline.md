# 01. The Naive Baseline

I started this project by implementing the classic O(N^3) triple `for` loop matrix multiplication in both C and C++. I wanted to see just how slow unoptimized code really is compared to industry titans.

I ran a sweep from N=10 to N=2000 and dumped the data. 

For N=2000, here are the numbers:
- PyTorch: 0.23s
- NumPy: 0.22s
- My C Engine: 16.48s
- My C++ Engine: 16.92s

It took nearly 17 seconds to multiply two 2000x2000 matrices lmao. The gap between my unoptimized code and PyTorch was roughly 73x. 

The primary reason for this massive delay isn't just the math, it's how the CPU interacts with memory. Matrix multiplication in a naive nested loop iterates over the columns of the second matrix sequentially, meaning the CPU is constantly jumping around in physical memory rather than reading contiguous blocks. This causes an insane amount of cache misses. 

To fix this, I needed to step in and restructure the memory access patterns manually.
