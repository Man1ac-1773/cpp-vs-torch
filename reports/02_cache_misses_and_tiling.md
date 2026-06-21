# 02. Cache Misses and Tiling

To fix the 17-second disaster from the naive implementation, I implemented Matrix Tiling. I broke the large matrices down into small 32x32 blocks. 

This dropped the execution time by roughly 2x in both my C and C++ engines. But I wanted to prove exactly why this happened at the hardware level.

I tapped directly into the CPU's Performance Monitoring Unit (PMU) using the Linux `perf_event_open` syscall. I instructed the CPU to track every single time it tried to fetch data from the L1 Data Cache and failed, forcing a stall.

I ran the benchmark for an N=1000 matrix. Here is the data:
- Naive Matmul: 996,030,754 L1 cache misses
- Tiled Matmul: 3,409,932 L1 cache misses

The naive algorithm triggered nearly a billion cache misses. Tiling reduced this by 292x down to just 3.4 million. By making sure the 32x32 blocks fit perfectly inside the CPU's ultra-fast 32KB L1 cache, I completely bypassed the main memory bottleneck. 

It is actually crazy how much performance you lose just by reading memory in the wrong order lmao.
