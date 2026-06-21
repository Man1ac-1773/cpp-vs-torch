# 04. Beating PyTorch

Even after AVX SIMD intrinsics, PyTorch was still destroying my custom engines. PyTorch finished the N=2000 benchmark in 0.23 seconds, while my best single-threaded C kernel took 1.78 seconds.

PyTorch was winning because it leverages all 16 cores of my laptop by default. So I decided to level the playing field.

I implemented multithreading in two different ways to compare paradigms:
1. C Engine: I hand-rolled `pthreads`, spawning 8 worker threads and manually partitioning the matrix rows into chunks.
2. C++ Engine: I used OpenMP compiler directives (`#pragma omp parallel for`), allowing the compiler to dynamically distribute loop iterations.

I ran the N=2000 benchmark again. Here is the data:
- C (`pthreads`): 0.25s 
- C++ (`OpenMP`): 0.20s
- PyTorch: 0.23s

I actually managed to dethrone PyTorch lmao. 

By unlocking hardware threading, the C++ OpenMP kernel finished in 0.20s. 

It was also super interesting to see C++ beat C here. My manual C `pthread` implementation statically partitioned the matrix into 8 equal chunks. Since the OS is noisy, if one core gets throttled, that thread finishes last and the whole `pthread_join` blocks waiting for it (the classic straggler problem).

OpenMP uses dynamic work stealing. As cores finish their chunks early, they dynamically pick up the remaining matrix rows, completely eliminating the straggler problem. This allowed the C++ engine to achieve a near-perfect 9.4x linear speedup and secure the win.
