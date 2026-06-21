#pragma once
#include <stdint.h>
#include <time.h>
#include <x86intrin.h>
// exact cpu cycles
static inline uint64_t get_cpu_cycles()
{
    return __rdtsc();
}

// time with nanosecond precision
static inline double get_wall_time()
{
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (double) time.tv_sec + (double) time.tv_nsec * 1e-9;
}
