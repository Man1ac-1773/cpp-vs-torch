#ifndef PERF_PROFILER_H
#define PERF_PROFILER_H

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <sys/ioctl.h>

// Helper to invoke the syscall
static inline long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                                   int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

typedef struct {
    int fd;
} PerfCounter;

// Initializes a counter to track L1 Data Cache Misses
static inline void perf_init_l1_misses(PerfCounter* pc) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.type = PERF_TYPE_HW_CACHE;
    pe.size = sizeof(struct perf_event_attr);
    
    // Config for: L1 Data Cache -> Read Operation -> Result is a Miss
    pe.config = (PERF_COUNT_HW_CACHE_L1D) |
                (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
                
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    pc->fd = perf_event_open(&pe, 0, -1, -1, 0);
    if (pc->fd == -1) {
        fprintf(stderr, "Warning: Failed to open perf_event for L1 Cache Misses. You may need to run:\n");
        fprintf(stderr, "sudo sysctl -w kernel.perf_event_paranoid=-1\n");
    }
}

// Resets and starts counting
static inline void perf_start(PerfCounter* pc) {
    if (pc->fd == -1) return;
    ioctl(pc->fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(pc->fd, PERF_EVENT_IOC_ENABLE, 0);
}

// Stops counting and returns the number of misses
static inline long long perf_read(PerfCounter* pc) {
    if (pc->fd == -1) return -1;
    ioctl(pc->fd, PERF_EVENT_IOC_DISABLE, 0);
    long long count;
    if (read(pc->fd, &count, sizeof(long long)) == -1) {
        return -1;
    }
    return count;
}

#endif
