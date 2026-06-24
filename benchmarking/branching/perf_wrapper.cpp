#include "../common/perf_profiler.h"

extern "C" {
    void* perf_init_branches() {
        PerfCounter* pc = new PerfCounter();
        perf_init_branch_instructions(pc);
        return (void*)pc;
    }

    void* perf_init_misses() {
        PerfCounter* pc = new PerfCounter();
        perf_init_branch_misses(pc);
        return (void*)pc;
    }

    void perf_start_c(void* pc) {
        perf_start((PerfCounter*)pc);
    }

    long long perf_read_c(void* pc) {
        long long val = perf_read((PerfCounter*)pc);
        delete (PerfCounter*)pc;
        return val;
    }
}
