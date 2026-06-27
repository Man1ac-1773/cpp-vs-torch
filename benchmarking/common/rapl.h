#pragma once
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>

// helper to read rapl energy counters (joules)
static inline double get_rapl_energy_joules() {
    int fd = open("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", O_RDONLY);
    if (fd < 0) {
        return 0.0;
    }
    
    char buf[64];
    int n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n < 0) return 0.0;
    
    buf[n] = '\0';
    uint64_t uj = 0;
    sscanf(buf, "%lu", &uj);
    return (double)uj / 1e6; // convert microjoules to joules
}
