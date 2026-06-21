#include <iostream>
#include "./../macrograd/src/macrograd.h"
#include "perf_profiler.h"

using namespace std;

int main() {
    PerfCounter l1_counter;
    perf_init_l1_misses(&l1_counter);

    size_t N = 1000;
    cout << "Tracking L1 Cache Misses for N=" << N << "..." << endl;
    
    Tensor* A = new_tensor(N, N);
    Tensor* B = new_tensor(N, N);
    
    perf_start(&l1_counter);
    Tensor* c_naive = tensor_matmul_naive(A, B);
    long long naive_misses = perf_read(&l1_counter);

    g_arena.top = 0;
    Tensor* A2 = new_tensor(N, N);
    Tensor* B2 = new_tensor(N, N);

    perf_start(&l1_counter);
    Tensor* c_tiled = tensor_matmul_tiled(A2, B2);
    long long tiled_misses = perf_read(&l1_counter);

    cout << "\n=== L1 Data Cache Misses (N=" << N << ") ===" << endl;
    cout << "Naive Matmul: " << naive_misses << " misses" << endl;
    cout << "Tiled Matmul: " << tiled_misses << " misses" << endl;
    
    return 0;
}
