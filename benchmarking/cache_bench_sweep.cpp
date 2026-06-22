#include <iostream>
#include <fstream>
#include "./../macrograd/src/macrograd.h"
#include "perf_profiler.h"

using namespace std;

int main() {
    PerfCounter l1_counter;
    PerfCounter instr_counter;
    perf_init_l1_misses(&l1_counter);
    perf_init_instructions(&instr_counter);

    ofstream json_out("./c-data/pmu_metrics.jsonl");

    cout << "Sweeping PMU metrics (N=10 to N=1000)..." << endl;
    
    uint step = 10;
    for (size_t N = 10; N <= 1000; N += step) {
        // --- NAIVE ---
        g_arena.top = 0;
        Tensor* A1 = new_tensor(N, N);
        Tensor* B1 = new_tensor(N, N);
        
        perf_start(&l1_counter);
        perf_start(&instr_counter);
        Tensor* c_naive = tensor_matmul_naive(A1, B1);
        long long naive_misses = perf_read(&l1_counter);
        long long naive_instr = perf_read(&instr_counter);
        
        json_out << "{\"N\": " << N << ", \"algorithm\": \"naive\", \"metric\": \"l1_misses\", \"value\": " << naive_misses << "}\n";
        json_out << "{\"N\": " << N << ", \"algorithm\": \"naive\", \"metric\": \"instructions\", \"value\": " << naive_instr << "}\n";

        // --- TILED ---
        g_arena.top = 0;
        Tensor* A2 = new_tensor(N, N);
        Tensor* B2 = new_tensor(N, N);
        
        perf_start(&l1_counter);
        perf_start(&instr_counter);
        Tensor* c_tiled = tensor_matmul_tiled(A2, B2);
        long long tiled_misses = perf_read(&l1_counter);
        long long tiled_instr = perf_read(&instr_counter);
        
        json_out << "{\"N\": " << N << ", \"algorithm\": \"tiled\", \"metric\": \"l1_misses\", \"value\": " << tiled_misses << "}\n";
        json_out << "{\"N\": " << N << ", \"algorithm\": \"tiled\", \"metric\": \"instructions\", \"value\": " << tiled_instr << "}\n";

        // --- SIMD ---
        g_arena.top = 0;
        Tensor* A3 = new_tensor(N, N);
        Tensor* B3 = new_tensor(N, N);
        
        perf_start(&l1_counter);
        perf_start(&instr_counter);
        Tensor* c_simd = tensor_matmul_simd(A3, B3);
        long long simd_misses = perf_read(&l1_counter);
        long long simd_instr = perf_read(&instr_counter);
        
        json_out << "{\"N\": " << N << ", \"algorithm\": \"simd\", \"metric\": \"l1_misses\", \"value\": " << simd_misses << "}\n";
        json_out << "{\"N\": " << N << ", \"algorithm\": \"simd\", \"metric\": \"instructions\", \"value\": " << simd_instr << "}\n";

        if (step == 10 && N >= 100) step = 100;
    }
    
    json_out.close();
    cout << "Done! Results saved to ./c-data/pmu_metrics.jsonl" << endl;
    return 0;
}
