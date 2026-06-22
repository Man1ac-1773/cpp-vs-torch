#include <iostream>
#include <fstream>
#include "../../minigrad/src/tensor.h"
#include "../common/perf_profiler.h"

using namespace std;

int main(int argc, char* argv[]) {
    string mode = "performance-plugged";
    if (argc > 1) mode = argv[1];
    
    PerfCounter l1_counter;
    PerfCounter instr_counter;
    perf_init_l1_misses(&l1_counter);
    perf_init_instructions(&instr_counter);

    string filepath = "./data/cpp_cache_" + mode + ".jsonl";
    ofstream json_out(filepath);

    cout << "Sweeping PMU metrics for C++ Engine (N=10 to N=1000)..." << endl;
    
    uint step = 10;
    for (size_t N = 10; N <= 1000; N += step) {
        // --- C++ (std::vector) TILED ---
        Tensor A(N, N);
        Tensor B(N, N);
        
        perf_start(&l1_counter);
        perf_start(&instr_counter);
        Tensor c_tiled = matmul_tiled(A, B);
        long long tiled_misses = perf_read(&l1_counter);
        long long tiled_instr = perf_read(&instr_counter);
        
        json_out << "{\"N\": " << N << ", \"algorithm\": \"tiled\", \"metric\": \"l1_misses\", \"value\": " << tiled_misses << "}\n";
        json_out << "{\"N\": " << N << ", \"algorithm\": \"tiled\", \"metric\": \"instructions\", \"value\": " << tiled_instr << "}\n";

        // --- C++ (std::vector) SIMD ---
        Tensor A2(N, N);
        Tensor B2(N, N);
        
        perf_start(&l1_counter);
        perf_start(&instr_counter);
        Tensor c_simd = matmul_simd(A2, B2);
        long long simd_misses = perf_read(&l1_counter);
        long long simd_instr = perf_read(&instr_counter);
        
        json_out << "{\"N\": " << N << ", \"algorithm\": \"simd\", \"metric\": \"l1_misses\", \"value\": " << simd_misses << "}\n";
        json_out << "{\"N\": " << N << ", \"algorithm\": \"simd\", \"metric\": \"instructions\", \"value\": " << simd_instr << "}\n";

        if (step == 10 && N >= 100) step = 100;
    }
    
    json_out.close();
    cout << "Done! Results saved to " << filepath << endl;
    return 0;
}
