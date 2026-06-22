#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>

#include "./../../macrograd/src/macrograd.h"
#include "../common/profiler.h"

using namespace std;

void run_single_thread_sweep(ofstream& json_out)
{
    uint step = 10;
    const int NUM_RUNS = 10;
    cout << " ========== Starting Single-threaded C Benchmarking ========= " << endl;
    for (size_t N = 10; N <= 2000; N += step)
    {
        double total_time = 0;
        double min_time = 1e9;
        uint64_t total_cycles = 0;
        uint64_t min_cycles = UINT64_MAX;
        
        Tensor* A = new_tensor(N, N);
        Tensor* B = new_tensor(N, N);
        for (size_t i = 0; i < N * N; i++) { A->data[i] = 1.0f; B->data[i] = 2.0f; }
        
        // 1. NAIVE
        Tensor* C_warmup = tensor_matmul_naive(A, B);
        for (int r = 0; r < NUM_RUNS; r++) {
            uint64_t cycles = get_cpu_cycles();
            double start_time = get_wall_time();
            Tensor* C_naive = tensor_matmul_naive(A, B);
            double time_elapsed = (get_wall_time() - start_time);
            cycles = (get_cpu_cycles() - cycles);
            min_time = min(min_time, time_elapsed);
            total_time += time_elapsed;
            min_cycles = min(min_cycles, cycles);
            total_cycles += cycles;
        }
        json_out << "{\"benchmark\": \"matmul\", \"N\": " << N
                 << ", \"kernel\": \"naive\", \"lang\": \"c\", \"threads\": 1, \"avg_time\": " << (total_time/NUM_RUNS) << ", \"min_time\": " << min_time 
                 << ", \"avg_cycles\": " << (total_cycles/NUM_RUNS) << ", \"min_cycles\": " << min_cycles << "}\n";
        g_arena.top = 0;

        // 2. TILED
        A = new_tensor(N, N); B = new_tensor(N, N);
        for (size_t i = 0; i < N * N; i++) { A->data[i] = 1.0f; B->data[i] = 2.0f; }
        
        total_time = 0; min_time = 1e9; total_cycles = 0; min_cycles = UINT64_MAX;
        Tensor* C_tiled_warmup = tensor_matmul_tiled(A, B);
        for (int r = 0; r < NUM_RUNS; r++) {
            uint64_t cycles = get_cpu_cycles();
            double start_time = get_wall_time();
            Tensor* C_tiled = tensor_matmul_tiled(A, B);
            double time_elapsed = (get_wall_time() - start_time);
            cycles = (get_cpu_cycles() - cycles);
            min_time = min(min_time, time_elapsed);
            total_time += time_elapsed;
            min_cycles = min(min_cycles, cycles);
            total_cycles += cycles;
        }
        json_out << "{\"benchmark\": \"matmul\", \"N\": " << N
                 << ", \"kernel\": \"tiled\", \"lang\": \"c\", \"threads\": 1, \"avg_time\": " << (total_time/NUM_RUNS) << ", \"min_time\": " << min_time 
                 << ", \"avg_cycles\": " << (total_cycles/NUM_RUNS) << ", \"min_cycles\": " << min_cycles << "}\n";
        g_arena.top = 0;

        // 3. SIMD
        A = new_tensor(N, N); B = new_tensor(N, N);
        for (size_t i = 0; i < N * N; i++) { A->data[i] = 1.0f; B->data[i] = 2.0f; }
        
        total_time = 0; min_time = 1e9; total_cycles = 0; min_cycles = UINT64_MAX;
        Tensor* C_simd_warmup = tensor_matmul_simd(A, B);
        for (int r = 0; r < NUM_RUNS; r++) {
            uint64_t cycles = get_cpu_cycles();
            double start_time = get_wall_time();
            Tensor* C_simd = tensor_matmul_simd(A, B);
            double time_elapsed = (get_wall_time() - start_time);
            cycles = (get_cpu_cycles() - cycles);
            min_time = min(min_time, time_elapsed);
            total_time += time_elapsed;
            min_cycles = min(min_cycles, cycles);
            total_cycles += cycles;
        }
        json_out << "{\"benchmark\": \"matmul\", \"N\": " << N
                 << ", \"kernel\": \"simd\", \"lang\": \"c\", \"threads\": 1, \"avg_time\": " << (total_time/NUM_RUNS) << ", \"min_time\": " << min_time 
                 << ", \"avg_cycles\": " << (total_cycles/NUM_RUNS) << ", \"min_cycles\": " << min_cycles << "}\n";
        g_arena.top = 0;

        if (step == 10 && N >= 100) step = 100;
        if (step == 100 && N >= 1000) step = 1000;
    }
}

int main(int argc, char* argv[])
{
    string mode = "performance-plugged";
    if (argc > 1) mode = argv[1];
    
    string filepath = "./data/c_single_" + mode + ".jsonl";
    ofstream json_out(filepath);
    
    run_single_thread_sweep(json_out);
    json_out.close();
    
    cout << "Finished! Results saved to " << filepath << endl;
    return 0;
}
