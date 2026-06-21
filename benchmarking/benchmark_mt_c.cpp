#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>

#include "./../macrograd/src/macrograd.h"
#include "profiler.h"

using namespace std;

void run_mt_sweep_c(ofstream& json_out)
{
    uint step = 10;
    const int NUM_RUNS = 10;
    cout << " ========== Starting C MT Benchmarking ========= " << endl;
    for (size_t N = 10; N <= 2000; N += step)
    {
        double total_time = 0;
        double min_time = 1e9;
        
        Tensor* c_A = new_tensor(N, N);
        Tensor* c_B = new_tensor(N, N);
        for (size_t i = 0; i < N * N; i++) {
            c_A->data[i] = 1.0f; c_B->data[i] = 2.0f;
        }
        
        Tensor* c_warmup = tensor_matmul_simd_mt(c_A, c_B);
        
        for (int r = 0; r < NUM_RUNS; r++) {
            double start_time = get_wall_time();
            Tensor* c_mt = tensor_matmul_simd_mt(c_A, c_B);
            double time_elapsed = (get_wall_time() - start_time);
            min_time = min(min_time, time_elapsed);
            total_time += time_elapsed;
        }

        double avg_time = total_time / NUM_RUNS;
        json_out << "{\"benchmark\": \"matmul_mt\", \"N\": " << N
                 << ", \"kernel\": \"simd_mt\", \"lang\": \"c\", \"avg_time\": " << avg_time << ", \"min_time\": " << min_time << "}\n";
        g_arena.top = 0; 

        if (step == 10 && N >= 100) step = 100;
        if (step == 100 && N >= 1000) step = 1000;
    }
}

int main()
{
    ofstream json_out("./c-data/mt_benchmark_results.jsonl");
    run_mt_sweep_c(json_out);
    json_out.close();
    return 0;
}
