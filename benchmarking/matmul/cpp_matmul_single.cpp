#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>

#include "../common/profiler.h"
#include "./../../minigrad/src/tensor.h"

using namespace std;

void run_single_thread_sweep(ofstream& json_out)
{
    uint step = 10;
    const int NUM_RUNS = 10;
    cout << " ========== Starting Single-threaded C++ Benchmarking ========= " << endl;
    for (size_t N = 10; N <= 2000; N += step)
    {
        double total_time = 0;
        double min_time = 1e9;
        uint64_t total_cycles = 0;
        uint64_t min_cycles = UINT64_MAX;

        ::Tensor A(N, N);
        ::Tensor B(N, N);
        for (size_t i = 0; i < N * N; i++)
        {
            A(i / N, i % N) = 1.0f;
            B(i / N, i % N) = 2.0f;
        }

        // 1. naive
        ::Tensor C_warmup = A * B;
        for (int r = 0; r < NUM_RUNS; r++)
        {
            uint64_t cycles = get_cpu_cycles();
            double start_time = get_wall_time();
            ::Tensor C_naive = A * B;
            double time_elapsed = (get_wall_time() - start_time);
            cycles = (get_cpu_cycles() - cycles);
            min_time = min(min_time, time_elapsed);
            total_time += time_elapsed;
            min_cycles = min(min_cycles, cycles);
            total_cycles += cycles;
        }
        json_out << "{\"benchmark\": \"matmul\", \"N\": " << N
                 << ", \"kernel\": \"naive\", \"lang\": \"cpp\", \"threads\": 1, \"avg_time\": "
                 << (total_time / NUM_RUNS) << ", \"min_time\": " << min_time
                 << ", \"avg_cycles\": " << (total_cycles / NUM_RUNS) << ", \"min_cycles\": " << min_cycles << "}\n";

        // 2. tiled
        total_time = 0;
        min_time = 1e9;
        total_cycles = 0;
        min_cycles = UINT64_MAX;
        ::Tensor C_tiled_warmup = matmul_tiled(A, B);
        for (int r = 0; r < NUM_RUNS; r++)
        {
            uint64_t cycles = get_cpu_cycles();
            double start_time = get_wall_time();
            ::Tensor C_tiled = matmul_tiled(A, B);
            double time_elapsed = (get_wall_time() - start_time);
            cycles = (get_cpu_cycles() - cycles);
            min_time = min(min_time, time_elapsed);
            total_time += time_elapsed;
            min_cycles = min(min_cycles, cycles);
            total_cycles += cycles;
        }
        json_out << "{\"benchmark\": \"matmul\", \"N\": " << N
                 << ", \"kernel\": \"tiled\", \"lang\": \"cpp\", \"threads\": 1, \"avg_time\": "
                 << (total_time / NUM_RUNS) << ", \"min_time\": " << min_time
                 << ", \"avg_cycles\": " << (total_cycles / NUM_RUNS) << ", \"min_cycles\": " << min_cycles << "}\n";

        // 3. simd
        total_time = 0;
        min_time = 1e9;
        total_cycles = 0;
        min_cycles = UINT64_MAX;
        ::Tensor C_simd_warmup = matmul_simd(A, B);
        for (int r = 0; r < NUM_RUNS; r++)
        {
            uint64_t cycles = get_cpu_cycles();
            double start_time = get_wall_time();
            ::Tensor C_simd = matmul_simd(A, B);
            double time_elapsed = (get_wall_time() - start_time);
            cycles = (get_cpu_cycles() - cycles);
            min_time = min(min_time, time_elapsed);
            total_time += time_elapsed;
            min_cycles = min(min_cycles, cycles);
            total_cycles += cycles;
        }
        json_out << "{\"benchmark\": \"matmul\", \"N\": " << N
                 << ", \"kernel\": \"simd\", \"lang\": \"cpp\", \"threads\": 1, \"avg_time\": "
                 << (total_time / NUM_RUNS) << ", \"min_time\": " << min_time
                 << ", \"avg_cycles\": " << (total_cycles / NUM_RUNS) << ", \"min_cycles\": " << min_cycles << "}\n";

        if (step == 10 && N >= 100)
            step = 100;
        if (step == 100 && N >= 1000)
            step = 1000;
    }
}

int main(int argc, char* argv[])
{
    string mode = "performance-plugged";
    if (argc > 1)
        mode = argv[1];

    string filepath = "./data/cpp_single_" + mode + ".jsonl";
    ofstream json_out(filepath);

    run_single_thread_sweep(json_out);
    json_out.close();

    cout << "Finished! Results saved to " << filepath << endl;
    return 0;
}
