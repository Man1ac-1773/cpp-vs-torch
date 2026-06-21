#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>

#include "./../macrograd/src/macrograd.h"
#include "profiler.h"

using namespace std;

void run_kernel_sweep(ofstream& json_out)
{
    uint step = 10;
    const int NUM_RUNS = 10;
    cout << " ========== Starting benchmarking ========= " << endl;
    for (size_t N = 10; N <= 2000; N += step)
    {
        double total_time = 0;
        double min_time = 1e9;
        uint64_t total_cycles = 0;
        uint64_t min_cycles = UINT64_MAX;
        Tensor* A = new_tensor(N, N);
        Tensor* B = new_tensor(N, N);
        // Fill with some random data...
        for (size_t i = 0; i < N * N; i++)
        {
            A->data[i] = 1.0f;
            B->data[i] = 2.0f;
        }
        Tensor* C_warmup = tensor_matmul_naive(A, B);
        // 1. Measure Naive
        for (int r = 0; r < NUM_RUNS; r++)
        {
            uint64_t cycles = get_cpu_cycles();
            double start_time = get_wall_time();

            Tensor* C_naive = tensor_matmul_naive(A, B);

            double time_elapsed = (get_wall_time() - start_time);
            cycles = (get_cpu_cycles() - cycles);
            min_time = min(min_time, time_elapsed);
            min_cycles = min(min_cycles, cycles);
            total_time += time_elapsed;
            total_cycles += cycles;
        }

        uint64_t avg_cycles = total_cycles / NUM_RUNS;
        double avg_time = total_time / NUM_RUNS;
        json_out << "{\"benchmark\": \"matmul\", \"N\": " << N
                 << ", \"kernel\": \"naive\", \"avg_cycles\": " << avg_cycles << ", \"min_cycles\": " << min_cycles
                 << ", \"avg_time\": " << avg_time << ", \"min_time\": " << min_time << "}\n";
        g_arena.top = 0;

        // reinitilization
        A = new_tensor(N, N);
        B = new_tensor(N, N);
        // Fill with some random data...
        for (size_t i = 0; i < N * N; i++)
        {
            A->data[i] = 1.0f;
            B->data[i] = 2.0f;
        }
        // TILED BENCHMARK
        min_time = 1e9;
        min_cycles = UINT64_MAX;
        total_time = 0;
        total_cycles = 0;
        Tensor* C_tiled_warmup = tensor_matmul_tiled(A, B);
        for (int r = 0; r < NUM_RUNS; r++)
        {
            uint64_t cycles = get_cpu_cycles();
            double start_time = get_wall_time();

            Tensor* C_tiled = tensor_matmul_tiled(A, B);

            double time_elapsed = (get_wall_time() - start_time);
            cycles = (get_cpu_cycles() - cycles);
            min_time = min(min_time, time_elapsed);
            min_cycles = min(min_cycles, cycles);
            total_time += time_elapsed;
            total_cycles += cycles;
        }

        avg_cycles = total_cycles / NUM_RUNS;
        avg_time = total_time / NUM_RUNS;
        json_out << "{\"benchmark\": \"matmul\", \"N\": " << N
                 << ", \"kernel\": \"tiled\", \"avg_cycles\": " << avg_cycles << ", \"min_cycles\": " << min_cycles
                 << ", \"avg_time\": " << avg_time << ", \"min_time\": " << min_time << "}\n";
        g_arena.top = 0;

        // reinitilization
        A = new_tensor(N, N);
        B = new_tensor(N, N);
        // Fill with some random data...
        for (size_t i = 0; i < N * N; i++)
        {
            A->data[i] = 1.0f;
            B->data[i] = 2.0f;
        }
        // SIMD BENCHMARK
        min_time = 1e9;
        min_cycles = UINT64_MAX;
        total_time = 0;
        total_cycles = 0;
        Tensor* C_simd_warmup = tensor_matmul_simd(A, B);
        for (int r = 0; r < NUM_RUNS; r++)
        {
            uint64_t cycles = get_cpu_cycles();
            double start_time = get_wall_time();

            Tensor* C_simd = tensor_matmul_simd(A, B);

            double time_elapsed = (get_wall_time() - start_time);
            cycles = (get_cpu_cycles() - cycles);
            min_time = min(min_time, time_elapsed);
            min_cycles = min(min_cycles, cycles);
            total_time += time_elapsed;
            total_cycles += cycles;
        }

        avg_cycles = total_cycles / NUM_RUNS;
        avg_time = total_time / NUM_RUNS;
        json_out << "{\"benchmark\": \"matmul\", \"N\": " << N
                 << ", \"kernel\": \"simd\", \"avg_cycles\": " << avg_cycles << ", \"min_cycles\": " << min_cycles
                 << ", \"avg_time\": " << avg_time << ", \"min_time\": " << min_time << "}\n";
        g_arena.top = 0;
        if (step == 10 && N >= 100)
        {
            step = 100;
            cout << "N stepped up to 100" << endl;
        }
        if (step == 100 && N >= 1000)
        {
            step = 1000;
            cout << "N stepped up to 1000" << endl;
        }
    }
}

int main()
{
    double start_time = get_wall_time();
    ofstream json_out("./c-data/battery-saver_benchmark_results.jsonl");
    run_kernel_sweep(json_out);
    json_out.close();
    cout << "Total time elapsed : " << get_wall_time() - start_time << endl;
    return 0;
}
