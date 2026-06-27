#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <omp.h>

#include "../common/profiler.h"
#include "./../../macrograd/src/macrograd.h"

using namespace std;

// skipped multi-threaded naive implementation
Tensor* tensor_matmul_naive_omp(Tensor* a, Tensor* b)
{
    Tensor* out = new_tensor(a->shape[0], b->shape[1]);
    for (size_t i = 0; i < out->shape[0] * out->shape[1]; i++)
        out->data[i] = 0.0f;

#pragma omp parallel for
    for (size_t i = 0; i < a->shape[0]; i++)
    {
        for (size_t j = 0; j < b->shape[1]; j++)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < a->shape[1]; k++)
            {
                sum += a->data[i * a->shape[1] + k] * b->data[k * b->shape[1] + j];
            }
            out->data[i * out->shape[1] + j] = sum;
        }
    }
    return out;
}

// skipped multi-threaded tiled implementation
Tensor* tensor_matmul_tiled_omp(Tensor* a, Tensor* b)
{
    Tensor* out = new_tensor(a->shape[0], b->shape[1]);
    for (size_t i = 0; i < out->shape[0] * out->shape[1]; i++)
        out->data[i] = 0.0f;

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < a->shape[0]; i += TILE_SIZE)
    {
        for (size_t j = 0; j < b->shape[1]; j += TILE_SIZE)
        {
            for (size_t k = 0; k < a->shape[1]; k += TILE_SIZE)
            {
                for (size_t ii = i; ii < min(i + TILE_SIZE, (size_t) a->shape[0]); ii++)
                {
                    for (size_t jj = j; jj < min(j + TILE_SIZE, (size_t) b->shape[1]); jj++)
                    {
                        float sum = 0.0f;
                        for (size_t kk = k; kk < min(k + TILE_SIZE, (size_t) a->shape[1]); kk++)
                        {
                            sum += a->data[ii * a->shape[1] + kk] * b->data[kk * b->shape[1] + jj];
                        }
                        out->data[ii * out->shape[1] + jj] += sum;
                    }
                }
            }
        }
    }
    return out;
}

void run_multi_thread_sweep(ofstream& json_out)
{
    uint step = 10;
    const int NUM_RUNS = 10;
    int max_threads = omp_get_max_threads();
    cout << " ========== Starting Multi-threaded C Benchmarking (" << max_threads << " threads) ========= " << endl;

    for (size_t N = 10; N <= 2000; N += step)
    {
        double total_time = 0;
        double min_time = 1e9;
        uint64_t total_cycles = 0;
        uint64_t min_cycles = UINT64_MAX;

        Tensor* A = new_tensor(N, N);
        Tensor* B = new_tensor(N, N);
        for (size_t i = 0; i < N * N; i++)
        {
            A->data[i] = 1.0f;
            B->data[i] = 2.0f;
        }

        // 1. naive omp
        Tensor* C_warmup = tensor_matmul_naive_omp(A, B);
        for (int r = 0; r < NUM_RUNS; r++)
        {
            uint64_t cycles = get_cpu_cycles();
            double start_time = get_wall_time();
            Tensor* C_naive = tensor_matmul_naive_omp(A, B);
            double time_elapsed = (get_wall_time() - start_time);
            cycles = (get_cpu_cycles() - cycles);
            min_time = min(min_time, time_elapsed);
            total_time += time_elapsed;
            min_cycles = min(min_cycles, cycles);
            total_cycles += cycles;
        }
        json_out << "{\"benchmark\": \"matmul\", \"N\": " << N
                 << ", \"kernel\": \"naive\", \"lang\": \"c\", \"threads\": " << max_threads
                 << ", \"avg_time\": " << (total_time / NUM_RUNS) << ", \"min_time\": " << min_time
                 << ", \"avg_cycles\": " << (total_cycles / NUM_RUNS) << ", \"min_cycles\": " << min_cycles << "}\n";
        g_arena.top = 0;

        // 2. tiled omp
        A = new_tensor(N, N);
        B = new_tensor(N, N);
        for (size_t i = 0; i < N * N; i++)
        {
            A->data[i] = 1.0f;
            B->data[i] = 2.0f;
        }

        total_time = 0;
        min_time = 1e9;
        total_cycles = 0;
        min_cycles = UINT64_MAX;
        Tensor* C_tiled_warmup = tensor_matmul_tiled_omp(A, B);
        for (int r = 0; r < NUM_RUNS; r++)
        {
            uint64_t cycles = get_cpu_cycles();
            double start_time = get_wall_time();
            Tensor* C_tiled = tensor_matmul_tiled_omp(A, B);
            double time_elapsed = (get_wall_time() - start_time);
            cycles = (get_cpu_cycles() - cycles);
            min_time = min(min_time, time_elapsed);
            total_time += time_elapsed;
            min_cycles = min(min_cycles, cycles);
            total_cycles += cycles;
        }
        json_out << "{\"benchmark\": \"matmul\", \"N\": " << N
                 << ", \"kernel\": \"tiled\", \"lang\": \"c\", \"threads\": " << max_threads
                 << ", \"avg_time\": " << (total_time / NUM_RUNS) << ", \"min_time\": " << min_time
                 << ", \"avg_cycles\": " << (total_cycles / NUM_RUNS) << ", \"min_cycles\": " << min_cycles << "}\n";
        g_arena.top = 0;

        // 3. simd mt (macrograds native pthread implementation)
        A = new_tensor(N, N);
        B = new_tensor(N, N);
        for (size_t i = 0; i < N * N; i++)
        {
            A->data[i] = 1.0f;
            B->data[i] = 2.0f;
        }

        total_time = 0;
        min_time = 1e9;
        total_cycles = 0;
        min_cycles = UINT64_MAX;
        Tensor* C_simd_warmup = tensor_matmul_simd_mt(A, B);
        for (int r = 0; r < NUM_RUNS; r++)
        {
            uint64_t cycles = get_cpu_cycles();
            double start_time = get_wall_time();
            Tensor* C_simd = tensor_matmul_simd_mt(A, B);
            double time_elapsed = (get_wall_time() - start_time);
            cycles = (get_cpu_cycles() - cycles);
            min_time = min(min_time, time_elapsed);
            total_time += time_elapsed;
            min_cycles = min(min_cycles, cycles);
            total_cycles += cycles;
        }
        // macrograds simd mt uses thread_count (16), but i can report max_threads for consistency or read from
        // thread_count.
        json_out << "{\"benchmark\": \"matmul\", \"N\": " << N
                 << ", \"kernel\": \"simd\", \"lang\": \"c\", \"threads\": " << max_threads
                 << ", \"avg_time\": " << (total_time / NUM_RUNS) << ", \"min_time\": " << min_time
                 << ", \"avg_cycles\": " << (total_cycles / NUM_RUNS) << ", \"min_cycles\": " << min_cycles << "}\n";
        g_arena.top = 0;

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

    string filepath = "./data/c_multi_" + mode + ".jsonl";
    ofstream json_out(filepath);

    run_multi_thread_sweep(json_out);
    json_out.close();

    cout << "Finished! Results saved to " << filepath << endl;
    return 0;
}
