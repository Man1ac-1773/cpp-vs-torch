#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <omp.h>

#include "./../../minigrad/src/tensor.h"
#include "../common/profiler.h"

using namespace std;

// skipped multi-threaded naive implementation for c++
::Tensor matmul_naive_omp(const ::Tensor& a, const ::Tensor& b) {
    ::Tensor out(a.node->rows, b.node->cols);
    for(size_t i=0; i<out.node->rows*out.node->cols; i++) out.node->data[i] = 0.0f;
    
    #pragma omp parallel for
    for (size_t i = 0; i < a.node->rows; i++) {
        for (size_t j = 0; j < b.node->cols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < a.node->cols; k++) {
                sum += a.node->data[i * a.node->cols + k] * b.node->data[k * b.node->cols + j];
            }
            out.node->data[i * out.node->cols + j] = sum;
        }
    }
    return out;
}

// skipped multi-threaded tiled implementation for c++
::Tensor matmul_tiled_omp(const ::Tensor& a, const ::Tensor& b) {
    ::Tensor out(a.node->rows, b.node->cols);
    for(size_t i=0; i<out.node->rows*out.node->cols; i++) out.node->data[i] = 0.0f;
    
    size_t TILE_SIZE = 32;
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < a.node->rows; i += TILE_SIZE) {
        for (size_t j = 0; j < b.node->cols; j += TILE_SIZE) {
            for (size_t k = 0; k < a.node->cols; k += TILE_SIZE) {
                for (size_t ii = i; ii < min(i + TILE_SIZE, (size_t)a.node->rows); ii++) {
                    for (size_t jj = j; jj < min(j + TILE_SIZE, (size_t)b.node->cols); jj++) {
                        float sum = 0.0f;
                        for (size_t kk = k; kk < min(k + TILE_SIZE, (size_t)a.node->cols); kk++) {
                            sum += a.node->data[ii * a.node->cols + kk] * b.node->data[kk * b.node->cols + jj];
                        }
                        out.node->data[ii * out.node->cols + jj] += sum;
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
    cout << " ========== Starting Multi-threaded C++ Benchmarking (" << max_threads << " threads) ========= " << endl;
    
    for (size_t N = 10; N <= 2000; N += step)
    {
        double total_time = 0;
        double min_time = 1e9;
        uint64_t total_cycles = 0;
        uint64_t min_cycles = UINT64_MAX;
        
        ::Tensor A(N, N);
        ::Tensor B(N, N);
        for (size_t i = 0; i < N * N; i++) { A(i/N, i%N) = 1.0f; B(i/N, i%N) = 2.0f; }
        
        // 1. naive omp
        ::Tensor C_warmup = matmul_naive_omp(A, B);
        for (int r = 0; r < NUM_RUNS; r++) {
            uint64_t cycles = get_cpu_cycles();
            double start_time = get_wall_time();
            ::Tensor C_naive = matmul_naive_omp(A, B);
            double time_elapsed = (get_wall_time() - start_time);
            cycles = (get_cpu_cycles() - cycles);
            min_time = min(min_time, time_elapsed);
            total_time += time_elapsed;
            min_cycles = min(min_cycles, cycles);
            total_cycles += cycles;
        }
        json_out << "{\"benchmark\": \"matmul\", \"N\": " << N
                 << ", \"kernel\": \"naive\", \"lang\": \"cpp\", \"threads\": " << max_threads << ", \"avg_time\": " << (total_time/NUM_RUNS) << ", \"min_time\": " << min_time 
                 << ", \"avg_cycles\": " << (total_cycles/NUM_RUNS) << ", \"min_cycles\": " << min_cycles << "}\n";

        // 2. tiled omp
        total_time = 0; min_time = 1e9; total_cycles = 0; min_cycles = UINT64_MAX;
        ::Tensor C_tiled_warmup = matmul_tiled_omp(A, B);
        for (int r = 0; r < NUM_RUNS; r++) {
            uint64_t cycles = get_cpu_cycles();
            double start_time = get_wall_time();
            ::Tensor C_tiled = matmul_tiled_omp(A, B);
            double time_elapsed = (get_wall_time() - start_time);
            cycles = (get_cpu_cycles() - cycles);
            min_time = min(min_time, time_elapsed);
            total_time += time_elapsed;
            min_cycles = min(min_cycles, cycles);
            total_cycles += cycles;
        }
        json_out << "{\"benchmark\": \"matmul\", \"N\": " << N
                 << ", \"kernel\": \"tiled\", \"lang\": \"cpp\", \"threads\": " << max_threads << ", \"avg_time\": " << (total_time/NUM_RUNS) << ", \"min_time\": " << min_time 
                 << ", \"avg_cycles\": " << (total_cycles/NUM_RUNS) << ", \"min_cycles\": " << min_cycles << "}\n";

        // 3. simd mt (minigrads native pthread implementation)
        total_time = 0; min_time = 1e9; total_cycles = 0; min_cycles = UINT64_MAX;
        ::Tensor C_simd_warmup = matmul_simd_mt(A, B);
        for (int r = 0; r < NUM_RUNS; r++) {
            uint64_t cycles = get_cpu_cycles();
            double start_time = get_wall_time();
            ::Tensor C_simd = matmul_simd_mt(A, B);
            double time_elapsed = (get_wall_time() - start_time);
            cycles = (get_cpu_cycles() - cycles);
            min_time = min(min_time, time_elapsed);
            total_time += time_elapsed;
            min_cycles = min(min_cycles, cycles);
            total_cycles += cycles;
        }
        json_out << "{\"benchmark\": \"matmul\", \"N\": " << N
                 << ", \"kernel\": \"simd\", \"lang\": \"cpp\", \"threads\": " << max_threads << ", \"avg_time\": " << (total_time/NUM_RUNS) << ", \"min_time\": " << min_time 
                 << ", \"avg_cycles\": " << (total_cycles/NUM_RUNS) << ", \"min_cycles\": " << min_cycles << "}\n";

        if (step == 10 && N >= 100) step = 100;
        if (step == 100 && N >= 1000) step = 1000;
    }
}

int main(int argc, char* argv[])
{
    string mode = "performance-plugged";
    if (argc > 1) mode = argv[1];
    
    string filepath = "./data/cpp_multi_" + mode + ".jsonl";
    ofstream json_out(filepath);
    
    run_multi_thread_sweep(json_out);
    json_out.close();
    
    cout << "Finished! Results saved to " << filepath << endl;
    return 0;
}
