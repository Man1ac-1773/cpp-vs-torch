#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <omp.h>

#include "./../../minigrad/src/tensor.h"
#include "../common/profiler.h"

using namespace std;

// Missing Multi-threaded Naive Implementation for C++
::Tensor matmul_naive_omp(const ::Tensor& a, const ::Tensor& b) {
    ::Tensor out(a.rows, b.cols);
    for(size_t i=0; i<out.rows*out.cols; i++) out.node->data[i] = 0.0f;
    
    #pragma omp parallel for
    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < b.cols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < a.cols; k++) {
                sum += a.node->data[i * a.cols + k] * b.node->data[k * b.cols + j];
            }
            out.node->data[i * out.cols + j] = sum;
        }
    }
    return out;
}

// Missing Multi-threaded Tiled Implementation for C++
::Tensor matmul_tiled_omp(const ::Tensor& a, const ::Tensor& b) {
    ::Tensor out(a.rows, b.cols);
    for(size_t i=0; i<out.rows*out.cols; i++) out.node->data[i] = 0.0f;
    
    size_t TILE_SIZE = 32;
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < a.rows; i += TILE_SIZE) {
        for (size_t j = 0; j < b.cols; j += TILE_SIZE) {
            for (size_t k = 0; k < a.cols; k += TILE_SIZE) {
                for (size_t ii = i; ii < min(i + TILE_SIZE, (size_t)a.rows); ii++) {
                    for (size_t jj = j; jj < min(j + TILE_SIZE, (size_t)b.cols); jj++) {
                        float sum = 0.0f;
                        for (size_t kk = k; kk < min(k + TILE_SIZE, (size_t)a.cols); kk++) {
                            sum += a.node->data[ii * a.cols + kk] * b.node->data[kk * b.cols + jj];
                        }
                        out.node->data[ii * out.cols + jj] += sum;
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
        
        ::Tensor A(N, N);
        ::Tensor B(N, N);
        for (size_t i = 0; i < N * N; i++) { A(i/N, i%N) = 1.0f; B(i/N, i%N) = 2.0f; }
        
        // 1. NAIVE OMP
        ::Tensor C_warmup = matmul_naive_omp(A, B);
        for (int r = 0; r < NUM_RUNS; r++) {
            double start_time = get_wall_time();
            ::Tensor C_naive = matmul_naive_omp(A, B);
            double time_elapsed = (get_wall_time() - start_time);
            min_time = min(min_time, time_elapsed);
            total_time += time_elapsed;
        }
        json_out << "{\"benchmark\": \"matmul\", \"N\": " << N
                 << ", \"kernel\": \"naive\", \"lang\": \"cpp\", \"threads\": " << max_threads << ", \"avg_time\": " << (total_time/NUM_RUNS) << ", \"min_time\": " << min_time << "}\n";

        // 2. TILED OMP
        total_time = 0; min_time = 1e9;
        ::Tensor C_tiled_warmup = matmul_tiled_omp(A, B);
        for (int r = 0; r < NUM_RUNS; r++) {
            double start_time = get_wall_time();
            ::Tensor C_tiled = matmul_tiled_omp(A, B);
            double time_elapsed = (get_wall_time() - start_time);
            min_time = min(min_time, time_elapsed);
            total_time += time_elapsed;
        }
        json_out << "{\"benchmark\": \"matmul\", \"N\": " << N
                 << ", \"kernel\": \"tiled\", \"lang\": \"cpp\", \"threads\": " << max_threads << ", \"avg_time\": " << (total_time/NUM_RUNS) << ", \"min_time\": " << min_time << "}\n";

        // 3. SIMD MT (Minigrad's native pthread implementation)
        total_time = 0; min_time = 1e9;
        ::Tensor C_simd_warmup = matmul_simd_mt(A, B);
        for (int r = 0; r < NUM_RUNS; r++) {
            double start_time = get_wall_time();
            ::Tensor C_simd = matmul_simd_mt(A, B);
            double time_elapsed = (get_wall_time() - start_time);
            min_time = min(min_time, time_elapsed);
            total_time += time_elapsed;
        }
        json_out << "{\"benchmark\": \"matmul\", \"N\": " << N
                 << ", \"kernel\": \"simd\", \"lang\": \"cpp\", \"threads\": " << max_threads << ", \"avg_time\": " << (total_time/NUM_RUNS) << ", \"min_time\": " << min_time << "}\n";

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
