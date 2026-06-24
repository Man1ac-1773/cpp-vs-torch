#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <omp.h>
#include <immintrin.h>
#include <chrono>
#include "../common/perf_profiler.h"

void matmul_naive(int N, const float* A, const float* B, float* C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matmul_tiled(int N, const float* A, const float* B, float* C) {
    const int TILE_SIZE = 32;
    for (int i = 0; i < N; i += TILE_SIZE) {
        for (int j = 0; j < N; j += TILE_SIZE) {
            for (int k = 0; k < N; k += TILE_SIZE) {
                for (int ii = i; ii < std::min(i + TILE_SIZE, N); ++ii) {
                    for (int jj = j; jj < std::min(j + TILE_SIZE, N); ++jj) {
                        float sum = 0.0f;
                        for (int kk = k; kk < std::min(k + TILE_SIZE, N); ++kk) {
                            sum += A[ii * N + kk] * B[kk * N + jj];
                        }
                        C[ii * N + jj] += sum;
                    }
                }
            }
        }
    }
}

void matmul_simd(int N, const float* A, const float* B, float* C) {
    const int TILE_SIZE = 32;
    for (int i = 0; i < N; i += TILE_SIZE) {
        for (int j = 0; j < N; j += TILE_SIZE) {
            for (int k = 0; k < N; k += TILE_SIZE) {
                int i_end = std::min(i + TILE_SIZE, N);
                int j_end = std::min(j + TILE_SIZE, N);
                int k_end = std::min(k + TILE_SIZE, N);
                for (int ii = i; ii < i_end; ++ii) {
                    for (int jj = j; jj < j_end; jj += 8) {
                        __m256 c_vec = _mm256_loadu_ps(&C[ii * N + jj]);
                        for (int kk = k; kk < k_end; ++kk) {
                            __m256 a_vec = _mm256_broadcast_ss(&A[ii * N + kk]);
                            __m256 b_vec = _mm256_loadu_ps(&B[kk * N + jj]);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                        }
                        _mm256_storeu_ps(&C[ii * N + jj], c_vec);
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <N> <backend: naive|tiled|simd>\n";
        return 1;
    }

    int N = std::atoi(argv[1]);
    std::string backend = argv[2];

    int N_padded = N;
    if (backend == "simd" && N % 8 != 0) {
        N_padded = N + (8 - (N % 8));
    }

    std::vector<float> A(N_padded * N_padded, 1.0f);
    std::vector<float> B(N_padded * N_padded, 1.0f);
    std::vector<float> C(N_padded * N_padded, 0.0f);

    PerfCounter pc_branches, pc_misses;
    perf_init_branch_instructions(&pc_branches);
    perf_init_branch_misses(&pc_misses);

    auto start = std::chrono::high_resolution_clock::now();

    perf_start(&pc_branches);
    perf_start(&pc_misses);

    if (backend == "naive") {
        matmul_naive(N, A.data(), B.data(), C.data());
    } else if (backend == "tiled") {
        matmul_tiled(N, A.data(), B.data(), C.data());
    } else if (backend == "simd") {
        matmul_simd(N_padded, A.data(), B.data(), C.data());
    } else {
        std::cerr << "Unknown backend: " << backend << "\n";
        return 1;
    }

    long long branches = perf_read(&pc_branches);
    long long misses = perf_read(&pc_misses);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    float miss_rate = (branches > 0) ? ((float)misses / branches) * 100.0f : 0.0f;

    std::cout << "{\"N\": " << N << ", \"backend\": \"" << backend << "\", "
              << "\"branches\": " << branches << ", \"branch_misses\": " << misses << ", "
              << "\"miss_rate_percent\": " << miss_rate << ", \"time\": " << diff.count() << "}\n";

    volatile float sum = 0;
    for (int i=0; i<N*N; i+=1000) sum += C[i];
    
    return 0;
}
