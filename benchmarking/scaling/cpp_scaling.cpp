#include <chrono>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

// sort ra
void matmul_naive(int N, const float* A, const float* B, float* C)
{
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matmul_tiled(int N, const float* A, const float* B, float* C)
{
    const int TILE_SIZE = 32;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += TILE_SIZE)
    {
        for (int j = 0; j < N; j += TILE_SIZE)
        {
            for (int k = 0; k < N; k += TILE_SIZE)
            {
                for (int ii = i; ii < std::min(i + TILE_SIZE, N); ++ii)
                {
                    for (int jj = j; jj < std::min(j + TILE_SIZE, N); ++jj)
                    {
                        float sum = 0.0f;
                        for (int kk = k; kk < std::min(k + TILE_SIZE, N); ++kk)
                        {
                            sum += A[ii * N + kk] * B[kk * N + jj];
                        }
                        C[ii * N + jj] += sum;
                    }
                }
            }
        }
    }
}

void matmul_simd(int N, const float* A, const float* B, float* C)
{
    const int TILE_SIZE = 32;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += TILE_SIZE)
    {
        for (int j = 0; j < N; j += TILE_SIZE)
        {
            for (int k = 0; k < N; k += TILE_SIZE)
            {
                int i_end = std::min(i + TILE_SIZE, N);
                int j_end = std::min(j + TILE_SIZE, N);
                int k_end = std::min(k + TILE_SIZE, N);
                for (int ii = i; ii < i_end; ++ii)
                {
                    for (int jj = j; jj < j_end; jj += 8)
                    {
                        __m256 c_vec = _mm256_loadu_ps(&C[ii * N + jj]);
                        for (int kk = k; kk < k_end; ++kk)
                        {
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

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <N> <backend: naive|tiled|simd>\n";
        return 1;
    }

    int N = std::atoi(argv[1]);
    std::string backend = argv[2];

    int N_padded = N;
    if (backend == "simd" && N % 8 != 0)
    {
        N_padded = N + (8 - (N % 8));
    }

    std::vector<float> A(N_padded * N_padded, 1.0f);
    std::vector<float> B(N_padded * N_padded, 1.0f);
    std::vector<float> C(N_padded * N_padded, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();

    if (backend == "naive")
    {
        matmul_naive(N, A.data(), B.data(), C.data());
    }
    else if (backend == "tiled")
    {
        matmul_tiled(N, A.data(), B.data(), C.data());
    }
    else if (backend == "simd")
    {
        matmul_simd(N_padded, A.data(), B.data(), C.data());
    }
    else
    {
        std::cerr << "Unknown backend: " << backend << "\n";
        return 1;
    }
    // i wonder if anyone reads the comments but me lmao
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << diff.count() << std::endl;

    volatile float sum = 0;
    for (int i = 0; i < N * N; i += 1000)
        sum += C[i];

    return 0;
}
