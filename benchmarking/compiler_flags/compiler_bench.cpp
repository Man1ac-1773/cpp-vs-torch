#include <iostream>
#include <vector>
#include <chrono>

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

int main() {
    int N = 1000;

    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 1.0f);
    std::vector<float> C(N * N, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();

    matmul_naive(N, A.data(), B.data(), C.data());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << diff.count() << std::endl;

    volatile float sum = 0;
    for (int i=0; i<N*N; i+=1000) sum += C[i];
    
    return 0;
}
