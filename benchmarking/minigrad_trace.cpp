#include <iostream>
#include "./../minigrad/src/tensor.h"

using namespace std;

int main() {
    cout << "Tracing minigrad N=1000..." << endl;
    Tensor A(1000, 1000);
    Tensor B(1000, 1000);
    
    // Warmup memory pages
    for(int i=0; i<1000*1000; i++) {
        A.node->data[i] = 1.0f;
        B.node->data[i] = 2.0f;
    }

    Tensor C_naive = A * B;
    Tensor C_tiled = matmul_tiled(A, B);
    Tensor C_simd = matmul_simd(A, B);
    
    PROFILER_DUMP("trace_minigrad.json");
    cout << "Dumped to trace_minigrad.json" << endl;
    return 0;
}
