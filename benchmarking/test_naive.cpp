#include "./../macrograd/src/macrograd.h"
int main() {
    Tensor* A = new_tensor(1000, 1000);
    Tensor* B = new_tensor(1000, 1000);
    tensor_matmul_naive(A, B);
    return 0;
}
