#ifndef MINIGRAD_ENGINE_H
#define MINIGRAD_ENGINE_H

#include "tensor.h"
#include <vector>

enum MatmulBackend {
    MATMUL_NAIVE,
    MATMUL_TILED,
    MATMUL_SIMD
};

static MatmulBackend g_matmul_backend = MATMUL_SIMD;

class Linear {
public:
    Tensor weight;
    Linear(int in_features, int out_features) : weight(in_features, out_features) {}
    Tensor operator()(const Tensor& x) {
        if (g_matmul_backend == MATMUL_NAIVE) return x * weight;
        if (g_matmul_backend == MATMUL_TILED) return matmul_tiled(x, weight);
        return matmul_simd_mt(x, weight);
    }
};

class MLP {
public:
    Linear fc1;
    Linear fc2;
    MLP(int input_dim, int hidden_dim, int output_dim) 
        : fc1(input_dim, hidden_dim), fc2(hidden_dim, output_dim) {}
    
    Tensor forward(const Tensor& x) {
        Tensor h = fc1(x);
        Tensor a = relu(h);
        return fc2(a);
    }
    
    std::vector<Tensor*> parameters() {
        return {&fc1.weight, &fc2.weight};
    }
};

class MLP3 {
public:
    Linear fc1;
    Linear fc2;
    Linear fc3;
    MLP3(int input_dim, int hidden1_dim, int hidden2_dim, int output_dim) 
        : fc1(input_dim, hidden1_dim), fc2(hidden1_dim, hidden2_dim), fc3(hidden2_dim, output_dim) {}
    
    Tensor forward(const Tensor& x) {
        Tensor h1 = fc1(x);
        Tensor a1 = relu(h1);
        Tensor h2 = fc2(a1);
        Tensor a2 = relu(h2);
        return fc3(a2);
    }
    
    std::vector<Tensor*> parameters() {
        return {&fc1.weight, &fc2.weight, &fc3.weight};
    }
};

class SGD {
    float lr;
    std::vector<Tensor*> params;
public:
    SGD(std::vector<Tensor*> parameters, float learning_rate) 
        : params(parameters), lr(learning_rate) {}
    
    void step() {
        for (auto p : params) {
            for (size_t i = 0; i < p->node->data.size(); i++) {
                p->node->data[i] -= lr * p->node->grad[i];
            }
        }
    }

    void zero_grad() {
        for (auto p : params) {
            std::fill(p->node->grad.begin(), p->node->grad.end(), 0.0f);
        }
    }
};

#endif
