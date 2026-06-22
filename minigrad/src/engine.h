#ifndef MINIGRAD_ENGINE_H
#define MINIGRAD_ENGINE_H

#include "tensor.h"
#include <vector>

class Linear {
public:
    Tensor weight;
    Linear(int in_features, int out_features) : weight(in_features, out_features) {}
    Tensor operator()(const Tensor& x) {
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
};

#endif
