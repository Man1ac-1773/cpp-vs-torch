#ifndef MACROGRAD_ENGINE_H
#define MACROGRAD_ENGINE_H

#include "macrograd.h"

typedef struct {
    Tensor* weight;
} Linear;

static inline Linear linear_init(int in_features, int out_features) {
    Linear l;
    l.weight = new_tensor(in_features, out_features);
    return l;
}

static inline Tensor* linear_forward(Linear* l, Tensor* x) {
    return tensor_matmul_simd_mt(x, l->weight);
}

typedef struct {
    Linear fc1;
    Linear fc2;
} MLP;

static inline MLP mlp_init(int input_dim, int hidden_dim, int output_dim) {
    MLP m;
    m.fc1 = linear_init(input_dim, hidden_dim);
    m.fc2 = linear_init(hidden_dim, output_dim);
    return m;
}

static inline Tensor* mlp_forward(MLP* m, Tensor* x) {
    Tensor* h = linear_forward(&m->fc1, x);
    Tensor* a = tensor_relu(h);
    return linear_forward(&m->fc2, a);
}

typedef struct {
    Tensor** params;
    int num_params;
    float lr;
} SGD;

static inline SGD sgd_init(Tensor** params, int num_params, float lr) {
    SGD opt = {params, num_params, lr};
    return opt;
}

static inline void sgd_step(SGD* opt) {
    for (int p = 0; p < opt->num_params; p++) {
        Tensor* param = opt->params[p];
        for (uint i = 0; i < param->shape[0] * param->shape[1]; i++) {
            param->data[i] -= opt->lr * param->grad[i];
        }
    }
}

#endif
