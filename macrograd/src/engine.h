#ifndef MACROGRAD_ENGINE_H
#define MACROGRAD_ENGINE_H

#include "macrograd.h"

typedef enum {
    MATMUL_NAIVE,
    MATMUL_TILED,
    MATMUL_SIMD
} MatmulBackend;

static MatmulBackend g_matmul_backend = MATMUL_SIMD;

typedef struct {
    Tensor* weight;
} Linear;

static inline Linear linear_init(int in_features, int out_features) {
    Linear l;
    l.weight = new_tensor(in_features, out_features);
    return l;
}

static inline Tensor* linear_forward(Linear* l, Tensor* x) {
    if (g_matmul_backend == MATMUL_NAIVE) return tensor_matmul_naive(x, l->weight);
    if (g_matmul_backend == MATMUL_TILED) return tensor_matmul_tiled(x, l->weight);
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
    Linear fc1;
    Linear fc2;
    Linear fc3;
} MLP3;

static inline MLP3 mlp3_init(int input_dim, int hidden1_dim, int hidden2_dim, int output_dim) {
    MLP3 m;
    m.fc1 = linear_init(input_dim, hidden1_dim);
    m.fc2 = linear_init(hidden1_dim, hidden2_dim);
    m.fc3 = linear_init(hidden2_dim, output_dim);
    return m;
}

static inline Tensor* mlp3_forward(MLP3* m, Tensor* x) {
    Tensor* h1 = linear_forward(&m->fc1, x);
    Tensor* a1 = tensor_relu(h1);
    Tensor* h2 = linear_forward(&m->fc2, a1);
    Tensor* a2 = tensor_relu(h2);
    return linear_forward(&m->fc3, a2);
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

static inline void sgd_zero_grad(SGD* opt) {
    for (int p = 0; p < opt->num_params; p++) {
        Tensor* param = opt->params[p];
        memset(param->grad, 0, param->shape[0] * param->shape[1] * sizeof(float));
    }
}

#endif
