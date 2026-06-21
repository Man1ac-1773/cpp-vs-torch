#ifndef macrograd_h
#define macrograd_h
#include <cstdio>
#include <cstring>     // for memset
#include <immintrin.h> // raw cpu instructions for SIMD

#include "allocator.h"

// typedef for easier typinh
#define f32 float
#define uint unsigned int

// for computation graph;
// defines that an operation can only have two operands
#define MAX_PARENT 2

typedef struct tensor_float_t Tensor;
typedef void (*BackwardFn)(Tensor* self);

// defining limits of computational graph
// for ease in building topological sort
#define MAX_GRAPH_NODES 10000
static Tensor* topo_order[MAX_GRAPH_NODES];
static int topo_size = 0;

struct tensor_float_t
{
    f32* data;
    f32* grad;
    uint shape[2];  // 2 dimensional for now
    uint stride[2]; // byte offset in any direction
    Tensor* parent[MAX_PARENT];
    int n_parent;
    BackwardFn backward;
    int visited;
};

static Arena g_arena = {{NULL}, 0, 0};

// ==== ====

// Shape checking function; compare shapes of two tensors
// returns `false` if shape mismatch
static inline bool _compare_shape(Tensor* a, Tensor* b)
{
    return ((a->shape[0] == b->shape[0]) && (a->shape[1] == b->shape[1]));
}

// Create new tensor with given dimensions, and operation
static inline Tensor* new_tensor(uint rows, uint cols)
{
    Tensor* t = (Tensor*) arena_alloc(&g_arena, sizeof(Tensor));

    size_t size_req = rows * cols * sizeof(f32);
    t->data = (f32*) arena_alloc(&g_arena, size_req);
    t->grad = (f32*) arena_alloc(&g_arena, size_req);

    t->shape[0] = rows;
    t->shape[1] = cols;
    t->stride[0] = cols;
    t->stride[1] = 1;
    t->n_parent = 0;
    t->backward = NULL;
    t->visited = 0;

    memset(t->data, 0, size_req);
    memset(t->grad, 0, size_req);
    return t;
}

// ==== ADDITION OPERATIONS ====

// matrix a + b
// employed only to write data
// does not perform safety checks
static inline void __mat_add(Tensor* a, Tensor* b, Tensor* out)
{
    for (uint i = 0; i < a->shape[0] * a->shape[1]; i++)
    {
        out->data[i] = a->data[i] + b->data[i];
    }
}

// matrix a += b;
// employed to write data
// no safety checks
static inline void __self_mat_add(Tensor* in, f32* b)
{
    for (uint i = 0; i < in->shape[0] * in->shape[1]; i++)
    {
        in->data[i] += b[i];
    }
}

// backprop internal function for ADDITION
static void _add_backward(Tensor* self)
{
    __self_mat_add(self->parent[0], self->grad);
    __self_mat_add(self->parent[1], self->grad);
}

// Add two tensor (tracks gradient)
static inline Tensor* tensor_add(Tensor* a, Tensor* b)
{
    if (!_compare_shape(a, b))
    {
        fprintf(stderr, "Shape mismatch when addition\n");
    }
    Tensor* out = new_tensor(a->shape[0], a->shape[1]);
    __mat_add(a, b, out); // just write data

    // gradient tracking
    out->parent[0] = a;
    out->parent[1] = b;
    out->n_parent = 2;
    out->backward = _add_backward;
    return out;
}

// ==== ====
//
// ==== MULTIPLY OPERATIONS ====
static inline void __mat_hadmard_mul(Tensor* a, Tensor* b, Tensor* out)
{
    for (uint i = 0; i < a->shape[0] * a->shape[1]; i++)
    {
        out->data[i] = a->data[i] * b->data[i];
    }
}

static void _hadmard_mul_backward(Tensor* self)
{
    for (uint i = 0; i < self->shape[0] * self->shape[1]; i++)
    {
        self->parent[0]->grad[i] += self->parent[1]->data[i] * self->grad[i];
        self->parent[1]->grad[i] += self->parent[0]->data[i] * self->grad[i];
    }
}

// Tensor Hadmard product (tracks gradient)
static inline Tensor* tensor_hadmard(Tensor* a, Tensor* b)
{
    if (!_compare_shape(a, b))
    {
        fprintf(stderr, "Shape mismatch for hadmard product\n");
        return NULL;
    }
    Tensor* out = new_tensor(a->shape[0], a->shape[1]);
    __mat_hadmard_mul(a, b, out);
    out->parent[0] = a;
    out->parent[1] = b;
    out->n_parent = 2;
    out->backward = _hadmard_mul_backward;
    return out;
}

// asummes matmul of the form a@b
static inline bool _shape_check_matmul(Tensor* a, Tensor* b)
{
    return (a->shape[1] == b->shape[0]);
}

// ==== ====

// backprop function for standard matmul
static void _matmul_backward(Tensor* self)
{
    Tensor* A = self->parent[0];
    Tensor* B = self->parent[1];

    // we simulate transpose without actually doing it

    // grad_A += upstream_grad @ B^T
    for (uint i = 0; i < A->shape[0]; i++)
    {
        for (uint j = 0; j < B->shape[1]; j++)
        {
            for (uint k = 0; k < A->shape[1]; k++)
            {
                // B is accessed as [k, j] instead of [j, k]
                A->grad[i * A->shape[1] + k] += self->grad[i * self->shape[1] + j] * B->data[k * B->shape[1] + j];
            }
        }
    }
    // grad_B += A^T @ upstream_grad
    for (uint k = 0; k < A->shape[1]; k++)
    {
        for (uint j = 0; j < B->shape[1]; j++)
        {
            for (uint i = 0; i < A->shape[0]; i++)
            {
                //  A is accessed as [i, k] instead of [k, i]
                B->grad[k * B->shape[1] + j] += A->data[i * A->shape[1] + k] * self->grad[i * self->shape[1] + j];
            }
        }
    }
}

// optimized backward matmul
// with simd instructions
static void _matmul_backward_simd(Tensor* self)
{
    Tensor* A = self->parent[0];
    Tensor* B = self->parent[1];

    uint M = A->shape[0];
    uint K = A->shape[1];
    uint N = B->shape[1];

    // 1. Calculate grad_B (A^T @ grad_out) natively using loop order (k, i, j)
    for (uint k = 0; k < K; k++)
    {
        for (uint i = 0; i < M; i++)
        {
            __m256 a_val = _mm256_set1_ps(A->data[i * K + k]);
            uint j = 0;
            for (; j + 8 <= N; j += 8)
            {
                __m256 grad_out_vals = _mm256_loadu_ps(&self->grad[i * N + j]);
                __m256 grad_b_vals = _mm256_loadu_ps(&B->grad[k * N + j]);

                grad_b_vals = _mm256_fmadd_ps(a_val, grad_out_vals, grad_b_vals);
                _mm256_storeu_ps(&B->grad[k * N + j], grad_b_vals);
            }
            for (; j < N; j++)
            {
                B->grad[k * N + j] += A->data[i * K + k] * self->grad[i * N + j];
            }
        }
    }

    // 2. Calculate grad_A (grad_out @ B^T)
    // Step 2a: Physically transpose B into the Arena! (N x K)
    f32* B_T = (f32*) arena_alloc(&g_arena, K * N * sizeof(f32));
    for (uint k = 0; k < K; k++)
    {
        for (uint j = 0; j < N; j++)
        {
            B_T[j * K + k] = B->data[k * N + j];
        }
    }

    // Step 2b: SIMD multiply grad_out @ B_T with loop order (i, j, k)
    for (uint i = 0; i < M; i++)
    {
        for (uint j = 0; j < N; j++)
        {
            __m256 grad_out_val = _mm256_set1_ps(self->grad[i * N + j]);
            uint k = 0;
            for (; k + 8 <= K; k += 8)
            {
                __m256 bt_vals = _mm256_loadu_ps(&B_T[j * K + k]);
                __m256 grad_a_vals = _mm256_loadu_ps(&A->grad[i * K + k]);

                grad_a_vals = _mm256_fmadd_ps(grad_out_val, bt_vals, grad_a_vals);
                _mm256_storeu_ps(&A->grad[i * K + k], grad_a_vals);
            }
            for (; k < K; k++)
            {
                A->grad[i * K + k] += self->grad[i * N + j] * B_T[j * K + k];
            }
        }
    }
}

// Tensor matrix multiplication
static inline Tensor* tensor_matmul_naive(Tensor* a, Tensor* b)
{
    if (!_shape_check_matmul(a, b))
    {
        fprintf(stderr, "Matmul shape mismatch\n");
        return NULL;
    }
    Tensor* out = new_tensor(a->shape[0], b->shape[1]);
    // Standard Naive GEMM (O(N^3))
    for (uint i = 0; i < a->shape[0]; i++)
    {
        for (uint j = 0; j < b->shape[1]; j++)
        {
            for (uint k = 0; k < a->shape[1]; k++)
            {
                out->data[i * b->shape[1] + j] += a->data[i * a->shape[1] + k] * b->data[k * b->shape[1] + j];
            }
        }
    }

    out->parent[0] = a;
    out->parent[1] = b;
    out->n_parent = 2;
    out->visited = 0;
    out->backward = _matmul_backward;

    return out;
}

#define TILE_SIZE 32 // Fits in L1 Cache

// Slightly cache friendly matmul
// Pushes a tile into L1 then matmuls it
static inline Tensor* tensor_matmul_tiled(Tensor* a, Tensor* b)
{
    if (!_shape_check_matmul(a, b))
    {
        fprintf(stderr, "Matmul shape mismatch\n");
        return NULL;
    }

    Tensor* out = new_tensor(a->shape[0], b->shape[1]);

    // navigating between tiles)
    for (uint i = 0; i < a->shape[0]; i += TILE_SIZE)
    {
        for (uint j = 0; j < b->shape[1]; j += TILE_SIZE)
        {
            for (uint k = 0; k < a->shape[1]; k += TILE_SIZE)
            {

                // multiplying the 32x32 tiles
                // we use min bounds checking in case matrix dimensions arent multiples of 32
                for (uint ii = i; ii < i + TILE_SIZE && ii < a->shape[0]; ii++)
                {
                    for (uint jj = j; jj < j + TILE_SIZE && jj < b->shape[1]; jj++)
                    {
                        for (uint kk = k; kk < k + TILE_SIZE && kk < a->shape[1]; kk++)
                        {

                            out->data[ii * b->shape[1] + jj] +=
                                a->data[ii * a->shape[1] + kk] * b->data[kk * b->shape[1] + jj];
                        }
                    }
                }
            }
        }
    }

    out->parent[0] = a;
    out->parent[1] = b;
    out->n_parent = 2;
    out->visited = 0;

    out->backward = _matmul_backward;

    return out;
}

// SIMD matmul with raw cpu instructions
// written with gemini
static inline Tensor* tensor_matmul_simd(Tensor* a, Tensor* b)
{
    if (!_shape_check_matmul(a, b))
        return NULL;
    Tensor* out = new_tensor(a->shape[0], b->shape[1]);

    uint M = a->shape[0];
    uint K = a->shape[1];
    uint N = b->shape[1];

    // loop order (i, k, j)
    for (uint i = 0; i < M; i++)
    {
        for (uint k = 0; k < K; k++)
        {

            // We load a single scalar from A, and "broadcast" it 8 times into a 256-bit register
            __m256 a_val = _mm256_set1_ps(a->data[i * K + k]);

            // We step through j in chunks of 8 floats!
            uint j = 0;
            for (; j + 8 <= N; j += 8)
            {
                // Load 8 contiguous floats from B
                __m256 b_vals = _mm256_loadu_ps(&b->data[k * N + j]);

                // Load 8 contiguous floats from C (our output)
                __m256 c_vals = _mm256_loadu_ps(&out->data[i * N + j]);

                // Fused Multiply-Add: C = C + (A * B)
                c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);

                // Store the 8 floats back to C
                _mm256_storeu_ps(&out->data[i * N + j], c_vals);
            }

            // Handle the "tail" if the matrix width isn't a perfect multiple of 8
            for (; j < N; j++)
            {
                out->data[i * N + j] += a->data[i * K + k] * b->data[k * N + j];
            }
        }
    }

    out->parent[0] = a;
    out->parent[1] = b;
    out->n_parent = 2;
    out->visited = 0;
    out->backward = _matmul_backward_simd;
    return out;
}

// ==== ====

// ==== Autograd engine ====

// Build the topographically sorted array of nodes
// used in the computation graph
static inline void build_topo(Tensor* node)
{
    if (!node || node->visited == 1)
    {
        return;
    }
    node->visited = 1;
    for (uint i = 0; i < node->n_parent; i++)
    {
        build_topo(node->parent[i]);
    }
    topo_order[topo_size++] = node;
}

// Run the backward pass on a given root.
// Root's dimensions are not assumed.
static inline void backwardPass(Tensor* root)
{
    topo_size = 0;
    build_topo(root);
    for (uint i = 0; i < root->shape[0] * root->shape[1]; i++)
    {
        root->grad[i] = 1.0f; // set grad baseline
    }
    for (int i = topo_size - 1; i >= 0; i--)
    {
        if (topo_order[i]->backward)
        {
            topo_order[i]->backward(topo_order[i]);
        }
    }
}

#endif
