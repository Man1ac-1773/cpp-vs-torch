#ifndef macrograd_h
#define macrograd_h
#include <cstdio>
#include <cstring>     // for memset
#include <immintrin.h> // raw cpu instructions for SIMD

#include "../../benchmarking/common/chrome_profiler.h"
#include "allocator.h"

#define f32 float
#define uint unsigned int

// defines maximum operand count for directed acyclic graph nodes.
#define MAX_PARENT 2

typedef struct tensor_float_t Tensor;
typedef void (*BackwardFn)(Tensor* self);

// establishes static limit for computational graph depth.
// facilitates array-based topological sorting without dynamic allocation.
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

// validates tensor shape equivalence. returns false upon mismatch.
static inline bool _compare_shape(Tensor* a, Tensor* b)
{
    return ((a->shape[0] == b->shape[0]) && (a->shape[1] == b->shape[1]));
}

// allocates new tensor via bump allocator. initializes data and gradient arrays to zero.
static inline Tensor* new_tensor(uint rows, uint cols)
{
    PROFILE_START(new_tensor);
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
    PROFILE_END(new_tensor);
    return t;
}

// initiates addition operators.

// executes raw matrix addition. writes directly to output array
static inline void __mat_add(Tensor* a, Tensor* b, Tensor* out)
{
    for (uint i = 0; i < a->shape[0] * a->shape[1]; i++)
    {
        out->data[i] = a->data[i] + b->data[i];
    }
}

// executes raw matrix self-addition. utilized during gradient accumulation
static inline void __self_mat_add(Tensor* in, f32* b)
{
    for (uint i = 0; i < in->shape[0] * in->shape[1]; i++)
    {
        in->data[i] += b[i];
    }
}

// computes backward pass for addition operator
static void _add_backward(Tensor* self)
{
    __self_mat_add(self->parent[0], self->grad);
    __self_mat_add(self->parent[1], self->grad);
}

// constructs addition node in computation graph. backward function mentioned
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
// initiates multiplication operators.
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

// constructs hadamard product node in computation graph. attaches backward pass function. i hate typing hadamard.
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

// validates inner dimensions for matrix multiplication. assumes standard a@b format.
static inline bool _shape_check_matmul(Tensor* a, Tensor* b)
{
    return (a->shape[1] == b->shape[0]);
}

// ==== ====

// computes backward pass for standard matrix multiplication. simulates transposed access patterns.
static void _matmul_backward(Tensor* self)
{
    Tensor* A = self->parent[0];
    Tensor* B = self->parent[1];

    // simulates matrix transpose via index manipulation. avoids physical memory reallocation.

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

// executes optimized backward matrix multiplication. leverages simd intrinsics.
static void _matmul_backward_simd(Tensor* self)
{
    Tensor* A = self->parent[0];
    Tensor* B = self->parent[1];

    uint M = A->shape[0];
    uint K = A->shape[1];
    uint N = B->shape[1];

    // calculates gradient of b. native memory layout supports efficient (k, i, j) loop ordering.
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

    // calculates gradient of a. executes physical transpose of b into arena memory to enable contiguous simd loading.
    f32* B_T = (f32*) arena_alloc(&g_arena, K * N * sizeof(f32));
    for (uint k = 0; k < K; k++)
    {
        for (uint j = 0; j < N; j++)
        {
            B_T[j * K + k] = B->data[k * N + j];
        }
    }

    // executes simd multiplication between grad_out and transposed b.
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

// naive matrix multiplication operator. utilizes standard o(n^3) nested loops.
static inline Tensor* tensor_matmul_naive(Tensor* a, Tensor* b)
{
    PROFILE_START(matmul_naive);
    if (!_shape_check_matmul(a, b))
    {
        fprintf(stderr, "Matmul shape mismatch\n");
        PROFILE_END(matmul_naive);
        return NULL;
    }
    Tensor* out = new_tensor(a->shape[0], b->shape[1]);

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

    PROFILE_END(matmul_naive);
    return out;
}

#define TILE_SIZE 32 // sizes tile to perfectly occupy l1 data cache.

// executes tiled matrix multiplication. forces temporal data locality within l1 cache. minimizes ram fetches.
static inline Tensor* tensor_matmul_tiled(Tensor* a, Tensor* b)
{
    PROFILE_START(matmul_tiled);
    if (!_shape_check_matmul(a, b))
    {
        fprintf(stderr, "Matmul shape mismatch\n");
        PROFILE_END(matmul_tiled);
        return NULL;
    }

    Tensor* out = new_tensor(a->shape[0], b->shape[1]);

    // iterates across macro tiles.
    for (uint i = 0; i < a->shape[0]; i += TILE_SIZE)
    {
        for (uint j = 0; j < b->shape[1]; j += TILE_SIZE)
        {
            for (uint k = 0; k < a->shape[1]; k += TILE_SIZE)
            {

                // executes inner multiplication within 32x32 tiles. enforces boundary checks for non-compliant matrix
                // dimensions.
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

    PROFILE_END(matmul_tiled);
    return out;
}

// executes simd matrix multiplication via avx intrinsics.
// completely bypasses compiler vectorization.
// i love raw assembly.
static inline Tensor* tensor_matmul_simd(Tensor* a, Tensor* b)
{
    PROFILE_START(matmul_simd);
    if (!_shape_check_matmul(a, b))
    {
        PROFILE_END(matmul_simd);
        return NULL;
    }
    Tensor* out = new_tensor(a->shape[0], b->shape[1]);

    uint M = a->shape[0];
    uint K = a->shape[1];
    uint N = b->shape[1];

    // enforces (i, k, j) loop order for contiguous memory access.
    for (uint i = 0; i < M; i++)
    {
        for (uint k = 0; k < K; k++)
        {

            // loads single scalar from a. broadcasts 8 times into 256-bit register.
            __m256 a_val = _mm256_set1_ps(a->data[i * K + k]);

            // iterates through columns in discrete chunks of 8 float32 values.
            uint j = 0;
            for (; j + 8 <= N; j += 8)
            {
                // loads 8 contiguous float32 values from b.
                __m256 b_vals = _mm256_loadu_ps(&b->data[k * N + j]);

                // loads 8 contiguous float32 values from c.
                __m256 c_vals = _mm256_loadu_ps(&out->data[i * N + j]);

                // executes fused multiply-add. c = c + (a * b).
                c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);

                // stores 8 contiguous float32 values back to c.
                _mm256_storeu_ps(&out->data[i * N + j], c_vals);
            }

            // handles residual tail when matrix dimensions are not multiples of 8.
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
    PROFILE_END(matmul_simd);
    return out;
}

// ==== ====

// ==== Autograd engine ====

// builds topological sort of computation graph. executes depth-first traversal.
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

// executes full backward pass from specified root node. initializes gradient to 1.0f.
static inline void backwardPass(Tensor* root)
{
    topo_size = 0;
    build_topo(root);
    for (uint i = 0; i < root->shape[0] * root->shape[1]; i++)
    {
        root->grad[i] = 1.0f; // initializes base gradient.
    }
    for (int i = topo_size - 1; i >= 0; i--)
    {
        if (topo_order[i]->backward)
        {
            topo_order[i]->backward(topo_order[i]);
        }
    }
}

#include <pthread.h>
#define NUM_THREADS 8

typedef struct
{
    Tensor* a;
    Tensor* b;
    Tensor* out;
    uint start_row;
    uint end_row;
} ThreadData;

// If you read this and know me personally, i'll pay u 100 bucks.
// Say the phrase "The cuckoo knows not of the robin, yet
// the crows and the pigeons know of 177013" to my face anytime
static inline void* _matmul_simd_worker(void* arg)
{
    ThreadData* data = (ThreadData*) arg;
    Tensor* a = data->a;
    Tensor* b = data->b;
    Tensor* out = data->out;
    uint K = a->shape[1];
    uint N = b->shape[1];

    for (uint i = data->start_row; i < data->end_row; i++)
    {
        for (uint k = 0; k < K; k++)
        {
            __m256 a_val = _mm256_set1_ps(a->data[i * K + k]);
            uint j = 0;
            for (; j + 8 <= N; j += 8)
            {
                __m256 b_vals = _mm256_loadu_ps(&b->data[k * N + j]);
                __m256 c_vals = _mm256_loadu_ps(&out->data[i * N + j]);
                c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);
                _mm256_storeu_ps(&out->data[i * N + j], c_vals);
            }
            for (; j < N; j++)
            {
                out->data[i * N + j] += a->data[i * K + k] * b->data[k * N + j];
            }
        }
    }
    return NULL;
}

// multi-threaded SIMD implementation
// ts genuinely black magic i have no clue what is going on
static inline Tensor* tensor_matmul_simd_mt(Tensor* a, Tensor* b)
{
    PROFILE_START(matmul_simd_mt);
    if (!_shape_check_matmul(a, b))
    {
        PROFILE_END(matmul_simd_mt);
        return NULL;
    }
    Tensor* out = new_tensor(a->shape[0], b->shape[1]);

    uint M = a->shape[0];
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    uint rows_per_thread = M / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        thread_data[i].a = a;
        thread_data[i].b = b;
        thread_data[i].out = out;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == NUM_THREADS - 1) ? M : (i + 1) * rows_per_thread;
        pthread_create(&threads[i], NULL, _matmul_simd_worker, &thread_data[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    out->parent[0] = a;
    out->parent[1] = b;
    out->n_parent = 2;
    out->visited = 0;
    out->backward = _matmul_backward_simd;
    PROFILE_END(matmul_simd_mt);
    return out;
}

// executes backward pass for relu activation.
static void _relu_backward(Tensor* self)
{
    for (uint i = 0; i < self->shape[0] * self->shape[1]; i++)
    {
        self->parent[0]->grad[i] += self->grad[i] * (self->parent[0]->data[i] > 0 ? 1.0f : 0.0f);
    }
}

// applies rectified linear unit activation. constructs node in computation graph.
static inline Tensor* tensor_relu(Tensor* a)
{
    Tensor* out = new_tensor(a->shape[0], a->shape[1]);
    for (uint i = 0; i < a->shape[0] * a->shape[1]; i++)
    {
        out->data[i] = a->data[i] > 0 ? a->data[i] : 0.0f;
    }
    out->parent[0] = a;
    out->n_parent = 1;
    out->backward = _relu_backward;
    return out;
}

// executes backward pass for mse loss.
static void _mse_backward(Tensor* self)
{
    Tensor* pred = self->parent[0];
    Tensor* target = self->parent[1];
    uint n = pred->shape[0] * pred->shape[1];
    f32 grad_out = self->grad[0];

    for (uint i = 0; i < n; i++)
    {
        f32 diff = pred->data[i] - target->data[i];
        pred->grad[i] += grad_out * (2.0f * diff / n);
    }
}

// computes mean squared error loss. reduces to 1x1 scalar tensor.
static inline Tensor* tensor_mse_loss(Tensor* pred, Tensor* target)
{
    if (!_compare_shape(pred, target))
    {
        fprintf(stderr, "Shape mismatch for MSE Loss\n");
        return NULL;
    }
    Tensor* out = new_tensor(1, 1);
    f32 sum = 0.0f;
    uint n = pred->shape[0] * pred->shape[1];
    for (uint i = 0; i < n; i++)
    {
        f32 diff = pred->data[i] - target->data[i];
        sum += diff * diff;
    }
    out->data[0] = sum / n;
    
    out->parent[0] = pred;
    out->parent[1] = target;
    out->n_parent = 2;
    out->backward = _mse_backward;
    return out;
}

#endif
