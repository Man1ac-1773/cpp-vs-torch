#ifndef macrograd_h
#define macrograd_h
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#define f32 float
#define MAX_PARENT 2
#define ARENA_PAGE_SIZE (4 * 1024 * 1024) // 4 MB since tensors are large
#define MAX_PAGES 1024

typedef enum tensor_ops
{
    ADD = 1,
    HADMUL,
    MATMUL,

} OPS;

typedef struct tensor_float_t Tensor;
typedef void (*BackwardFn)(Tensor* self);

struct tensor_float_t
{
    f32* data;
    f32* grad;
    uint shape[2];  // 2 dimensional for now
    uint stride[2]; // byte offset in any direction
    Tensor* parent[MAX_PARENT];
    int n_parent;
    OPS op;
    BackwardFn backward;
    int visited;
};

typedef struct
{
    uint8_t* pages[MAX_PAGES];
    int num_pages;
    int top;
} Arena;

static Arena g_arena = {{NULL}, 0, 0};

// ==== ARENA FUNCTIONS ====
static inline void arena_init()
{
    if (g_arena.num_pages > 0)
        return;
    g_arena.pages[0] = (uint8_t*) malloc(ARENA_PAGE_SIZE);
    if (!g_arena.pages[0])
    {
        fprintf(stderr, "Arena init failed\n");
        exit(1);
    }
    g_arena.num_pages = 1;
    g_arena.top = 0;
}

static inline void arena_grow()
{
    if (g_arena.num_pages >= MAX_PAGES)
    {
        fprintf(stderr, "Arena overflow\n");
        exit(1);
    }
    g_arena.pages[g_arena.num_pages] = (uint8_t*) malloc(ARENA_PAGE_SIZE);
    if (!g_arena.pages[g_arena.num_pages])
    {
        fprintf(stderr, "Arena grow failed\n");
        exit(1);
    }
    g_arena.num_pages++;
}

// Helper to bump up to 32 byte allocation for later AVX SIMD instructions
static inline size_t align32(size_t size)
{
    return (size + 31) & ~31;
}

static inline void* arena_alloc(size_t size)
{
    if (g_arena.num_pages == 0)
        arena_init();
    size_t aligned_size = align32(size);

    if (g_arena.top + aligned_size > ARENA_PAGE_SIZE)
    {
        arena_grow();
        g_arena.top = 0;
    }
    void* ptr = g_arena.pages[g_arena.num_pages - 1] + g_arena.top;
    g_arena.top += aligned_size;

    return ptr;
}

static inline int get_arena_top()
{
    return g_arena.top;
}

static inline void reset_arena_and_zero_grad(int mark)
{
    g_arena.top = mark;
    for (int i = 0; i < mark; i++)
    {
        int p = i / ARENA_PAGE_SIZE;
        int o = i % ARENA_PAGE_SIZE;
        g_arena.pages[p][o].grad = {0.0};
    }
}

static inline void free_arena()
{
    for (int i = 0; i < g_arena.num_pages; i++)
    {
        free(g_arena.pages[i]);
        g_arena.pages[i] = NULL;
    }
    g_arena.num_pages = 0;
    g_arena.top = 0;
}

// ==== ====

// Shape checking function; compare shapes of two tensors
// returns `false` if shape mismatch
static inline bool _compare_shape(Tensor* a, Tensor* b)
{
    return ((a->shape[0] == b->shape[0]) && (a->shape[1] == b->shape[1]));
}

// Create new tensor with given dimensions, and operation
static inline Tensor* new_tensor(uint rows, uint cols, OPS op)
{
    Tensor* t = (Tensor*) arena_alloc(sizeof(Tensor));

    size_t size_req = rows * cols * sizeof(f32);
    t->data = (f32*) arena_alloc(size_req);
    t->grad = (f32*) arena_alloc(size_req);

    t->shape[0] = rows;
    t->shape[1] = cols;
    t->stride[0] = cols;
    t->stride[1] = 1;
    t->n_parent = 0;
    t->op = op;
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
    Tensor* out = new_tensor(a->shape[0], a->shape[1], ADD);
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
static inline Tensor* hadmard_mul(Tensor* a, Tensor* b)
{
    if (!_compare_shape(a, b))
    {
        fprintf(stderr, "Shape mismatch for hadmard product\n");
        return NULL;
    }
    Tensor* out = new_tensor(a->shape[0], a->shape[1], HADMUL);
    out->parent[0] = a;
    out->parent[1] = b;
    out->n_parent = 2;
    out->backward = _hadmard_mul_backward;
    return out;
}

static void tanh_backward(Value* self)
{
    double t = self->data;
    self->parent[0]->grad += (1.0 - t * t) * self->grad;
}

static inline Value* val_tanh(Value* a)
{
    Value* out = new_value(tanh(a->data), "tanh");
    out->parent[0] = a;
    out->n_parent = 1;
    out->backward = tanh_backward;
    return out;
}

static void relu_backward(Value* self)
{
    self->parent[0]->grad += ((self->data > 0) ? 1.0 : 0.0) * self->grad;
}

static inline Value* val_relu(Value* a)
{
    Value* out = new_value((a->data > 0) ? a->data : 0.0, "relu");
    out->parent[0] = a;
    out->n_parent = 1;
    out->backward = relu_backward;
    return out;
}
static void log_backward(Value* self)
{
    Value* a = self->parent[0];
    a->grad += (1.0 / a->data) * self->grad;
}

static inline Value* val_log(Value* a)
{
    Value* out = new_value(log(a->data), "log");
    out->parent[0] = a;
    out->n_parent = 1;
    out->backward = log_backward;
    return out;
}

static void exp_backward(Value* self)
{
    Value* a = self->parent[0];
    a->grad += self->data * self->grad;
}

static inline Value* val_exp(Value* a)
{
    Value* out = new_value(exp(a->data), "exp");
    out->parent[0] = a;
    out->n_parent = 1;
    out->backward = exp_backward;
    return out;
}

static inline void backwardPass(Value* root)
{
    if (!root)
        return;
    root->grad = 1.0;
    for (int i = g_arena.top - 1; i >= 0; i--)
    {
        int p = i / ARENA_PAGE_SIZE;
        int o = i % ARENA_PAGE_SIZE;
        Value* v = &g_arena.pages[p][o];
        if (v->backward)
            v->backward(v);
    }
}

#endif
