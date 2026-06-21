#pragma once
#include <cassert>
#include <functional>
#include <immintrin.h>
#include <memory>
#include <unordered_set>
#include <vector>
#include "../../benchmarking/chrome_profiler.h"

using namespace std;

// inherit from enable_shared_from_this so we can pass 'this' into our lambda captures safely
class TensorNode : public enable_shared_from_this<TensorNode>
{
  public:
    size_t rows, cols;
    vector<float> data;
    vector<float> grad;

    vector<shared_ptr<TensorNode>> _prev;
    function<void()> _backward;

    TensorNode(size_t _rows, size_t _cols) : rows(_rows), cols(_cols)
    {
        data.resize(rows * cols, 0.0f);
        grad.resize(rows * cols, 0.0f);
        _backward = []() {}; // empty lambda
    }
};

class Tensor
{
  public:
    shared_ptr<TensorNode> node;

    // public constructor
    Tensor(size_t _rows, size_t _cols)
    {
        node = make_shared<TensorNode>(_rows, _cols);
    }

    // internal constructor
    // when constructing from existing node
    Tensor(shared_ptr<TensorNode> _node) : node(_node) {}

    // getters for data read and write
    float& operator()(size_t r, size_t c)
    {
        return node->data[r * node->cols + c];
    }
    const float& operator()(size_t r, size_t c) const
    {
        return node->data[r * node->cols + c];
    }

    void backward()
    {
        vector<shared_ptr<TensorNode>> topo;
        unordered_set<TensorNode*> visited;
        function<void(shared_ptr<TensorNode>)> build_topo = [&](shared_ptr<TensorNode> v)
        {
            if (visited.insert(v.get()).second)
            {
                for (auto& p : v->_prev)
                {
                    build_topo(p);
                }
                topo.push_back(v);
            }
        };

        build_topo(node);

        for (size_t i = 0; i < node->grad.size(); i++)
        {
            node->grad[i] = 0.0f;
        }

        for (auto it = topo.rbegin(); it != topo.rend(); it++)
        {
            (*it)->_backward();
        }
    }
};

inline Tensor operator+(const Tensor& a, const Tensor& b)
{
    PROFILE_FUNCTION();
    assert(a.node->rows == b.node->rows && a.node->cols == b.node->cols);

    Tensor out(a.node->rows, a.node->cols);

    for (size_t i = 0; i < a.node->data.size(); i++)
    {
        out.node->data[i] = a.node->data[i] + b.node->data[i];
    }

    out.node->_prev = {a.node, b.node};

    // lambda copies shared_ptr
    // getting raw pointer of out node, because otherwise it would be a cyclical reference
    // and ptr count would never hit 0, meaning no memory cleanup
    out.node->_backward = [a_node = a.node, b_node = b.node, out_node = out.node.get()]()
    {
        for (size_t i = 0; i < a_node->data.size(); i++)
        {
            a_node->grad[i] += out_node->grad[i];
            b_node->grad[i] += out_node->grad[i];
        }
    };

    return out;
}

// Naive matmul operator
// uses unoptimized O(N^3) GEMM
inline Tensor operator*(const Tensor& a, const Tensor& b)
{
    PROFILE_FUNCTION();
    assert(a.node->cols == b.node->rows);

    Tensor out(a.node->rows, b.node->cols);

    for (size_t i = 0; i < a.node->rows; i++)
    {
        for (size_t j = 0; j < b.node->cols; j++)
        {
            for (size_t k = 0; k < a.node->cols; k++)
            {
                out.node->data[i * b.node->cols + j] +=
                    a.node->data[i * a.node->cols + k] * b.node->data[k * b.node->cols + j];
            }
        }
    }
    out.node->_prev = {a.node, b.node};

    out.node->_backward = [a_node = a.node, b_node = b.node, out_node = out.node.get()]()
    {
        // grad_A += grad_out @B ^ T
        for (size_t i = 0; i < a_node->rows; i++)
        {
            for (size_t j = 0; j < b_node->cols; j++)
            {
                for (size_t k = 0; k < a_node->cols; k++)
                {
                    a_node->grad[i * a_node->cols + k] +=
                        out_node->grad[i * out_node->cols + j] * b_node->data[k * b_node->cols + j];
                }
            }
        }

        // grad_B += A^T @ grad_out
        for (size_t k = 0; k < a_node->cols; k++)
        {
            for (size_t j = 0; j < b_node->cols; j++)
            {
                for (size_t i = 0; i < a_node->rows; i++)
                {
                    b_node->grad[k * b_node->cols + j] +=
                        a_node->data[i * a_node->cols + k] * out_node->grad[i * out_node->cols + j];
                }
            }
        }
    };

    return out;
}

// Tiled, cache-friendly matmul operator
// TILE_SIZE = 32
inline Tensor matmul_tiled(const Tensor& a, const Tensor& b)
{
    PROFILE_FUNCTION();
    assert(a.node->cols == b.node->rows);
    Tensor out(a.node->rows, b.node->cols);

    const uint TILE_SIZE = 32;
    // Exactly the same 6-loop structure as macrograd!
    for (size_t i = 0; i < a.node->rows; i += TILE_SIZE)
    {
        for (size_t j = 0; j < b.node->cols; j += TILE_SIZE)
        {
            for (size_t k = 0; k < a.node->cols; k += TILE_SIZE)
            {
                for (size_t ii = i; ii < i + TILE_SIZE && ii < a.node->rows; ii++)
                {
                    for (size_t jj = j; jj < j + TILE_SIZE && jj < b.node->cols; jj++)
                    {
                        for (size_t kk = k; kk < k + TILE_SIZE && kk < a.node->cols; kk++)
                        {
                            out(ii, jj) += a(ii, kk) * b(kk, jj);
                        }
                    }
                }
            }
        }
    }

    out.node->_prev = {a.node, b.node};
    // naive backward pass. does not use tiling
    out.node->_backward = [a_node = a.node, b_node = b.node, out_node = out.node.get()]()
    {
        for (size_t i = 0; i < a_node->rows; i++)
        {
            for (size_t j = 0; j < b_node->cols; j++)
            {
                for (size_t k = 0; k < a_node->cols; k++)
                {
                    a_node->grad[i * a_node->cols + k] +=
                        out_node->grad[i * out_node->cols + j] * b_node->data[k * b_node->cols + j];
                }
            }
        }

        for (size_t k = 0; k < a_node->cols; k++)
        {
            for (size_t j = 0; j < b_node->cols; j++)
            {
                for (size_t i = 0; i < a_node->rows; i++)
                {
                    b_node->grad[k * b_node->cols + j] +=
                        a_node->data[i * a_node->cols + k] * out_node->grad[i * out_node->cols + j];
                }
            }
        }
    };

    return out;
}

// Optimized SIMD AVX enabled matmul
// backward pass also uses SIMD for high optimization
inline Tensor matmul_simd(const Tensor& a, const Tensor& b)
{
    PROFILE_FUNCTION();
    assert(a.node->cols == b.node->rows);
    Tensor out(a.node->rows, b.node->cols);

    size_t M = a.node->rows;
    size_t K = a.node->cols;
    size_t N = b.node->cols;

    // ==== FORWARD PASS (SIMD) ====
    for (size_t i = 0; i < M; i++)
    {
        for (size_t k = 0; k < K; k++)
        {
            __m256 a_val = _mm256_set1_ps(a(i, k));
            size_t j = 0;
            for (; j + 8 <= N; j += 8)
            {
                __m256 b_vals = _mm256_loadu_ps(&b(k, j));
                __m256 c_vals = _mm256_loadu_ps(&out(i, j));

                c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);
                _mm256_storeu_ps(&out(i, j), c_vals);
            }
            for (; j < N; j++)
            {
                out(i, j) += a(i, k) * b(k, j);
            }
        }
    }

    out.node->_prev = {a.node, b.node};

    // ==== BACKWARD PASS (SIMD) ====
    out.node->_backward = [a_node = a.node, b_node = b.node, out_node = out.node.get(), M, K, N]()
    {
        // grad_B += A^T @ grad_out
        for (size_t k = 0; k < K; k++)
        {
            for (size_t i = 0; i < M; i++)
            {
                __m256 a_val = _mm256_set1_ps(a_node->data[i * K + k]);
                size_t j = 0;
                for (; j + 8 <= N; j += 8)
                {
                    __m256 grad_out_vals = _mm256_loadu_ps(&out_node->grad[i * N + j]);
                    __m256 grad_b_vals = _mm256_loadu_ps(&b_node->grad[k * N + j]);

                    grad_b_vals = _mm256_fmadd_ps(a_val, grad_out_vals, grad_b_vals);
                    _mm256_storeu_ps(&b_node->grad[k * N + j], grad_b_vals);
                }
                for (; j < N; j++)
                {
                    b_node->grad[k * N + j] += a_node->data[i * K + k] * out_node->grad[i * N + j];
                }
            }
        }

        // grad_A += grad_out @ B^T
        std::vector<float> B_T(K * N);
        for (size_t k = 0; k < K; k++)
        {
            for (size_t j = 0; j < N; j++)
            {
                B_T[j * K + k] = b_node->data[k * N + j];
            }
        }

        for (size_t i = 0; i < M; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                __m256 grad_out_val = _mm256_set1_ps(out_node->grad[i * N + j]);
                size_t k = 0;
                for (; k + 8 <= K; k += 8)
                {
                    __m256 bt_vals = _mm256_loadu_ps(&B_T[j * K + k]);
                    __m256 grad_a_vals = _mm256_loadu_ps(&a_node->grad[i * K + k]);

                    grad_a_vals = _mm256_fmadd_ps(grad_out_val, bt_vals, grad_a_vals);
                    _mm256_storeu_ps(&a_node->grad[i * K + k], grad_a_vals);
                }
                for (; k < K; k++)
                {
                    a_node->grad[i * K + k] += out_node->grad[i * N + j] * B_T[j * K + k];
                }
            }
        }
    };

    return out;
}

// Multithreaded SIMD AVX Matmul (OpenMP)
inline Tensor matmul_simd_mt(const Tensor& a, const Tensor& b)
{
    PROFILE_FUNCTION();
    assert(a.node->cols == b.node->rows);
    Tensor out(a.node->rows, b.node->cols);

    size_t M = a.node->rows;
    size_t K = a.node->cols;
    size_t N = b.node->cols;

    // ==== FORWARD PASS (SIMD + OpenMP) ====
    #pragma omp parallel for
    for (size_t i = 0; i < M; i++)
    {
        for (size_t k = 0; k < K; k++)
        {
            __m256 a_val = _mm256_set1_ps(a(i, k));
            size_t j = 0;
            for (; j + 8 <= N; j += 8)
            {
                __m256 b_vals = _mm256_loadu_ps(&b(k, j));
                __m256 c_vals = _mm256_loadu_ps(&out(i, j));

                c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);
                _mm256_storeu_ps(&out(i, j), c_vals);
            }
            for (; j < N; j++)
            {
                out(i, j) += a(i, k) * b(k, j);
            }
        }
    }

    out.node->_prev = {a.node, b.node};
    // backward pass (using the same logic as single threaded simd)
    out.node->_backward = [a_node = a.node, b_node = b.node, out_node = out.node.get(), M, K, N]()
    {
        for (size_t k = 0; k < K; k++)
        {
            for (size_t i = 0; i < M; i++)
            {
                __m256 a_val = _mm256_set1_ps(a_node->data[i * K + k]);
                size_t j = 0;
                for (; j + 8 <= N; j += 8)
                {
                    __m256 grad_out_vals = _mm256_loadu_ps(&out_node->grad[i * N + j]);
                    __m256 grad_b_vals = _mm256_loadu_ps(&b_node->grad[k * N + j]);

                    grad_b_vals = _mm256_fmadd_ps(a_val, grad_out_vals, grad_b_vals);
                    _mm256_storeu_ps(&b_node->grad[k * N + j], grad_b_vals);
                }
                for (; j < N; j++)
                {
                    b_node->grad[k * N + j] += a_node->data[i * K + k] * out_node->grad[i * N + j];
                }
            }
        }

        std::vector<float> B_T(K * N);
        for (size_t k = 0; k < K; k++)
        {
            for (size_t j = 0; j < N; j++)
            {
                B_T[j * K + k] = b_node->data[k * N + j];
            }
        }

        for (size_t i = 0; i < M; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                __m256 grad_out_val = _mm256_set1_ps(out_node->grad[i * N + j]);
                size_t k = 0;
                for (; k + 8 <= K; k += 8)
                {
                    __m256 bt_vals = _mm256_loadu_ps(&B_T[j * K + k]);
                    __m256 grad_a_vals = _mm256_loadu_ps(&a_node->grad[i * K + k]);

                    grad_a_vals = _mm256_fmadd_ps(grad_out_val, bt_vals, grad_a_vals);
                    _mm256_storeu_ps(&a_node->grad[i * K + k], grad_a_vals);
                }
                for (; k < K; k++)
                {
                    a_node->grad[i * K + k] += out_node->grad[i * N + j] * B_T[j * K + k];
                }
            }
        }
    };

    return out;
}
