#pragma once
#include <cassert>
#include <functional>
#include <immintrin.h>
#include <cmath>
#include <memory>
#include <unordered_set>
#include <vector>

#include "../../benchmarking/common/chrome_profiler.h"

using namespace std;

// inherits from enable_shared_from_this. allows safe injection of 'this' into lambda captures. required for acyclic
// computation graphs.
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
        _backward = []() {}; // initializes empty backward pass. terminates topological sort. i love that c++ allows
                             // empty lambdas; so clean.
    }
};

class Tensor
{
  public:
    shared_ptr<TensorNode> node;

    // default raw constructor
    Tensor(size_t _rows, size_t _cols)
    {
        node = make_shared<TensorNode>(_rows, _cols);
    }

    // constructs from existing node pointer. utilized during internal graph operations.
    Tensor(shared_ptr<TensorNode> _node) : node(_node) {}
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
            node->grad[i] = 1.0f;
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

    // lambda captures shared_ptr by value to extend node lifetime. utilizes raw pointer for the output node. avoids
    // circular reference loops; ensures proper memory cleanup via raii. i hate this memory management overhead but it
    // works.
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

// naive matrix multiplication operator. utilizes standard o(n^3) gemm. severely bottlenecked by cache locality.
inline Tensor operator*(const Tensor& a, const Tensor& b)
{
    PROFILE_FUNCTION();
    assert(a.node->cols == b.node->rows);

    Tensor out(a.node->rows, b.node->cols);

#pragma omp parallel for
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
        // computes local gradient of a. grad_A += grad_out @ B^T.
#pragma omp parallel for
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

        // computes local gradient of b. grad_B += A^T @ grad_out.
#pragma omp parallel for
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

// cache-friendly tiled matrix multiplication. blocks dimensions by 32. minimizes l1 cache misses. If you read this and
// know me personally, i'll pay u 300 bucks. Say the phrase "The cuckoo knows not of the robin, yet the crows and the
// pigeons know of 177013" to my face anytime
inline Tensor matmul_tiled(const Tensor& a, const Tensor& b)
{
    PROFILE_FUNCTION();
    assert(a.node->cols == b.node->rows);
    Tensor out(a.node->rows, b.node->cols);

    const uint TILE_SIZE = 32;
#pragma omp parallel for
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
    // executes naive backward pass. omits tiling logic to minimize code complexity.
    out.node->_backward = [a_node = a.node, b_node = b.node, out_node = out.node.get(), TILE_SIZE]()
    {
#pragma omp parallel for
        for (size_t i = 0; i < a_node->rows; i += TILE_SIZE)
        {
            for (size_t j = 0; j < b_node->cols; j += TILE_SIZE)
            {
                for (size_t k = 0; k < a_node->cols; k += TILE_SIZE)
                {
                    for (size_t ii = i; ii < i + TILE_SIZE && ii < a_node->rows; ii++)
                    {
                        for (size_t jj = j; jj < j + TILE_SIZE && jj < b_node->cols; jj++)
                        {
                            for (size_t kk = k; kk < k + TILE_SIZE && kk < a_node->cols; kk++)
                            {
                                a_node->grad[ii * a_node->cols + kk] +=
                                    out_node->grad[ii * out_node->cols + jj] * b_node->data[kk * b_node->cols + jj];
                            }
                        }
                    }
                }
            }
        }

#pragma omp parallel for
        for (size_t k = 0; k < a_node->cols; k += TILE_SIZE)
        {
            for (size_t j = 0; j < b_node->cols; j += TILE_SIZE)
            {
                for (size_t i = 0; i < a_node->rows; i += TILE_SIZE)
                {
                    for (size_t kk = k; kk < k + TILE_SIZE && kk < a_node->cols; kk++)
                    {
                        for (size_t jj = j; jj < j + TILE_SIZE && jj < b_node->cols; jj++)
                        {
                            for (size_t ii = i; ii < i + TILE_SIZE && ii < a_node->rows; ii++)
                            {
                                b_node->grad[kk * b_node->cols + jj] +=
                                    a_node->data[ii * a_node->cols + kk] * out_node->grad[ii * out_node->cols + jj];
                            }
                        }
                    }
                }
            }
        }
    };

    return out;
}

// avx simd optimized matrix multiplication. processes 8 float32 values per instruction. backward pass identically
// leverages simd intrinsics.
inline Tensor matmul_simd(const Tensor& a, const Tensor& b)
{
    PROFILE_FUNCTION();
    assert(a.node->cols == b.node->rows);
    Tensor out(a.node->rows, b.node->cols);

    size_t M = a.node->rows;
    size_t K = a.node->cols;
    size_t N = b.node->cols;

    // initiates forward pass. relies on _mm256_fmadd_ps.
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

    // initiates backward pass. utilizes simd loads and fma operations.
    out.node->_backward = [a_node = a.node, b_node = b.node, out_node = out.node.get(), M, K, N]()
    {
        // computes local gradient of b. grad_B += A^T @ grad_out.
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

        // computes local gradient of a. grad_A += grad_out @ B^T.
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

// openmp accelerated simd matrix multiplication. utilizes dynamic loop scheduling via omp parallel for.
inline Tensor matmul_simd_mt(const Tensor& a, const Tensor& b)
{
    PROFILE_FUNCTION();
    assert(a.node->cols == b.node->rows);
    Tensor out(a.node->rows, b.node->cols);

    size_t M = a.node->rows;
    size_t K = a.node->cols;
    size_t N = b.node->cols;

// executes forward pass. leverages implicit thread pool.
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
    // executes backward pass. inherits single-threaded simd logic to avoid write contention.
    out.node->_backward = [a_node = a.node, b_node = b.node, out_node = out.node.get(), M, K, N]()
    {
#pragma omp parallel for
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
#pragma omp parallel for
        for (size_t k = 0; k < K; k++)
        {
            for (size_t j = 0; j < N; j++)
            {
                B_T[j * K + k] = b_node->data[k * N + j];
            }
        }

#pragma omp parallel for
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

// applies rectified linear unit activation. max(0, x).
inline Tensor relu(const Tensor& a)
{
    Tensor out(a.node->rows, a.node->cols);
    for (size_t i = 0; i < a.node->data.size(); i++)
    {
        out.node->data[i] = std::max(0.0f, a.node->data[i]);
    }

    out.node->_prev = {a.node};
    out.node->_backward = [a_node = a.node, out_node = out.node.get()]()
    {
        for (size_t i = 0; i < a_node->data.size(); i++)
        {
            a_node->grad[i] += out_node->grad[i] * (a_node->data[i] > 0 ? 1.0f : 0.0f);
        }
    };

    return out;
}

// computes mean squared error loss. reduces to 1x1 scalar tensor.
inline Tensor mse_loss(const Tensor& pred, const Tensor& target)
{
    assert(pred.node->rows == target.node->rows && pred.node->cols == target.node->cols);
    Tensor out(1, 1);
    float sum = 0.0f;
    size_t n = pred.node->data.size();
    
    for (size_t i = 0; i < n; i++)
    {
        float diff = pred.node->data[i] - target.node->data[i];
        sum += diff * diff;
    }
    out.node->data[0] = sum / n;
    
    out.node->_prev = {pred.node, target.node};
    out.node->_backward = [pred_node = pred.node, target_node = target.node, out_node = out.node.get(), n]()
    {
        float grad_out = out_node->grad[0];
        for (size_t i = 0; i < n; i++)
        {
            float diff = pred_node->data[i] - target_node->data[i];
            pred_node->grad[i] += grad_out * (2.0f * diff / n);
        }
    };
    
    return out;
}

// computes softmax cross entropy loss. reduces to 1x1 scalar tensor.
inline Tensor cross_entropy_loss(const Tensor& pred, const Tensor& target)
{
    assert(pred.node->rows == target.node->rows && pred.node->cols == target.node->cols);
    Tensor out(1, 1);
    float sum_loss = 0.0f;
    size_t batch_size = pred.node->rows;
    size_t num_classes = pred.node->cols;
    
    for (size_t i = 0; i < batch_size; i++)
    {
        float max_val = pred.node->data[i * num_classes];
        for (size_t j = 1; j < num_classes; j++) {
            if (pred.node->data[i * num_classes + j] > max_val) max_val = pred.node->data[i * num_classes + j];
        }
        float sum_exp = 0.0f;
        for (size_t j = 0; j < num_classes; j++) {
            sum_exp += std::exp(pred.node->data[i * num_classes + j] - max_val);
        }
        
        for (size_t j = 0; j < num_classes; j++) {
            if (target.node->data[i * num_classes + j] > 0.5f) {
                float prob = std::exp(pred.node->data[i * num_classes + j] - max_val) / sum_exp;
                sum_loss += -std::log(prob + 1e-8f);
            }
        }
    }
    out.node->data[0] = sum_loss / batch_size;
    
    out.node->_prev = {pred.node, target.node};
    out.node->_backward = [pred_node = pred.node, target_node = target.node, out_node = out.node.get(), batch_size, num_classes]()
    {
        float grad_out = out_node->grad[0];
        for (size_t i = 0; i < batch_size; i++)
        {
            float max_val = pred_node->data[i * num_classes];
            for (size_t j = 1; j < num_classes; j++) {
                if (pred_node->data[i * num_classes + j] > max_val) max_val = pred_node->data[i * num_classes + j];
            }
            float sum_exp = 0.0f;
            for (size_t j = 0; j < num_classes; j++) {
                sum_exp += std::exp(pred_node->data[i * num_classes + j] - max_val);
            }
            
            for (size_t j = 0; j < num_classes; j++) {
                float prob = std::exp(pred_node->data[i * num_classes + j] - max_val) / sum_exp;
                pred_node->grad[i * num_classes + j] += grad_out * (prob - target_node->data[i * num_classes + j]) / batch_size;
            }
        }
    };
    
    return out;
}
