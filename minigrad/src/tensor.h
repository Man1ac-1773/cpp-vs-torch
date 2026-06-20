#pragma once
#include <cassert>
#include <functional>
#include <memory>
#include <vector>

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

    float& operator()(size_t r, size_t c)
    {
        return node->data[r * node->cols + c];
    }
    const float& operator()(size_t r, size_t c) const
    {
        return node->data[r * node->cols + c];
    }
};

inline Tensor operator+(const Tensor& a, const Tensor& b)
{
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
