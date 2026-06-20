#pragma once
#include <functional>
#include <memory>
#include <vector>

using namespace std;

// inherit from enable_shared_from_this so we can pass 'this' into our lambda captures safely
class Tensor : public enable_shared_from_this<Tensor>
{
  public:
    size_t rows, cols;
    vector<float> data;
    vector<float> grad;

    vector<shared_ptr<Tensor>> _prev;
    function<void()> _backward;

    Tensor(size_t _rows, size_t _cols) : rows(_rows), cols(_cols)
    {
        data.resize(rows * cols, 0.0f);
        grad.resize(rows * cols, 0.0f);
        _backward = []() {}; // empty lambda
    }
};
