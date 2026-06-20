#pragma once
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ostream>
#include <set>
#include <vector>
using namespace std;

struct Node
{ // the actual data container. For memory safety purposes
    double data;
    double grad;
    vector<shared_ptr<Node>> _prev;
    function<void()> _backward;
    string label;
    string op;
    int id;
    inline static int next_id = 1;
};

class Value
{
  public:
    shared_ptr<Node> node;

    Value(double _data)
    {
        node = make_shared<Node>();
        node->data = _data;
        node->id = node->next_id++;
        node->label = "";
        node->op = "";
        node->grad = 0.0;
    }

    void set_label(string _label) // name from value
    {
        node->label = _label;
    }
    void zero_grad()
    {
        node->grad = 0;
    }
    double get_grad()
    {
        return node->grad;
    }
    double get_data()
    {
        return node->data;
    }
    string get_op()
    {
        return node->op;
    }
    string get_label()
    {
        return node->label;
    }
    vector<shared_ptr<Node>>& get_prev()
    {
        return node->_prev;
    }
    void backward()
    {
        vector<shared_ptr<Node>> topo;
        set<Node*> visited; // raw pointer since shared_ptr comparison is weird
        function<void(shared_ptr<Node>)> build_topo;

        build_topo = [&](shared_ptr<Node> v)
        {
            if (visited.insert(v.get()).second)
            {
                for (auto& prev : v->_prev)
                {
                    build_topo(prev);
                }
                topo.push_back(v);
            }
        };

        build_topo(node); // node = this->node

        node->grad = 1.0;

        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        {
            if ((*it)->_backward)
                (*it)->_backward();
        } // reverse traverse
    }
    Value relu()
    {
        Value out(0);
        if (node->data > 0)
        {
            out.node->data = node->data;
        }
        out.node->op = "reLU";
        out.node->_prev = {node};
        out.node->_backward = [self = this->node, out_node = out.node]()
        { self->grad += (self->data > 0) * out_node->grad; };
        return out;
    }
    Value Tanh()
    {
        Value out(tanh(node->data));
        out.node->op = "tanh";
        out.node->_prev = {node};
        out.node->_backward = [self = this->node, out_node = out.node]()
        { self->grad += out_node->grad * (1 - pow(out_node->data, 2)); };
        return out;
    }
    Value sigmoid()
    {
        Value out(1 / (1 + exp(-node->data)));
        out.node->op = "sigm";
        out.node->_prev = {node};
        out.node->_backward = [self = this->node, out_node = out.node]()
        { self->grad += out_node->grad * (out_node->data * (1 - out_node->data)); };
        return out;
    }
    void operator-=(const double& d)
    {
        node->data -= d;
    }

    friend ostream& operator<<(ostream& os, Value& v)
    {
        os << "Value(data=" << v.node->data << ")";
        return os;
    }
};

inline Value operator+(const Value& a, const Value& b)
{
    Value out(a.node->data + b.node->data);

    out.node->_prev = {a.node, b.node};
    out.node->op = "+";

    out.node->_backward = [a_node = a.node, b_node = b.node, out_node = out.node]()
    {
        a_node->grad += out_node->grad;
        b_node->grad += out_node->grad;
    };

    return out;
}
inline Value operator*(const Value& a, const Value& b)
{
    Value out(a.node->data * b.node->data);

    out.node->_prev = {a.node, b.node};
    out.node->op = "*";

    out.node->_backward = [a_node = a.node, b_node = b.node, out_node = out.node]()
    {
        a_node->grad += b_node->data * out_node->grad;
        b_node->grad += a_node->data * out_node->grad;
    };

    return out;
}

inline Value operator^(const Value& a, const Value& b)
{
    // only value objects, not like karpathy
    // _prev vector is static type
    Value out(pow(a.node->data, b.node->data));
    out.node->_prev = {a.node, b.node};
    out.node->op = "^";
    out.node->_backward = [a_node = a.node, b_node = b.node, out_node = out.node]
    {
        a_node->grad +=
            b_node->data * pow(a_node->data, b_node->data - 1) * out_node->grad; // (d/dx (x^n)) = n * x^(n-1)
        b_node->grad += out_node->data * log(a_node->data) * out_node->grad;
        // (d/dn (x^n)) = x^n * ln(x);
    };

    return out;
}
inline Value operator^(const Value& a, const double& b)
{
    // only value objects, not like karpathy
    // _prev vector is static type
    Value out(pow(a.node->data, b));
    out.node->_prev = {a.node};
    out.node->op = "^";
    out.node->_backward = [a_node = a.node, b, out_node = out.node]
    {
        a_node->grad += b * pow(a_node->data, b - 1) * out_node->grad; // (d/dx (x^n)) = n * x^(n-1)
    };

    return out;
}
inline Value operator-(const Value& a) // unary negation
{
    Value neg(-1.0);
    Value out = a * neg;
    return out;
}
inline Value operator-(const Value& a, const Value& b) // binary subtraction
{
    Value out = a + (-b);
    return out;
}
inline Value operator/(const Value& a, const Value& b) // power + multiply wrapper
{

    Value out = a * (b ^ -1);
    return out;
}
