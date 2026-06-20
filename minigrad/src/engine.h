#pragma once
#include <cassert>
#include <cstdlib>

#include "graph.h"
#include "value.h"
using namespace std;

class Neuron
{
  public:
    uint nin;
    vector<Value> weights;
    Value bias;
    bool nonlin;
    Neuron(uint _nin, bool _nonlin = false) : bias(Value((((float) rand()) / RAND_MAX) * 2 - 1))
    {
        nin = _nin;
        weights.reserve(nin);
        for (uint i = 0; i < nin; i++)
        {
            // constant init for testing
            // weights.push_back(Value(2.0));
            weights.push_back(Value((((float) rand()) / RAND_MAX) * 2 - 1));
            weights[i].set_label("w"); // not needed, just there
        }
        bias.set_label("B"); // not needed
        nonlin = _nonlin;
    }
    Value operator()(const vector<Value>& x)
    {
        Value out(0.0);
        out = out + bias;
        for (uint i = 0; i < nin; i++)
        {
            out = out + weights[i] * x[i];
        }
        if (nonlin)
            out = out.Tanh();
        return out;
    }

    friend ostream& operator<<(ostream& os, Neuron neuron)
    {
        os << "Neuron weights : " << endl;
        ;
        for (uint i = 0; i < neuron.nin; i++)
        {
            os << neuron.weights[i] << endl;
        }
        return os;
    }
    vector<Value> params()
    {
        vector<Value> out;
        for (uint i = 0; i < nin; i++)
        {
            out.push_back(weights[i]);
        }
        out.push_back(bias);
        return out;
    }
    void zero_grad()
    {
        bias.zero_grad();
        for (auto& w : weights)
        {
            w.zero_grad();
        }
    }
};

class Layer
{
  public:
    uint nin;  // input number of each neuron in this layer
    uint nout; // number of neurons in this layer
    vector<Neuron> neurons;
    Layer(uint _n_prev, uint _n_this) : nin(_n_prev), nout(_n_this)
    {
        neurons.reserve(nout);
        for (uint i = 0; i < nout; i++)
        {
            neurons.push_back(Neuron(nin));
        }
    }
    vector<Value> operator()(const vector<Value>& x)
    {
        vector<Value> out;
        out.reserve(nout);
        for (uint i = 0; i < nout; i++)
        {
            out.push_back(neurons[i](x));
        }
        return out;
    }
    vector<Value> params()
    {
        vector<Value> out;
        for (int i = 0; i < nout; i++)
        {
            for (auto& a : neurons[i].params())
            {
                out.push_back(a);
            }
        }
        return out;
    }
    void zero_grad()
    {
        for (auto& neuron : neurons)
        {
            neuron.zero_grad();
        }
    }
};

class NeuralNet
{
  public:
    vector<uint> sizes;
    vector<Layer> layers;
    NeuralNet(vector<uint> _sizes) : sizes(_sizes)
    {
        // sizes[0] is inputs to neural net.
        // thereafter, sizes[i] defines number of neurons in that layer
        layers.reserve(sizes.size() - 1);
        for (int i = 1; i < sizes.size(); i++)
        {
            layers.push_back(Layer(sizes[i - 1], sizes[i]));
        }
    }
    // feed forward for single sample
    vector<Value> feedforward(vector<Value> x)
    {
        for (uint i = 0; i < layers.size(); i++)
        {
            x = layers[i](x);
        }
        return x;
    }
    // feedforward for multiple samples at once
    vector<vector<Value>> feedforward_batch(vector<vector<Value>> x)
    {
        for (uint i = 0; i < x.size(); i++)
        {

            for (uint j = 0; j < layers.size(); j++)
            {
                x[i] = layers[j](x[i]);
            }
        }
        return x;
    }
    vector<Value> params()
    {
        vector<Value> out;
        for (int i = 0; i < layers.size(); i++)
        {
            for (auto& a : layers[i].params())
            {
                out.push_back(a);
            }
        }
        return out;
    }
    void zero_grad()
    {
        for (auto& layer : layers)
        {
            layer.zero_grad();
        }
    }
    // single sample
    Value mse_loss(const vector<Value>& out, const vector<Value>& ans)
    {
        Value loss(0.0);
        assert(out.size() == ans.size());   // ans and out must be same
        assert(out.size() == sizes.back()); // ans and last layer output size must be same
        for (uint i = 0; i < ans.size(); i++)
        {
            loss = loss + ((out[i] - ans[i]) ^ 2);
        }
        return loss;
    }
    // multiple samples at once
    Value mse_loss_batch(const vector<vector<Value>>& out, const vector<vector<Value>>& ans)
    {
        Value loss(0.0);
        assert(out.size() == ans.size());
        // other asserts handled by per-sample
        for (uint i = 0; i < out.size(); i++)
        {
            loss = loss + mse_loss(out[i], ans[i]); // use previous, per-sample method
        }
        return loss;
    }
    Value CrossEntropy_loss(const vector<Value>& out, const vector<Value>& ans)
    {
        // TODO
        Value loss(0.0);
        return loss;
    }
};
