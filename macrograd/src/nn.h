#ifndef nn_h
#define nn_h

#include "macrograd.h"

typedef enum { ACT_NONE, ACT_TANH, ACT_RELU, ACT_SOFTMAX } Activation;

typedef struct {
  Value **w;
  Value *b;
  int nin;
} Neuron;

static inline Neuron *new_neuron(int nin) {
  Neuron *n = (Neuron *)malloc(sizeof(Neuron));
  n->nin = nin;
  n->w = (Value **)malloc(sizeof(Value *) * nin);
  double scale = sqrt(2.0 / nin);
  for (int i = 0; i < nin; i++) {
    n->w[i] = new_value(((double)rand() / RAND_MAX * 2 - 1) * scale, "w");
  }
  n->b = new_value(0.0, "b");
  return n;
}

static inline Value *neuron_forward(Neuron *n, Value **x) {
  Value *out = n->b;
  for (int i = 0; i < n->nin; i++) {
    out = val_add(out, val_mul(n->w[i], x[i]));
  }
  return out;
}

typedef struct {
  Neuron **neurons;
  int nout;
  Activation act;
} Layer;

static inline Layer *new_layer(int nin, int nout, Activation act) {
  Layer *l = (Layer *)malloc(sizeof(Layer));
  l->nout = nout;
  l->act = act;
  l->neurons = (Neuron **)malloc(sizeof(Neuron *) * nout);
  for (int i = 0; i < nout; i++)
    l->neurons[i] = new_neuron(nin);
  return l;
}

static inline Value **softmax(Value **logits, int n) {
  Value *max_val = logits[0];
  for (int i = 1; i < n; i++) {
    if (logits[i]->data > max_val->data)
      max_val = logits[i];
  }

  Value **probs = (Value **)malloc(sizeof(Value *) * n);
  Value *sum = new_value(1e-15, "sum");
  Value *neg_one = new_value(-1.0, "const");
  Value *neg_max = val_mul(neg_one, max_val);

  for (int i = 0; i < n; i++) {
    Value *shifted = val_add(logits[i], neg_max);
    probs[i] = val_exp(shifted);
    sum = val_add(sum, probs[i]);
  }
  Value *inv_sum = val_log(sum);
  inv_sum = val_mul(neg_one, inv_sum);
  inv_sum = val_exp(inv_sum);

  for (int i = 0; i < n; i++) {
    probs[i] = val_mul(probs[i], inv_sum);
  }
  return probs;
}

static inline Value **layer_forward(Layer *l, Value **x) {
  Value **out = (Value **)malloc(sizeof(Value *) * l->nout);
  for (int i = 0; i < l->nout; i++)
    out[i] = neuron_forward(l->neurons[i], x);
  switch (l->act) {
  case ACT_TANH:
    for (int i = 0; i < l->nout; i++)
      out[i] = val_tanh(out[i]);
    break;
  case ACT_NONE:
    break;
  case ACT_RELU:
    for (int i = 0; i < l->nout; i++)
      out[i] = val_relu(out[i]);
    break;

  case ACT_SOFTMAX:
    Value **probs = softmax(out, l->nout);
    free(out);
    return probs;
    break;
  }

  return out;
}

typedef struct {
  Layer **layers;
  int nlayers;
} MLP;

static inline MLP *new_mlp(int nin, int *sizes, Activation *acts, int nlayers) {
  MLP *m = (MLP *)malloc(sizeof(MLP));
  m->nlayers = nlayers;
  m->layers = (Layer **)malloc(sizeof(Layer *) * nlayers);
  int in = nin;
  for (int i = 0; i < nlayers; i++) {
    m->layers[i] = new_layer(in, sizes[i], acts[i]);
    in = sizes[i];
  }
  return m;
}

static inline Value **mlp_forward(MLP *m, Value **x) {
  Value **current_in = x;
  for (int i = 0; i < m->nlayers; i++) {
    Value **next_in = layer_forward(m->layers[i], current_in);
    if (i > 0)
      free(current_in);
    current_in = next_in;
  }
  return current_in;
}

static inline Value *mse(Value **pred, double *target, int n) {
  Value *loss = new_value(0.0, "loss");
  for (int i = 0; i < n; i++) {
    Value *t = new_value(target[i], "target");
    Value *neg_one = new_value(-1.0, "const");
    Value *diff = val_add(pred[i], val_mul(neg_one, t));
    loss = val_add(loss, val_mul(diff, diff));
  }
  return loss;
}
static inline Value *cross_entropy(Value **pred, double *target, int n) {
  for (int i = 0; i < n; i++) {
    if (target[i] > 0.5) {
      Value *log_p = val_log(pred[i]);
      Value *neg_one = new_value(-1.0, "const");
      return val_mul(neg_one, log_p);
    }
  }
  return new_value(0.0, "zero_loss");
}
static inline void update_params(MLP *m, double lr) {
  for (int i = 0; i < m->nlayers; i++) {
    for (int j = 0; j < m->layers[i]->nout; j++) {
      Neuron *n = m->layers[i]->neurons[j];
      for (int k = 0; k < n->nin; k++)
        n->w[k]->data -= lr * n->w[k]->grad;
      n->b->data -= lr * n->b->grad;
    }
  }
}

#endif
