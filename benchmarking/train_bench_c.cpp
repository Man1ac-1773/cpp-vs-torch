#include <iostream>
#include <fstream>
#include <cmath>
#include "./../macrograd/src/macrograd.h"
#include "profiler.h"

using namespace std;

void sgd_step(Tensor* param, float lr) {
    for (uint i = 0; i < param->shape[0] * param->shape[1]; i++) {
        param->data[i] -= lr * param->grad[i];
    }
}

int main() {
    ofstream json_out("./c-data/train_results.jsonl");

    uint BATCH_SIZE = 1024;
    uint INPUT_DIM = 128;
    uint HIDDEN_DIM = 256;
    uint OUTPUT_DIM = 64;
    uint EPOCHS = 100;
    float LR = 0.01f;

    cout << "Training 2-Layer MLP on C Engine (Batch=" << BATCH_SIZE << ")..." << endl;

    g_arena.top = 0; // init
    
    // Allocate Inputs and Targets
    Tensor* X = new_tensor(BATCH_SIZE, INPUT_DIM);
    Tensor* Y = new_tensor(BATCH_SIZE, OUTPUT_DIM);
    for (uint i = 0; i < BATCH_SIZE * INPUT_DIM; i++) X->data[i] = 0.5f;
    for (uint i = 0; i < BATCH_SIZE * OUTPUT_DIM; i++) Y->data[i] = 0.1f;

    // Allocate Weights
    Tensor* W1 = new_tensor(INPUT_DIM, HIDDEN_DIM);
    Tensor* W2 = new_tensor(HIDDEN_DIM, OUTPUT_DIM);
    for (uint i = 0; i < INPUT_DIM * HIDDEN_DIM; i++) W1->data[i] = 0.01f;
    for (uint i = 0; i < HIDDEN_DIM * OUTPUT_DIM; i++) W2->data[i] = -0.01f;

    // CHECKPOINT THE ARENA
    // This allows us to keep X, Y, W1, W2 in memory, but discard all 
    // intermediate activations (forward pass artifacts) at the end of each epoch!
    size_t arena_checkpoint = g_arena.top;

    double total_time = 0;

    for (uint epoch = 0; epoch < EPOCHS; epoch++) {
        double start_time = get_wall_time();

        // 1. Forward Pass (uses SIMD MT)
        Tensor* h1 = tensor_matmul_simd_mt(X, W1);
        Tensor* a1 = tensor_relu(h1);
        Tensor* pred = tensor_matmul_simd_mt(a1, W2);

        // 2. Loss
        Tensor* loss = tensor_mse_loss(pred, Y);
        float loss_val = loss->data[0];

        // 3. Backward Pass
        backwardPass(loss);

        // 4. Optimizer Step
        sgd_step(W1, LR);
        sgd_step(W2, LR);

        double epoch_time = get_wall_time() - start_time;
        total_time += epoch_time;

        if (epoch % 10 == 0) {
            cout << "Epoch " << epoch << " | Loss: " << loss_val << " | Time: " << epoch_time << "s" << endl;
        }

        json_out << "{\"engine\": \"C (macrograd)\", \"epoch\": " << epoch << ", \"loss\": " << loss_val << ", \"time\": " << epoch_time << "}\n";

        // 5. Memory Management: Pop Arena back to Checkpoint!
        // This is the genius of the bump allocator for ML training loops.
        // We instantly free `h1`, `a1`, `pred`, and `loss` in 0(1) time without Page Faults.
        g_arena.top = arena_checkpoint;
    }

    cout << "Total Training Time (C Engine): " << total_time << "s" << endl;
    json_out.close();
    return 0;
}
