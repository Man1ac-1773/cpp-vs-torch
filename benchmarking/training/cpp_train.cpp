#include <iostream>
#include <fstream>
#include <cmath>
#include "./../minigrad/src/tensor.h"
#include "profiler.h"

using namespace std;

void sgd_step(Tensor& param, float lr) {
    for (size_t i = 0; i < param.node->data.size(); i++) {
        param.node->data[i] -= lr * param.node->grad[i];
    }
}

int main(int argc, char* argv[]) {
    string mode = "performance-plugged";
    if (argc > 1) mode = argv[1];
    
    string filepath = "./data/cpp_train_" + mode + ".jsonl";
    ofstream json_out(filepath);

    uint BATCH_SIZE = 1024;
    uint INPUT_DIM = 128;
    uint HIDDEN_DIM = 256;
    uint OUTPUT_DIM = 64;
    uint EPOCHS = 100;
    float LR = 0.01f;

    cout << "Training 2-Layer MLP on C++ Engine (Batch=" << BATCH_SIZE << ")..." << endl;

    Tensor X(BATCH_SIZE, INPUT_DIM);
    Tensor Y(BATCH_SIZE, OUTPUT_DIM);
    for (size_t i = 0; i < BATCH_SIZE * INPUT_DIM; i++) X.node->data[i] = 0.5f;
    for (size_t i = 0; i < BATCH_SIZE * OUTPUT_DIM; i++) Y.node->data[i] = 0.1f;

    Tensor W1(INPUT_DIM, HIDDEN_DIM);
    Tensor W2(HIDDEN_DIM, OUTPUT_DIM);
    for (size_t i = 0; i < INPUT_DIM * HIDDEN_DIM; i++) W1.node->data[i] = 0.01f;
    for (size_t i = 0; i < HIDDEN_DIM * OUTPUT_DIM; i++) W2.node->data[i] = -0.01f;

    double total_time = 0;

    for (uint epoch = 0; epoch < EPOCHS; epoch++) {
        double start_time = get_wall_time();

        // 1. Forward Pass
        Tensor h1 = matmul_simd_mt(X, W1);
        Tensor a1 = relu(h1);
        Tensor pred = matmul_simd_mt(a1, W2);

        // 2. Loss
        Tensor loss = mse_loss(pred, Y);
        float loss_val = loss.node->data[0];

        // 3. Backward Pass
        loss.backward();

        // 4. Optimizer Step
        sgd_step(W1, LR);
        sgd_step(W2, LR);

        double epoch_time = get_wall_time() - start_time;
        total_time += epoch_time;

        if (epoch % 10 == 0) {
            cout << "Epoch " << epoch << " | Loss: " << loss_val << " | Time: " << epoch_time << "s" << endl;
        }

        json_out << "{\"engine\": \"C++ (minigrad)\", \"epoch\": " << epoch << ", \"loss\": " << loss_val << ", \"time\": " << epoch_time << "}\n";
    }

    cout << "Total Training Time (C++ Engine): " << total_time << "s" << endl;
    json_out.close();
    return 0;
}
