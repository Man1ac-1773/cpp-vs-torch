#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include "../../macrograd/src/macrograd.h"
#include "../common/profiler.h"
#include "../common/rapl.h"

using namespace std;

void sgd_step(Tensor* param, float lr) {
    for (uint i = 0; i < param->shape[0] * param->shape[1]; i++) {
        param->data[i] -= lr * param->grad[i];
    }
}

int main(int argc, char* argv[]) {
    string mode = "performance-plugged";
    if (argc > 1) mode = argv[1];
    
    string filepath = "./data/c_train_" + mode + ".jsonl";
    ofstream json_out(filepath);

    uint BATCH_SIZE = 1024;
    uint INPUT_DIM = 128;
    uint HIDDEN_DIM = 256;
    uint OUTPUT_DIM = 64;
    uint EPOCHS = 100;
    float LR = 0.01f;

    cout << "Training 2-Layer MLP on C Engine (Batch=" << BATCH_SIZE << ")..." << endl;

    typedef Tensor* (*MatmulFunc)(Tensor*, Tensor*);
    MatmulFunc backends[] = {tensor_matmul_naive, tensor_matmul_tiled, tensor_matmul_simd_mt};
    string backend_names[] = {"naive", "tiled", "simd"};

    for (int b = 0; b < 3; b++) {
        MatmulFunc matmul = backends[b];
        string backend_name = backend_names[b];
        
        cout << "\n--- Backend: " << backend_name << " ---" << endl;

        g_arena.top = 0; // init
        
        Tensor* X = new_tensor(BATCH_SIZE, INPUT_DIM);
        Tensor* Y = new_tensor(BATCH_SIZE, OUTPUT_DIM);
        for (uint i = 0; i < BATCH_SIZE * INPUT_DIM; i++) X->data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        for (uint i = 0; i < BATCH_SIZE * OUTPUT_DIM; i++) Y->data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

        Tensor* W1 = new_tensor(INPUT_DIM, HIDDEN_DIM);
        Tensor* W2 = new_tensor(HIDDEN_DIM, OUTPUT_DIM);
        for (uint i = 0; i < INPUT_DIM * HIDDEN_DIM; i++) W1->data[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
        for (uint i = 0; i < HIDDEN_DIM * OUTPUT_DIM; i++) W2->data[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;

        size_t arena_checkpoint = g_arena.top;

        double total_time = 0;
        double start_energy = get_rapl_energy_joules();

        for (uint epoch = 0; epoch < EPOCHS; epoch++) {
            double start_time = get_wall_time();

            Tensor* h1 = matmul(X, W1);
            Tensor* a1 = tensor_relu(h1);
            Tensor* pred = matmul(a1, W2);

            Tensor* loss = tensor_mse_loss(pred, Y);
            float loss_val = loss->data[0];

            backwardPass(loss);

            sgd_step(W1, LR);
            sgd_step(W2, LR);

            double epoch_time = get_wall_time() - start_time;
            total_time += epoch_time;

            if (epoch % 10 == 0) {
                cout << "Epoch " << epoch << " | Loss: " << loss_val << " | Time: " << epoch_time << "s" << endl;
            }

            g_arena.top = arena_checkpoint;
        }

        double end_energy = get_rapl_energy_joules();
        double energy_joules = end_energy - start_energy;

        cout << "Total Training Time (" << backend_name << "): " << total_time << "s | Energy: " << energy_joules << "J" << endl;
        
        json_out << "{\"task\": \"dummy_mlp\", \"engine\": \"C\", \"backend\": \"" << backend_name 
                 << "\", \"total_time\": " << total_time << ", \"avg_epoch_time\": " << (total_time / EPOCHS)
                 << ", \"energy_joules\": " << energy_joules << "}\n";
    }

    json_out.close();
    return 0;
}
