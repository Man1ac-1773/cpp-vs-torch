#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include "../../minigrad/src/engine.h"
#include "../common/profiler.h"
#include "../common/rapl.h"

using namespace std;

// define the global matmul backend for minigrad


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

    MatmulBackend backends[] = {MATMUL_NAIVE, MATMUL_TILED, MATMUL_SIMD};
    string backend_names[] = {"naive", "tiled", "simd"};

    for (int b = 0; b < 3; b++) {
        g_matmul_backend = backends[b];
        string backend_name = backend_names[b];
        
        cout << "\n--- Backend: " << backend_name << " ---" << endl;

        Tensor X(BATCH_SIZE, INPUT_DIM);
        Tensor Y(BATCH_SIZE, OUTPUT_DIM);
        for (uint i = 0; i < BATCH_SIZE * INPUT_DIM; i++) X.node->data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        for (uint i = 0; i < BATCH_SIZE * OUTPUT_DIM; i++) Y.node->data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

        MLP model(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM);
        for (uint i = 0; i < INPUT_DIM * HIDDEN_DIM; i++) model.fc1.weight.node->data[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
        for (uint i = 0; i < HIDDEN_DIM * OUTPUT_DIM; i++) model.fc2.weight.node->data[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;

        SGD optimizer(model.parameters(), LR);

        double total_time = 0;
        double start_energy = get_rapl_energy_joules();

        for (uint epoch = 0; epoch < EPOCHS; epoch++) {
            double start_time = get_wall_time();

            Tensor pred = model.forward(X);
            Tensor loss = mse_loss(pred, Y);
            float loss_val = loss.node->data[0];

            loss.backward();
            optimizer.step();

            double epoch_time = get_wall_time() - start_time;
            total_time += epoch_time;

            if (epoch % 10 == 0) {
                cout << "Epoch " << epoch << " | Loss: " << loss_val << " | Time: " << epoch_time << "s" << endl;
            }
            
            // re-allocate computation graph nodes to fake zeroing grad
            // actually, minigrads c++ relies on raii and scope to destroy the graph
            // but i created pred, loss inside the loop, so they will be destroyed properly
            // however, model weights accumulate gradients i must zero them out.
            for (auto p : model.parameters()) {
                std::fill(p->node->grad.begin(), p->node->grad.end(), 0.0f);
            }
        }

        double end_energy = get_rapl_energy_joules();
        double energy_joules = end_energy - start_energy;

        cout << "Total Training Time (" << backend_name << "): " << total_time << "s | Energy: " << energy_joules << "J" << endl;
        
        json_out << "{\"task\": \"dummy_mlp\", \"engine\": \"C++\", \"backend\": \"" << backend_name 
                 << "\", \"total_time\": " << total_time << ", \"avg_epoch_time\": " << (total_time / EPOCHS)
                 << ", \"energy_joules\": " << energy_joules << "}\n";
    }

    json_out.close();
    return 0;
}
