#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include "../../minigrad/src/engine.h"
#include "../common/profiler.h"
#include "../common/rapl.h"

using namespace std;



static inline void load_mnist(const char* image_path, const char* label_path, Tensor& X, Tensor& Y) {
    FILE* f_img = fopen(image_path, "rb");
    if (f_img) {
        size_t read = fread(X.node->data.data(), sizeof(float), X.node->rows * X.node->cols, f_img);
        (void)read;
        fclose(f_img);
    } else {
        fprintf(stderr, "Failed to load MNIST images: %s\n", image_path);
    }

    FILE* f_lbl = fopen(label_path, "rb");
    if (f_lbl) {
        size_t read = fread(Y.node->data.data(), sizeof(float), Y.node->rows * Y.node->cols, f_lbl);
        (void)read;
        fclose(f_lbl);
    } else {
        fprintf(stderr, "Failed to load MNIST labels: %s\n", label_path);
    }
}

float compute_accuracy(Tensor& pred, Tensor& target) {
    uint correct = 0;
    uint batch_size = pred.node->rows;
    uint num_classes = pred.node->cols;
    
    for (uint i = 0; i < batch_size; i++) {
        uint best_pred = 0;
        float max_pred = pred.node->data[i * num_classes];
        uint best_target = 0;
        float max_target = target.node->data[i * num_classes];
        
        for (uint j = 1; j < num_classes; j++) {
            if (pred.node->data[i * num_classes + j] > max_pred) {
                max_pred = pred.node->data[i * num_classes + j];
                best_pred = j;
            }
            if (target.node->data[i * num_classes + j] > max_target) {
                max_target = target.node->data[i * num_classes + j];
                best_target = j;
            }
        }
        if (best_pred == best_target) correct++;
    }
    return (float)correct / batch_size;
}

int main(int argc, char* argv[]) {
    string mode = "performance-plugged";
    if (argc > 1) mode = argv[1];
    
    string filepath = "./data/cpp_mnist_" + mode + ".jsonl";
    ofstream json_out(filepath);

    uint TRAIN_SIZE = 60000;
    uint TEST_SIZE = 10000;
    uint INPUT_DIM = 784;
    uint HIDDEN1 = 64;
    uint HIDDEN2 = 32;
    uint OUTPUT_DIM = 10;
    uint EPOCHS = 10; 
    uint BATCH_SIZE = 128;
    float LR = 0.01f;

    cout << "Training MNIST 3-Layer MLP on C++ Engine..." << endl;

    MatmulBackend backends[] = {MATMUL_NAIVE, MATMUL_TILED, MATMUL_SIMD};
    string backend_names[] = {"naive", "tiled", "simd"};

    for (int b = 0; b < 3; b++) {
        g_matmul_backend = backends[b];
        string backend_name = backend_names[b];
        
        cout << "\n--- Backend: " << backend_name << " ---" << endl;

        Tensor X_train(TRAIN_SIZE, INPUT_DIM);
        Tensor Y_train(TRAIN_SIZE, OUTPUT_DIM);
        load_mnist("../../resources/data/train_images.bin", "../../resources/data/train_labels.bin", X_train, Y_train);

        Tensor X_test(TEST_SIZE, INPUT_DIM);
        Tensor Y_test(TEST_SIZE, OUTPUT_DIM);
        load_mnist("../../resources/data/test_images.bin", "../../resources/data/test_labels.bin", X_test, Y_test);

        MLP3 model(INPUT_DIM, HIDDEN1, HIDDEN2, OUTPUT_DIM);
        for (uint i = 0; i < INPUT_DIM * HIDDEN1; i++) model.fc1.weight.node->data[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
        for (uint i = 0; i < HIDDEN1 * HIDDEN2; i++) model.fc2.weight.node->data[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
        for (uint i = 0; i < HIDDEN2 * OUTPUT_DIM; i++) model.fc3.weight.node->data[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;

        SGD optimizer(model.parameters(), LR);

        Tensor X_batch(BATCH_SIZE, INPUT_DIM);
        Tensor Y_batch(BATCH_SIZE, OUTPUT_DIM);
        uint NUM_BATCHES = TRAIN_SIZE / BATCH_SIZE;

        double total_time = 0;
        double start_energy = get_rapl_energy_joules();

        for (uint epoch = 0; epoch < EPOCHS; epoch++) {
            double epoch_time = 0;
            float total_loss = 0.0f;

            for (uint b = 0; b < NUM_BATCHES; b++) {
                std::copy(X_train.node->data.begin() + b * BATCH_SIZE * INPUT_DIM, 
                          X_train.node->data.begin() + (b + 1) * BATCH_SIZE * INPUT_DIM, 
                          X_batch.node->data.begin());
                std::copy(Y_train.node->data.begin() + b * BATCH_SIZE * OUTPUT_DIM, 
                          Y_train.node->data.begin() + (b + 1) * BATCH_SIZE * OUTPUT_DIM, 
                          Y_batch.node->data.begin());

                double start_time = get_wall_time();

                Tensor pred = model.forward(X_batch);
                Tensor loss = cross_entropy_loss(pred, Y_batch);
                float loss_val = loss.node->data[0];

                loss.backward();
                optimizer.step();

                epoch_time += get_wall_time() - start_time;
                total_loss += loss_val;

                optimizer.zero_grad();
            }

            cout << "Epoch " << epoch << " | Avg Loss: " << (total_loss / NUM_BATCHES) << " | Compute Time: " << epoch_time << "s" << endl;
            total_time += epoch_time;
        }

        // Test Evaluation
        Tensor test_pred = model.forward(X_test);
        float test_acc = compute_accuracy(test_pred, Y_test);
        cout << "Test Accuracy: " << (test_acc * 100.0f) << "%" << endl;

        double end_energy = get_rapl_energy_joules();
        double energy_joules = end_energy - start_energy;

        cout << "Total Training Time (" << backend_name << "): " << total_time << "s | Energy: " << energy_joules << "J" << endl;
        
        json_out << "{\"task\": \"mnist_mlp\", \"engine\": \"C++\", \"backend\": \"" << backend_name 
                 << "\", \"total_time\": " << total_time << ", \"avg_epoch_time\": " << (total_time / EPOCHS)
                 << ", \"energy_joules\": " << energy_joules << ", \"test_acc\": " << test_acc << "}\n";
    }

    json_out.close();
    return 0;
}
