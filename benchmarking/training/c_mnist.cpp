#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include "../../macrograd/src/engine.h"
#include "../common/profiler.h"
#include "../common/rapl.h"

using namespace std;

static inline void load_mnist(const char* image_path, const char* label_path, Tensor* X, Tensor* Y) {
    FILE* f_img = fopen(image_path, "rb");
    if (f_img) {
        size_t read = fread(X->data, sizeof(float), X->shape[0] * X->shape[1], f_img);
        (void)read;
        fclose(f_img);
    } else {
        fprintf(stderr, "Failed to load MNIST images: %s\n", image_path);
    }

    FILE* f_lbl = fopen(label_path, "rb");
    if (f_lbl) {
        size_t read = fread(Y->data, sizeof(float), Y->shape[0] * Y->shape[1], f_lbl);
        (void)read;
        fclose(f_lbl);
    } else {
        fprintf(stderr, "Failed to load MNIST labels: %s\n", label_path);
    }
}

float compute_accuracy(Tensor* pred, Tensor* target) {
    uint correct = 0;
    uint batch_size = pred->shape[0];
    uint num_classes = pred->shape[1];
    
    for (uint i = 0; i < batch_size; i++) {
        uint best_pred = 0;
        float max_pred = pred->data[i * num_classes];
        uint best_target = 0;
        float max_target = target->data[i * num_classes];
        
        for (uint j = 1; j < num_classes; j++) {
            if (pred->data[i * num_classes + j] > max_pred) {
                max_pred = pred->data[i * num_classes + j];
                best_pred = j;
            }
            if (target->data[i * num_classes + j] > max_target) {
                max_target = target->data[i * num_classes + j];
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
    
    string filepath = "./data/c_mnist_" + mode + ".jsonl";
    ofstream json_out(filepath);

    uint TRAIN_SIZE = 60000;
    uint TEST_SIZE = 10000;
    uint INPUT_DIM = 784;
    uint HIDDEN1 = 64;
    uint HIDDEN2 = 32;
    uint OUTPUT_DIM = 10;
    uint EPOCHS = 5; // MNIST full batch takes a long time
    float LR = 0.5f;

    cout << "Training MNIST 3-Layer MLP on C Engine..." << endl;

    MatmulBackend backends[] = {MATMUL_NAIVE, MATMUL_TILED, MATMUL_SIMD};
    string backend_names[] = {"naive", "tiled", "simd"};

    for (int b = 0; b < 3; b++) {
        g_matmul_backend = backends[b];
        string backend_name = backend_names[b];
        
        cout << "\n--- Backend: " << backend_name << " ---" << endl;

        g_arena.top = 0;
        
        Tensor* X_train = new_tensor(TRAIN_SIZE, INPUT_DIM);
        Tensor* Y_train = new_tensor(TRAIN_SIZE, OUTPUT_DIM);
        load_mnist("../../resources/data/train_images.bin", "../../resources/data/train_labels.bin", X_train, Y_train);

        Tensor* X_test = new_tensor(TEST_SIZE, INPUT_DIM);
        Tensor* Y_test = new_tensor(TEST_SIZE, OUTPUT_DIM);
        load_mnist("../../resources/data/test_images.bin", "../../resources/data/test_labels.bin", X_test, Y_test);

        MLP3 model = mlp3_init(INPUT_DIM, HIDDEN1, HIDDEN2, OUTPUT_DIM);
        // Random init
        for (uint i = 0; i < INPUT_DIM * HIDDEN1; i++) model.fc1.weight->data[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
        for (uint i = 0; i < HIDDEN1 * HIDDEN2; i++) model.fc2.weight->data[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
        for (uint i = 0; i < HIDDEN2 * OUTPUT_DIM; i++) model.fc3.weight->data[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;

        Tensor* params[] = {model.fc1.weight, model.fc2.weight, model.fc3.weight};
        SGD opt = sgd_init(params, 3, LR);

        size_t arena_checkpoint = g_arena.top;

        double total_time = 0;
        double start_energy = get_rapl_energy_joules();

        for (uint epoch = 0; epoch < EPOCHS; epoch++) {
            double start_time = get_wall_time();

            Tensor* pred = mlp3_forward(&model, X_train);
            Tensor* loss = tensor_cross_entropy_loss(pred, Y_train);
            float loss_val = loss->data[0];

            backwardPass(loss);
            sgd_step(&opt);

            double epoch_time = get_wall_time() - start_time;
            total_time += epoch_time;

            float train_acc = compute_accuracy(pred, Y_train);

            cout << "Epoch " << epoch << " | Loss: " << loss_val << " | Train Acc: " << (train_acc * 100.0f) << "% | Time: " << epoch_time << "s" << endl;

            g_arena.top = arena_checkpoint;
        }

        // Test Evaluation
        Tensor* test_pred = mlp3_forward(&model, X_test);
        float test_acc = compute_accuracy(test_pred, Y_test);
        cout << "Test Accuracy: " << (test_acc * 100.0f) << "%" << endl;

        double end_energy = get_rapl_energy_joules();
        double energy_joules = end_energy - start_energy;

        cout << "Total Training Time (" << backend_name << "): " << total_time << "s | Energy: " << energy_joules << "J" << endl;
        
        json_out << "{\"task\": \"mnist_mlp\", \"engine\": \"C\", \"backend\": \"" << backend_name 
                 << "\", \"total_time\": " << total_time << ", \"avg_epoch_time\": " << (total_time / EPOCHS)
                 << ", \"energy_joules\": " << energy_joules << ", \"test_acc\": " << test_acc << "}\n";
    }

    json_out.close();
    return 0;
}
