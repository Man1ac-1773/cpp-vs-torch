#include <iostream>
#include <fstream>
#include <cmath>
#include "./../macrograd/src/engine.h"
#include "perf_profiler.h"
#include "profiler.h"

using namespace std;

// Wrapper for Macrograd (C Engine)
double power_bench_c(uint N) {
    size_t checkpoint = g_arena.top;
    Tensor* A = new_tensor(N, N);
    Tensor* B = new_tensor(N, N);
    for(uint i=0; i<N*N; i++) { A->data[i] = 1.0f; B->data[i] = 1.0f; }

    uint iterations = (N < 100) ? 1000 : ((N < 500) ? 50 : 5);

    double start_energy = get_rapl_energy_joules();
    for (uint i = 0; i < iterations; i++) {
        Tensor* C = tensor_matmul_simd_mt(A, B);
        (void)C;
    }
    double end_energy = get_rapl_energy_joules();
    
    g_arena.top = checkpoint;
    return (end_energy - start_energy) / iterations;
}

void run_mnist_power_bench(ofstream& json_out) {
    cout << "Running Power Benchmark for MNIST (1 Epoch)..." << endl;
    
    // Load MNIST
    uint INPUT_DIM = 784;
    uint HIDDEN_DIM = 256;
    uint OUTPUT_DIM = 10;
    uint BATCH_SIZE = 60000;
    
    size_t checkpoint = g_arena.top;
    Tensor* X = new_tensor(BATCH_SIZE, INPUT_DIM);
    Tensor* Y = new_tensor(BATCH_SIZE, OUTPUT_DIM);
    load_mnist("../resources/data/train_images.bin", "../resources/data/train_labels.bin", X, Y);
    
    MLP model = mlp_init(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM);
    for(uint i=0; i<INPUT_DIM*HIDDEN_DIM; i++) model.fc1.weight->data[i] = 0.01f;
    for(uint i=0; i<HIDDEN_DIM*OUTPUT_DIM; i++) model.fc2.weight->data[i] = -0.01f;
    Tensor* params[] = {model.fc1.weight, model.fc2.weight};
    SGD optimizer = sgd_init(params, 2, 0.01f);
    
    double start_energy = get_rapl_energy_joules();
    for (int epoch = 0; epoch < 5; epoch++) {
        Tensor* pred = mlp_forward(&model, X);
        Tensor* loss = tensor_mse_loss(pred, Y);
        backwardPass(loss);
        sgd_step(&optimizer);
    }
    double end_energy = get_rapl_energy_joules();
    
    double joules = (end_energy - start_energy) / 5.0;
    cout << "Macrograd (C) MNIST Energy: " << joules << " Joules" << endl;
    json_out << "{\"task\": \"mnist_epoch\", \"engine\": \"macrograd\", \"energy_joules\": " << joules << "}\n";
    
    g_arena.top = checkpoint;
}

int main(int argc, char* argv[]) {
    string mode = "performance_plugged";
    if (argc > 1) mode = argv[1];
    
    string filepath = "./data/power/c_" + mode + ".jsonl";
    ofstream json_out(filepath, std::ios_base::app);

    cout << "Sweeping Power metrics for C Engine (" << mode << ") (N=10 to N=1000)..." << endl;
    
    uint step = 10;
    for (size_t N = 10; N <= 1000; N += step) {
        double c_energy = power_bench_c(N);
        json_out << "{\"task\": \"matmul\", \"N\": " << N << ", \"engine\": \"macrograd\", \"energy_joules\": " << c_energy << "}\n";

        if (step == 10 && N >= 100) step = 100;
    }
    
    run_mnist_power_bench(json_out);
    
    json_out.close();
    cout << "Done! Results saved to " << filepath << endl;
    return 0;
}
