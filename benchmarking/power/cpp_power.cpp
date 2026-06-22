#include <iostream>
#include <fstream>
#include <cmath>
#include "./../minigrad/src/engine.h"
#include "perf_profiler.h"
#include "profiler.h"

using namespace std;

// Wrapper for Minigrad (C++ Engine)
double power_bench_cpp(uint N) {
    Tensor A(N, N);
    Tensor B(N, N);
    for(uint i=0; i<N*N; i++) { A.node->data[i] = 1.0f; B.node->data[i] = 1.0f; }

    uint iterations = (N < 100) ? 1000 : ((N < 500) ? 50 : 5);

    double start_energy = get_rapl_energy_joules();
    for (uint i = 0; i < iterations; i++) {
        Tensor C = matmul_simd_mt(A, B);
    }
    double end_energy = get_rapl_energy_joules();
    
    return (end_energy - start_energy) / iterations;
}

void run_mnist_power_bench(ofstream& json_out) {
    cout << "Running Power Benchmark for MNIST (1 Epoch)..." << endl;
    
    // Load MNIST
    uint INPUT_DIM = 784;
    uint HIDDEN_DIM = 256;
    uint OUTPUT_DIM = 10;
    uint BATCH_SIZE = 60000;
    
    Tensor X(BATCH_SIZE, INPUT_DIM);
    Tensor Y(BATCH_SIZE, OUTPUT_DIM);
    load_mnist("../resources/data/train_images.bin", "../resources/data/train_labels.bin", X, Y);
    
    MLP model(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM);
    for(size_t i=0; i<INPUT_DIM*HIDDEN_DIM; i++) model.fc1.weight.node->data[i] = 0.01f;
    for(size_t i=0; i<HIDDEN_DIM*OUTPUT_DIM; i++) model.fc2.weight.node->data[i] = -0.01f;
    SGD optimizer(model.parameters(), 0.01f);
    
    double start_energy = get_rapl_energy_joules();
    for (int epoch = 0; epoch < 5; epoch++) {
        Tensor pred = model.forward(X);
        Tensor loss = mse_loss(pred, Y);
        loss.backward();
        optimizer.step();
    }
    double end_energy = get_rapl_energy_joules();
    
    double joules = (end_energy - start_energy) / 5.0;
    cout << "Minigrad (C++) MNIST Energy: " << joules << " Joules" << endl;
    json_out << "{\"task\": \"mnist_epoch\", \"engine\": \"minigrad\", \"energy_joules\": " << joules << "}\n";
}

int main(int argc, char* argv[]) {
    string mode = "performance_plugged";
    if (argc > 1) mode = argv[1];
    
    string filepath = "./data/power/cpp_" + mode + ".jsonl";
    ofstream json_out(filepath, std::ios_base::app);

    cout << "Sweeping Power metrics for C++ Engine (" << mode << ") (N=10 to N=1000)..." << endl;
    
    uint step = 10;
    for (size_t N = 10; N <= 1000; N += step) {
        double cpp_energy = power_bench_cpp(N);
        json_out << "{\"task\": \"matmul\", \"N\": " << N << ", \"engine\": \"minigrad\", \"energy_joules\": " << cpp_energy << "}\n";

        if (step == 10 && N >= 100) step = 100;
    }
    
    run_mnist_power_bench(json_out);
    
    json_out.close();
    cout << "Done! Results saved to " << filepath << endl;
    return 0;
}
