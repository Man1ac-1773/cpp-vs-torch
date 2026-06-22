#include <iostream>
#include <fstream>
#include <sys/resource.h>
#include "./../macrograd/src/macrograd.h"

using namespace std;

int main(int argc, char* argv[]) {
    string mode = "performance-plugged";
    if (argc > 1) mode = argv[1];
    
    string filepath = "./data/c_pagefaults_" + mode + ".jsonl";
    ofstream json_out(filepath, ios::app);
    
    uint step = 10;
    for (size_t N = 10; N <= 2000; N += step) {
        struct rusage usage1, usage2;
        g_arena.top = 0; 
        
        Tensor* c_A1 = new_tensor(N, N);
        Tensor* c_B1 = new_tensor(N, N);
        
        g_arena.top = 0;
        getrusage(RUSAGE_SELF, &usage1);
        Tensor* c_A2 = new_tensor(N, N);
        Tensor* c_B2 = new_tensor(N, N);
        getrusage(RUSAGE_SELF, &usage2);
        long c_faults = usage2.ru_minflt - usage1.ru_minflt;

        json_out << "{\"N\": " << N << ", \"engine\": \"C (Bump Allocator)\", \"metric\": \"minor_page_faults\", \"value\": " << c_faults << "}\n";

        if (step == 10 && N >= 100) step = 100;
        if (step == 100 && N >= 1000) step = 1000;
    }
    
    json_out.close();
    return 0;
}
