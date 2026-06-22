#include <iostream>
#include <fstream>
#include <sys/resource.h>
#include "./../minigrad/src/tensor.h"

using namespace std;

int main(int argc, char* argv[]) {
    string mode = "performance-plugged";
    if (argc > 1) mode = argv[1];
    
    string filepath = "./data/cpp_pagefaults_" + mode + ".jsonl";
    ofstream json_out(filepath, ios::app);
    
    uint step = 10;
    for (size_t N = 10; N <= 2000; N += step) {
        struct rusage usage1, usage2;
        
        {
            ::Tensor cpp_A1(N, N);
            ::Tensor cpp_B1(N, N);
        }
        
        getrusage(RUSAGE_SELF, &usage1);
        {
            ::Tensor cpp_A2(N, N);
            ::Tensor cpp_B2(N, N);
        }
        getrusage(RUSAGE_SELF, &usage2);
        long cpp_faults = usage2.ru_minflt - usage1.ru_minflt;
        
        json_out << "{\"N\": " << N << ", \"engine\": \"C++ (std::vector)\", \"metric\": \"minor_page_faults\", \"value\": " << cpp_faults << "}\n";

        if (step == 10 && N >= 100) step = 100;
        if (step == 100 && N >= 1000) step = 1000;
    }
    
    json_out.close();
    return 0;
}
