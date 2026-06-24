#!/bin/bash

# Ensure we are in the right directory
cd "$(dirname "$0")"

# Create data directory
mkdir -p data
LOG_FILE="data/bandwidth_data.jsonl"
rm -f $LOG_FILE

echo "Compiling C++ engine with OpenMP..."
g++ -O3 -march=native -fopenmp cpp_bandwidth.cpp -o cpp_bandwidth -mavx2

SIZES=(32 64 128 256 512 1024 2048)
BACKENDS=("py" "naive" "tiled" "simd")

echo "Starting Memory Bandwidth Roofline Benchmarks..."

for N in "${SIZES[@]}"; do
    echo "====================================="
    echo "Sweeping Matrix Size: N=$N"
    echo "====================================="
    for BACKEND in "${BACKENDS[@]}"; do
        
        if [ "$BACKEND" == "py" ]; then
            CMD="python3 py_bandwidth.py $N"
        else
            CMD="./cpp_bandwidth $N $BACKEND"
        fi
        
        EXEC_TIME=$($CMD)
        
        if ! [[ $EXEC_TIME =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            echo "Error running $CMD: $EXEC_TIME"
            continue
        fi

        # Calculate metrics using bc
        # Bytes = 3 * N^2 * 4 (float32 size)
        # GB/s = Bytes / (1024^3 * TIME)
        GB_S=$(echo "scale=4; (3 * $N * $N * 4) / (1073741824 * $EXEC_TIME)" | bc)
        
        # FLOPs = 2 * N^3
        # GFLOPS = FLOPs / (10^9 * TIME)
        GFLOPS=$(echo "scale=4; (2 * $N * $N * $N) / (1000000000 * $EXEC_TIME)" | bc)

        # Write JSONL
        echo "{\"N\": $N, \"backend\": \"$BACKEND\", \"time\": $EXEC_TIME, \"gb_s\": $GB_S, \"gflops\": $GFLOPS}" >> $LOG_FILE
        echo "  -> Backend=$BACKEND | Time: ${EXEC_TIME}s | Bandwidth: ${GB_S} GB/s | Compute: ${GFLOPS} GFLOPS"
    done
done

echo "Benchmarking complete! Data saved to $LOG_FILE"
