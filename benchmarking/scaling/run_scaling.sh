#!/bin/bash

# Ensure we are in the right directory
cd "$(dirname "$0")"

# Create data directory
mkdir -p data
LOG_FILE="data/scaling_data.jsonl"
rm -f $LOG_FILE

echo "Compiling C++ engine with OpenMP..."
g++ -O3 -march=native -fopenmp cpp_scaling.cpp -o cpp_scaling -mavx2

N=2000
MAX_THREADS=20
BACKENDS=("py" "naive" "tiled" "simd")

echo "Starting Amdahl's Law Scaling Benchmarks at N=$N..."

# Associative array to store baseline (1-thread) execution time for Speedup calculation
declare -A BASELINES

for BACKEND in "${BACKENDS[@]}"; do
    echo "====================================="
    echo "Sweeping Backend: $BACKEND"
    echo "====================================="
    for THREADS in $(seq 1 $MAX_THREADS); do
        
        if [ "$BACKEND" == "py" ]; then
            CMD="python3 py_scaling.py $N $THREADS"
        else
            export OMP_NUM_THREADS=$THREADS
            CMD="./cpp_scaling $N $BACKEND"
        fi
        
        # Run command and capture execution time
        EXEC_TIME=$($CMD)
        
        # Check if the execution time is a valid number
        if ! [[ $EXEC_TIME =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            echo "Error running $CMD: $EXEC_TIME"
            continue
        fi

        # If 1 thread, record as baseline
        if [ "$THREADS" -eq 1 ]; then
            BASELINES[$BACKEND]=$EXEC_TIME
            SPEEDUP=1.0
        else
            # Calculate speedup factor = baseline / current
            BASELINE=${BASELINES[$BACKEND]}
            SPEEDUP=$(echo "scale=4; $BASELINE / $EXEC_TIME" | bc)
        fi

        # Write JSONL
        echo "{\"N\": $N, \"backend\": \"$BACKEND\", \"threads\": $THREADS, \"time\": $EXEC_TIME, \"speedup\": $SPEEDUP}" >> $LOG_FILE
        echo "  -> Threads=$THREADS | Time: ${EXEC_TIME}s | Speedup: ${SPEEDUP}x"
    done
done

echo "Benchmarking complete! Data saved to $LOG_FILE"
