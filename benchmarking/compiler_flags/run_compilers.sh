#!/bin/bash

# honestly ensure i are in the right directory
cd "$(dirname "$0")"

# create data directory
mkdir -p data
LOG_FILE="data/compiler_data.jsonl"
rm -f $LOG_FILE

FLAGS=("-O0" "-O1" "-O2" "-O3" "-Ofast" "-O3 -march=native")

echo "Starting Compiler Optimization Benchmarks (N=1000)..."

BASELINE_TIME=""

for FLAG in "${FLAGS[@]}"; do
    echo "====================================="
    echo "Compiling with: g++ $FLAG"
    
    # compile
    g++ $FLAG compiler_bench.cpp -o compiler_bench
    if [ $? -ne 0 ]; then
        echo "Compilation failed for $FLAG"
        continue
    fi
    
    # execute and capture time
    EXEC_TIME=$(./compiler_bench)
    
    # ensure its a valid number
    if ! [[ $EXEC_TIME =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "Error running with $FLAG: $EXEC_TIME"
        continue
    fi
    
    # speedup calculation
    if [ "$FLAG" == "-O0" ]; then
        BASELINE_TIME=$EXEC_TIME
        SPEEDUP=1.0
    else
        SPEEDUP=$(echo "scale=4; $BASELINE_TIME / $EXEC_TIME" | bc)
    fi
    
    # clean up the output string format to prevent json issues
    SAFE_FLAG=$(echo "$FLAG" | tr -d '"')

    # write jsonl
    echo "{\"flag\": \"$SAFE_FLAG\", \"time\": $EXEC_TIME, \"speedup\": $SPEEDUP}" >> $LOG_FILE
    echo "  -> Time: ${EXEC_TIME}s | Speedup vs -O0: ${SPEEDUP}x"
done

# cleanup binary
rm -f compiler_bench

echo "Benchmarking complete! Data saved to $LOG_FILE"
