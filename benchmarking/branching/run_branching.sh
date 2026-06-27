#!/bin/bash

# ensure i are in the right directory
cd "$(dirname "$0")"

# just create data directory
mkdir -p data
LOG_FILE="data/branch_data.jsonl"
rm -f $LOG_FILE

echo "Compiling C++ engine and perf wrapper..."
g++ -O3 -march=native -shared -fPIC perf_wrapper.cpp -o libperf.so
g++ -O3 -march=native cpp_branch.cpp -o cpp_branch -mavx2

SIZES=(32 64 128 256 512 1024 2048)
BACKENDS=("py" "naive" "tiled" "simd")

echo "Starting branch benchmarks..."

for N in "${SIZES[@]}"; do
    for BACKEND in "${BACKENDS[@]}"; do
        if [ "$BACKEND" == "py" ]; then
            CMD="python3 py_branch.py $N"
        else
            CMD="./cpp_branch $N $BACKEND"
        fi
        
        # the scripts output json to stdout. capture it directly to the jsonl file.
        OUTPUT=$($CMD)
        
        if [[ $OUTPUT == {* ]]; then
            echo "$OUTPUT" >> $LOG_FILE
            echo "  -> N=$N | $BACKEND | Recorded."
        else
            echo "  -> Error running $CMD: $OUTPUT"
        fi
    done
done

echo "Benchmarking complete! Data saved to $LOG_FILE"
