#!/bin/bash
# Simple script to test different split points

echo "=== Testing Different Split Points ==="
echo ""
echo "This script assumes you have already started workers on master-pi and core-pi"
echo ""

# Configuration
SPLITS_TO_TEST=(6 8 10 12)
BATCH_SIZE=8
NUM_SAMPLES=32

# Results file
RESULTS_FILE="split_test_results.txt"
echo "Split Block | Throughput | Accuracy | Time" > $RESULTS_FILE

for split in "${SPLITS_TO_TEST[@]}"; do
    echo ""
    echo "=== Testing Split at Block $split ==="
    
    # Run the test and capture output
    output=$(python distributed_runner.py \
        --rank 0 \
        --world-size 3 \
        --model mobilenetv2 \
        --batch-size $BATCH_SIZE \
        --num-test-samples $NUM_SAMPLES \
        --num-partitions 2 \
        --use-pipelining \
        --split-block $split 2>&1)
    
    # Extract metrics
    throughput=$(echo "$output" | grep "Overall throughput:" | tail -1 | awk '{print $3}')
    accuracy=$(echo "$output" | grep "Final accuracy:" | tail -1 | awk '{print $3}')
    total_time=$(echo "$output" | grep "Total time:" | tail -1 | awk '{print $3}')
    
    echo "Results: $throughput images/sec, $accuracy accuracy, $total_time"
    echo "$split | $throughput | $accuracy | $total_time" >> $RESULTS_FILE
    
    # Small delay between tests
    sleep 2
done

echo ""
echo "=== Test Complete ==="
echo "Results saved to: $RESULTS_FILE"
echo ""
cat $RESULTS_FILE