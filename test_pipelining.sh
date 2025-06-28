#!/bin/bash
# Script to test the new pipelining implementation

echo "=== Testing Pipelining Implementation ==="
echo ""
echo "This script will test both sequential and pipelined execution"
echo "to demonstrate the performance improvement from pipelining."
echo ""

# Test configuration
MODEL="mobilenetv2"
BATCH_SIZE=8
NUM_SAMPLES=64

echo "Configuration:"
echo "- Model: $MODEL"
echo "- Batch size: $BATCH_SIZE"
echo "- Test samples: $NUM_SAMPLES"
echo ""

# Create output directory for logs
mkdir -p pipelining_test_logs

echo "=== Test 1: Sequential Execution (No Pipelining) ==="
echo "Starting master and workers..."
echo ""

# First, deploy the updated code to workers
echo "Deploying updated code to workers..."
./deploy_to_workers.sh

# Start workers in background
echo "Starting worker 1..."
ssh master-pi "cd ~/projects/distributed-inference && python3 distributed_runner.py --rank 1 --world-size 3" > pipelining_test_logs/worker1_sequential.log 2>&1 &
WORKER1_PID=$!

echo "Starting worker 2..."
ssh core-pi "cd ~/projects/distributed-inference && python3 distributed_runner.py --rank 2 --world-size 3" > pipelining_test_logs/worker2_sequential.log 2>&1 &
WORKER2_PID=$!

# Give workers time to start
sleep 5

# Run master WITHOUT pipelining
echo "Running master (sequential mode)..."
python distributed_runner.py \
    --rank 0 \
    --world-size 3 \
    --model $MODEL \
    --batch-size $BATCH_SIZE \
    --num-test-samples $NUM_SAMPLES \
    --num-partitions 2 \
    --metrics-dir ./metrics_sequential \
    2>&1 | tee pipelining_test_logs/master_sequential.log

# Kill workers
echo "Stopping workers..."
kill $WORKER1_PID $WORKER2_PID 2>/dev/null
ssh master-pi "pkill -f distributed_runner" 2>/dev/null
ssh core-pi "pkill -f distributed_runner" 2>/dev/null

# Extract throughput from sequential run
SEQUENTIAL_IPS=$(grep "Overall throughput:" pipelining_test_logs/master_sequential.log | tail -1 | awk '{print $3}')
echo ""
echo "Sequential throughput: $SEQUENTIAL_IPS images/sec"
echo ""

# Wait before next test
sleep 5

echo "=== Test 2: Pipelined Execution ==="
echo "Starting master and workers with pipelining enabled..."
echo ""

# No need to copy again since we already did it above
# Just start workers in background
echo "Starting worker 1..."
ssh master-pi "cd ~/projects/distributed-inference && python3 distributed_runner.py --rank 1 --world-size 3" > pipelining_test_logs/worker1_pipelined.log 2>&1 &
WORKER1_PID=$!

echo "Starting worker 2..."
ssh core-pi "cd ~/projects/distributed-inference && python3 distributed_runner.py --rank 2 --world-size 3" > pipelining_test_logs/worker2_pipelined.log 2>&1 &
WORKER2_PID=$!

# Give workers time to start
sleep 5

# Run master WITH pipelining
echo "Running master (pipelined mode)..."
python distributed_runner.py \
    --rank 0 \
    --world-size 3 \
    --model $MODEL \
    --batch-size $BATCH_SIZE \
    --num-test-samples $NUM_SAMPLES \
    --num-partitions 2 \
    --use-pipelining \
    --metrics-dir ./metrics_pipelined \
    2>&1 | tee pipelining_test_logs/master_pipelined.log

# Kill workers
echo "Stopping workers..."
kill $WORKER1_PID $WORKER2_PID 2>/dev/null
ssh master-pi "pkill -f distributed_runner" 2>/dev/null
ssh core-pi "pkill -f distributed_runner" 2>/dev/null

# Extract throughput from pipelined run
PIPELINED_IPS=$(grep "Overall throughput:" pipelining_test_logs/master_pipelined.log | tail -1 | awk '{print $3}')
echo ""
echo "Pipelined throughput: $PIPELINED_IPS images/sec"
echo ""

echo "=== Summary ==="
echo "Sequential: $SEQUENTIAL_IPS images/sec"
echo "Pipelined:  $PIPELINED_IPS images/sec"

# Calculate speedup if both values exist
if [ ! -z "$SEQUENTIAL_IPS" ] && [ ! -z "$PIPELINED_IPS" ]; then
    SPEEDUP=$(echo "scale=2; $PIPELINED_IPS / $SEQUENTIAL_IPS" | bc)
    echo "Speedup:    ${SPEEDUP}x"
fi

echo ""
echo "Logs saved in: pipelining_test_logs/"
echo ""
echo "To view pipeline behavior:"
echo "- Check for 'Using PIPELINED inference' in master_pipelined.log"
echo "- Look for 'Starting batch X (pipeline)' messages showing concurrent processing"
echo "- Compare timing patterns between sequential and pipelined logs"