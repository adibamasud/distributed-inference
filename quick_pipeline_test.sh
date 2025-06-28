#!/bin/bash
# Quick test of pipelining functionality

echo "=== Quick Pipelining Test ==="
echo ""

# Deploy code first
echo "Deploying code to workers..."
./deploy_to_workers.sh

echo ""
echo "Starting workers..."

# Start workers
ssh master-pi "cd ~/projects/distributed-inference && python3 distributed_runner.py --rank 1 --world-size 3" > worker1_pipeline.log 2>&1 &
W1=$!

ssh core-pi "cd ~/projects/distributed-inference && python3 distributed_runner.py --rank 2 --world-size 3" > worker2_pipeline.log 2>&1 &
W2=$!

# Wait for workers to start
echo "Waiting for workers to initialize..."
sleep 10

echo ""
echo "Running master with pipelining enabled..."
echo ""

# Run master with pipelining
python distributed_runner.py \
    --rank 0 \
    --world-size 3 \
    --model mobilenetv2 \
    --batch-size 8 \
    --num-test-samples 32 \
    --num-partitions 2 \
    --use-pipelining \
    --metrics-dir ./metrics_pipelined_test

# Cleanup
echo ""
echo "Cleaning up..."
ssh master-pi "pkill -f distributed_runner" || true
ssh core-pi "pkill -f distributed_runner" || true
kill $W1 $W2 2>/dev/null || true

echo ""
echo "Test complete. Check output above for 'Using PIPELINED inference' message."