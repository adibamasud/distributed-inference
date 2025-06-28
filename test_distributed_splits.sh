#!/bin/bash
# Test different split points in distributed setup

echo "=== Testing Different Split Points in Distributed Setup ==="
echo ""

# Configuration
MODEL="mobilenetv2"
BATCH_SIZE=8
NUM_SAMPLES=32  # Smaller for faster testing

# Array of split blocks to test
SPLIT_BLOCKS=(6 8 10 12 14)

# Deploy code first
echo "Deploying code to workers..."
./deploy_to_workers.sh > /dev/null 2>&1

# Function to test a specific split
test_split() {
    local split_block=$1
    echo ""
    echo "=== Testing Split at Block $split_block ==="
    
    # Kill any existing processes
    ssh master-pi "pkill -f distributed_runner" 2>/dev/null || true
    ssh core-pi "pkill -f distributed_runner" 2>/dev/null || true
    pkill -f distributed_runner 2>/dev/null || true
    sleep 2
    
    # Start workers
    ssh master-pi "cd ~/projects/distributed-inference && python3 distributed_runner.py --rank 1 --world-size 3" > /tmp/worker1_split${split_block}.log 2>&1 &
    W1=$!
    
    ssh core-pi "cd ~/projects/distributed-inference && python3 distributed_runner.py --rank 2 --world-size 3" > /tmp/worker2_split${split_block}.log 2>&1 &
    W2=$!
    
    # Wait for workers
    sleep 5
    
    # Create custom distributed_runner command with split override
    # We'll need to modify the runner to accept a split-block parameter
    
    # Run master with specific split block (would need to add --split-block parameter)
    echo "Running with split at block $split_block..."
    
    # For now, let's use the environment variable approach
    export MOBILENET_SPLIT_BLOCK=$split_block
    
    # Run with pipelining
    timeout 60 python distributed_runner.py \
        --rank 0 \
        --world-size 3 \
        --model $MODEL \
        --batch-size $BATCH_SIZE \
        --num-test-samples $NUM_SAMPLES \
        --num-partitions 2 \
        --use-pipelining \
        2>&1 | grep -E "(Overall throughput|Split ratio|pipeline_time_ms)" | tail -5
    
    # Clean up
    ssh master-pi "pkill -f distributed_runner" 2>/dev/null || true
    ssh core-pi "pkill -f distributed_runner" 2>/dev/null || true
    kill $W1 $W2 2>/dev/null || true
}

# First run the analysis script locally
echo "Running local split analysis..."
python test_split_points.py --min-block 5 --max-block 15

echo ""
echo "=== Distributed Split Testing ==="
echo "Note: To test different splits in distributed mode, we need to modify"
echo "distributed_runner.py to accept a --split-block parameter."
echo ""
echo "For now, you can manually edit the split point in distributed_runner.py"
echo "at line 212 where it says 'split_at_block = 8'"
echo ""
echo "Alternatively, run the local analysis above to find the best split,"
echo "then update the code with that value."