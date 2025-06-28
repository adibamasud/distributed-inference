#!/bin/bash
# Manual commands to test different splits

echo "=== Manual Split Point Testing ==="
echo ""
echo "To test different split points:"
echo ""
echo "1. First, ensure workers are running on master-pi and core-pi:"
echo "   (In separate terminals or tmux/screen sessions)"
echo ""
echo "   On master-pi:"
echo "   cd ~/projects/distributed-inference && python3 distributed_runner.py --rank 1 --world-size 3"
echo ""
echo "   On core-pi:"
echo "   cd ~/projects/distributed-inference && python3 distributed_runner.py --rank 2 --world-size 3"
echo ""
echo "2. Then run these commands on your machine (PlamaLV):"
echo ""

# Test split at block 6 (best balance according to analysis)
echo "# Test Split at Block 6 (best load balance):"
echo "python distributed_runner.py --rank 0 --world-size 3 --model mobilenetv2 --batch-size 8 --num-test-samples 32 --num-partitions 2 --use-pipelining --split-block 6"
echo ""

# Test split at block 8 (current default)
echo "# Test Split at Block 8 (current default):"
echo "python distributed_runner.py --rank 0 --world-size 3 --model mobilenetv2 --batch-size 8 --num-test-samples 32 --num-partitions 2 --use-pipelining --split-block 8"
echo ""

# Test split at block 10
echo "# Test Split at Block 10:"
echo "python distributed_runner.py --rank 0 --world-size 3 --model mobilenetv2 --batch-size 8 --num-test-samples 32 --num-partitions 2 --use-pipelining --split-block 10"
echo ""

# Test split at block 12
echo "# Test Split at Block 12 (more params in shard 1):"
echo "python distributed_runner.py --rank 0 --world-size 3 --model mobilenetv2 --batch-size 8 --num-test-samples 32 --num-partitions 2 --use-pipelining --split-block 12"
echo ""

echo "Look for these metrics in the output:"
echo "- Overall throughput: X images/sec"
echo "- Split ratio: ShardX=Y%"
echo "- Network transfer size differences"