#!/usr/bin/env python3
"""
Test pure RPC performance without computation to isolate network/serialization overhead.
This will help us understand the real network bottleneck vs computation bottleneck.
"""

import torch
import torch.distributed.rpc as rpc
import time
import argparse
import os
from typing import Tuple, List
import statistics


def tensor_echo(tensor: torch.Tensor) -> torch.Tensor:
    """Simple echo function that just returns the tensor without computation."""
    return tensor


def measure_pure_rpc_latency(worker_name: str, tensor: torch.Tensor, iterations: int = 10) -> Tuple[float, float, float]:
    """
    Measure pure RPC round-trip time without computation.
    Returns: (mean_ms, std_ms, throughput_mbps)
    """
    latencies = []
    tensor_size_mb = (tensor.numel() * tensor.element_size()) / (1024 * 1024)
    
    # Warm up
    for _ in range(3):
        _ = rpc.rpc_sync(worker_name, tensor_echo, args=(tensor,))
    
    # Measure
    for _ in range(iterations):
        start_time = time.time()
        result = rpc.rpc_sync(worker_name, tensor_echo, args=(tensor,))
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
    
    mean_ms = statistics.mean(latencies)
    std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    
    # Calculate effective throughput (2x tensor size for round trip)
    throughput_mbps = (2 * tensor_size_mb * 8) / (mean_ms / 1000) if mean_ms > 0 else 0
    
    return mean_ms, std_ms, throughput_mbps


def test_tensor_sizes(worker_name: str):
    """Test various tensor sizes to understand RPC overhead."""
    print("\n🔍 Testing RPC Performance with Different Tensor Sizes")
    print("=" * 60)
    
    # Test sizes (similar to what would be transferred in model inference)
    test_configs = [
        ("Small (1KB)", (1, 256)),           # 1KB
        ("Medium (100KB)", (1, 25600)),      # 100KB  
        ("Large (1MB)", (1, 262144)),        # 1MB
        ("XL (10MB)", (1, 2621440)),         # 10MB
        ("Block-8 size", (8, 32, 28, 28)),   # MobileNetV2 block-8 transfer size
    ]
    
    results = []
    
    for name, shape in test_configs:
        tensor = torch.randn(shape)
        size_mb = (tensor.numel() * 4) / (1024 * 1024)
        
        print(f"\n📊 {name} - Tensor shape: {shape}, Size: {size_mb:.2f} MB")
        
        mean_ms, std_ms, throughput_mbps = measure_pure_rpc_latency(worker_name, tensor)
        
        print(f"   RPC latency: {mean_ms:.2f} ± {std_ms:.2f} ms")
        print(f"   Effective throughput: {throughput_mbps:.1f} Mbps")
        print(f"   Serialization overhead: ~{mean_ms - 0.5:.1f} ms (assuming 0.5ms network RTT)")
        
        results.append({
            'name': name,
            'size_mb': size_mb,
            'latency_ms': mean_ms,
            'throughput_mbps': throughput_mbps
        })
    
    return results


def run_worker(rank: int, world_size: int):
    """Worker process that just echoes tensors."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    
    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=4)
    )
    
    print(f"Worker {rank} ready and waiting for RPC calls...")
    rpc.shutdown()


def run_master(rank: int, world_size: int):
    """Master process that tests RPC performance."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    
    rpc.init_rpc(
        "master",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=4)
    )
    
    print("Master initialized. Testing RPC performance...")
    
    # Test with each worker
    for worker_rank in range(1, world_size):
        worker_name = f"worker{worker_rank}"
        print(f"\n\n🌐 Testing with {worker_name}")
        print("-" * 60)
        
        results = test_tensor_sizes(worker_name)
        
        # Summary
        print(f"\n📈 Summary for {worker_name}:")
        print(f"{'Size':<15} {'Latency (ms)':<15} {'Throughput (Mbps)':<20}")
        print("-" * 50)
        for r in results:
            print(f"{r['name']:<15} {r['latency_ms']:<15.2f} {r['throughput_mbps']:<20.1f}")
    
    print("\n✅ RPC performance testing complete!")
    rpc.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Test pure RPC performance')
    parser.add_argument('--rank', type=int, required=True, help='Process rank')
    parser.add_argument('--world-size', type=int, required=True, help='Total number of processes')
    args = parser.parse_args()
    
    if args.rank == 0:
        run_master(args.rank, args.world_size)
    else:
        run_worker(args.rank, args.world_size)


if __name__ == "__main__":
    main()