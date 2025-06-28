#!/usr/bin/env python3
"""
Test different split points for MobileNetV2 to find optimal partitioning.
"""

import os
os.environ['ATEN_CPU_CAPABILITY'] = ''

import torch
import torch.nn as nn
import torchvision.models as models
import time
import argparse
import json
from typing import List, Tuple, Dict

def analyze_mobilenetv2_structure():
    """Analyze MobileNetV2 structure to understand possible split points."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    print("=== MobileNetV2 Structure Analysis ===")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Analyze feature blocks
    feature_blocks = list(model.features.children())
    print(f"\nTotal feature blocks: {len(feature_blocks)}")
    
    cumulative_params = 0
    block_info = []
    
    for i, block in enumerate(feature_blocks):
        block_params = sum(p.numel() for p in block.parameters())
        cumulative_params += block_params
        total_params = sum(p.numel() for p in model.parameters())
        cumulative_percent = (cumulative_params / total_params) * 100
        
        block_info.append({
            'index': i,
            'name': f"features.{i}",
            'type': type(block).__name__,
            'params': block_params,
            'cumulative_params': cumulative_params,
            'cumulative_percent': cumulative_percent
        })
        
        print(f"Block {i:2d}: {type(block).__name__:20s} "
              f"Params: {block_params:8,} "
              f"Cumulative: {cumulative_percent:5.1f}%")
    
    # Add classifier info
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    print(f"\nClassifier parameters: {classifier_params:,}")
    
    return model, block_info


def create_split_at_block(model, split_block: int) -> Tuple[nn.Module, nn.Module]:
    """Create a split at a specific block number."""
    feature_blocks = list(model.features.children())
    
    # Create shard 1: features up to split_block
    shard1_modules = feature_blocks[:split_block]
    shard1 = nn.Sequential(*shard1_modules)
    
    # Create shard 2: remaining features + pooling + classifier
    shard2_modules = feature_blocks[split_block:]
    shard2_modules.append(nn.AdaptiveAvgPool2d((1, 1)))
    shard2_modules.append(nn.Flatten())
    shard2_modules.append(model.classifier)
    shard2 = nn.Sequential(*shard2_modules)
    
    return shard1, shard2


def benchmark_split(model, split_block: int, num_iterations: int = 100, 
                   batch_size: int = 8, warmup: int = 10) -> Dict:
    """Benchmark a specific split configuration."""
    shard1, shard2 = create_split_at_block(model, split_block)
    
    # Move to eval mode
    shard1.eval()
    shard2.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            intermediate = shard1(dummy_input)
            output = shard2(intermediate)
    
    # Benchmark shard 1
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(num_iterations):
        with torch.no_grad():
            intermediate = shard1(dummy_input)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    shard1_time = (time.time() - start_time) / num_iterations * 1000  # ms
    
    # Benchmark shard 2
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(num_iterations):
        with torch.no_grad():
            output = shard2(intermediate)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    shard2_time = (time.time() - start_time) / num_iterations * 1000  # ms
    
    # Calculate metrics
    shard1_params = sum(p.numel() for p in shard1.parameters())
    shard2_params = sum(p.numel() for p in shard2.parameters())
    total_params = shard1_params + shard2_params
    
    # Calculate load balance (lower is better)
    load_balance = abs(shard1_time - shard2_time) / max(shard1_time, shard2_time)
    
    # Estimate pipeline throughput (limited by slowest stage)
    pipeline_time = max(shard1_time, shard2_time)
    sequential_time = shard1_time + shard2_time
    speedup = sequential_time / pipeline_time
    
    # Get intermediate tensor size
    intermediate_size_mb = (intermediate.numel() * intermediate.element_size()) / (1024 * 1024)
    
    return {
        'split_block': split_block,
        'shard1_params': shard1_params,
        'shard2_params': shard2_params,
        'shard1_percent': (shard1_params / total_params) * 100,
        'shard2_percent': (shard2_params / total_params) * 100,
        'shard1_time_ms': shard1_time,
        'shard2_time_ms': shard2_time,
        'load_balance_score': load_balance,
        'pipeline_time_ms': pipeline_time,
        'sequential_time_ms': sequential_time,
        'theoretical_speedup': speedup,
        'intermediate_size_mb': intermediate_size_mb
    }


def find_optimal_splits(model, block_info: List[Dict], 
                       min_block: int = 3, max_block: int = 16) -> List[Dict]:
    """Test different split points and find optimal ones."""
    results = []
    
    print("\n=== Testing Different Split Points ===")
    print("Split | Shard1% | Shard2% | S1 Time | S2 Time | Balance | Speedup | Data MB")
    print("-" * 80)
    
    for split_block in range(min_block, min(max_block + 1, len(block_info))):
        result = benchmark_split(model, split_block)
        results.append(result)
        
        print(f"{split_block:5d} | {result['shard1_percent']:7.1f} | "
              f"{result['shard2_percent']:7.1f} | {result['shard1_time_ms']:7.2f} | "
              f"{result['shard2_time_ms']:7.2f} | {result['load_balance_score']:7.3f} | "
              f"{result['theoretical_speedup']:7.2f} | {result['intermediate_size_mb']:7.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test different split points for MobileNetV2")
    parser.add_argument("--min-block", type=int, default=3, 
                       help="Minimum block to test splitting at")
    parser.add_argument("--max-block", type=int, default=16,
                       help="Maximum block to test splitting at")
    parser.add_argument("--num-iterations", type=int, default=100,
                       help="Number of iterations for benchmarking")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for testing")
    parser.add_argument("--save-results", type=str, default="split_analysis.json",
                       help="File to save results")
    
    args = parser.parse_args()
    
    # Analyze model structure
    model, block_info = analyze_mobilenetv2_structure()
    
    # Test different splits
    results = find_optimal_splits(model, block_info, 
                                 min_block=args.min_block, 
                                 max_block=args.max_block)
    
    # Find best splits based on different criteria
    print("\n=== Best Split Points ===")
    
    # Best load balance
    best_balance = min(results, key=lambda x: x['load_balance_score'])
    print(f"\nBest Load Balance: Block {best_balance['split_block']}")
    print(f"  - Load balance score: {best_balance['load_balance_score']:.3f}")
    print(f"  - Shard times: {best_balance['shard1_time_ms']:.2f}ms / {best_balance['shard2_time_ms']:.2f}ms")
    print(f"  - Parameter split: {best_balance['shard1_percent']:.1f}% / {best_balance['shard2_percent']:.1f}%")
    
    # Best for 50/50 parameter split
    best_param_balance = min(results, key=lambda x: abs(50 - x['shard1_percent']))
    print(f"\nBest Parameter Balance (closest to 50/50): Block {best_param_balance['split_block']}")
    print(f"  - Parameter split: {best_param_balance['shard1_percent']:.1f}% / {best_param_balance['shard2_percent']:.1f}%")
    print(f"  - Shard times: {best_param_balance['shard1_time_ms']:.2f}ms / {best_param_balance['shard2_time_ms']:.2f}ms")
    
    # Fastest pipeline time
    fastest_pipeline = min(results, key=lambda x: x['pipeline_time_ms'])
    print(f"\nFastest Pipeline Time: Block {fastest_pipeline['split_block']}")
    print(f"  - Pipeline time: {fastest_pipeline['pipeline_time_ms']:.2f}ms")
    print(f"  - Theoretical speedup: {fastest_pipeline['theoretical_speedup']:.2f}x")
    
    # Save results
    with open(args.save_results, 'w') as f:
        json.dump({
            'model': 'mobilenetv2',
            'block_info': block_info,
            'results': results,
            'best_balance': best_balance,
            'best_param_balance': best_param_balance,
            'fastest_pipeline': fastest_pipeline
        }, f, indent=2)
    
    print(f"\nResults saved to {args.save_results}")
    
    # Recommendation
    print("\n=== Recommendation ===")
    print(f"For your Raspberry Pi setup, consider splitting at block {best_balance['split_block']}:")
    print(f"- This provides the best computational load balance")
    print(f"- Expected speedup with pipelining: {best_balance['theoretical_speedup']:.2f}x")
    print(f"- Network transfer per batch: {best_balance['intermediate_size_mb']:.2f}MB")


if __name__ == "__main__":
    main()