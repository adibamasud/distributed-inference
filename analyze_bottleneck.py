#!/usr/bin/env python3
"""
Analyze where the actual bottleneck is in the distributed inference system.
Separates network transfer time from computation time.
"""

import torch
import torch.nn as nn
import time
import pickle
from typing import Tuple


def measure_serialization_time(tensor: torch.Tensor, iterations: int = 10) -> Tuple[float, float]:
    """Measure time to serialize and deserialize a tensor."""
    times = []
    
    for _ in range(iterations):
        # Measure pickle serialize + deserialize
        start = time.time()
        serialized = pickle.dumps(tensor)
        _ = pickle.loads(serialized)
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    size_mb = len(serialized) / (1024 * 1024)
    
    return avg_time, size_mb


def analyze_mobilenetv2_block8_transfer():
    """Analyze the specific transfer that happens at block 8 split in MobileNetV2."""
    print("🔍 Analyzing MobileNetV2 Block-8 Transfer Bottleneck")
    print("=" * 60)
    
    # Create tensor similar to block-8 output
    # From the profile: features.7 output shape is [8, 32, 28, 28]
    batch_size = 8
    tensor_shape = (batch_size, 32, 28, 28)
    test_tensor = torch.randn(tensor_shape)
    
    print(f"\n📊 Tensor characteristics:")
    print(f"   Shape: {tensor_shape}")
    print(f"   Elements: {test_tensor.numel():,}")
    print(f"   Size: {test_tensor.numel() * 4:,} bytes ({test_tensor.numel() * 4 / (1024*1024):.2f} MB)")
    
    # Measure serialization overhead
    serialize_time, serialized_size = measure_serialization_time(test_tensor)
    print(f"\n⚙️  Serialization overhead:")
    print(f"   Pickle time (serialize + deserialize): {serialize_time:.2f} ms")
    print(f"   Serialized size: {serialized_size:.2f} MB")
    
    # Calculate theoretical transfer times
    print(f"\n📡 Theoretical network transfer times:")
    
    # At different bandwidths
    bandwidths = [
        ("Raw network (940 Mbps)", 940),
        ("100 Mbps (original estimate)", 100),
        ("WiFi typical (50 Mbps)", 50),
        ("Measured effective (3.5 Mbps)", 3.5),
        ("Observed (0.12 Mbps)", 0.12)
    ]
    
    for name, bandwidth_mbps in bandwidths:
        # Round trip (send input + receive output)
        transfer_time = (2 * serialized_size * 8 / bandwidth_mbps) * 1000  # ms
        total_time = serialize_time + transfer_time + 0.5  # +0.5ms for network RTT
        print(f"   {name}: {transfer_time:.1f} ms transfer + {serialize_time:.1f} ms serialize = {total_time:.1f} ms total")
    
    # From logs: actual "network latency" was 1824ms
    print(f"\n⚠️  Reality check:")
    print(f"   Actual measured 'network latency': 1824.31 ms")
    print(f"   This includes computation time on the worker!")
    
    # Estimate computation time
    estimated_pure_network = serialize_time + (2 * serialized_size * 8 / 940) * 1000 + 0.5
    estimated_computation = 1824.31 - estimated_pure_network
    
    print(f"\n💡 Breakdown estimate:")
    print(f"   Pure network overhead: ~{estimated_pure_network:.1f} ms")
    print(f"   Worker computation time: ~{estimated_computation:.1f} ms")
    print(f"   Computation is {estimated_computation/estimated_pure_network:.0f}x larger than network!")
    
    # Test a simple convolution to verify
    print(f"\n🧪 Testing actual computation time for similar operations...")
    
    # Create a simple conv layer similar to what might be in block 8
    conv = nn.Conv2d(32, 96, kernel_size=1, stride=1)
    
    # Warm up
    for _ in range(5):
        _ = conv(test_tensor)
    
    # Measure
    times = []
    for _ in range(10):
        start = time.time()
        _ = conv(test_tensor)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
    
    conv_time = sum(times) / len(times)
    print(f"   Single Conv2d forward pass: {conv_time:.2f} ms")
    print(f"   (A full block would have multiple layers, so ~1800ms is plausible on a Pi)")


if __name__ == "__main__":
    analyze_mobilenetv2_block8_transfer()