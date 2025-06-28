#!/usr/bin/env python3
"""
Quick test to show the performance difference with native CIFAR-10 size.
"""

import torch
import time
from core.model_loader import ModelLoader
from core.model_loader_cifar_native import ModelLoaderCifarNative

def benchmark_inference(model, input_tensor, name, iterations=10):
    """Benchmark model inference time."""
    # Warm up
    for _ in range(3):
        with torch.no_grad():
            _ = model(input_tensor)
    
    # Time it
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(input_tensor)
    elapsed = time.time() - start
    
    avg_time = (elapsed / iterations) * 1000  # ms
    images_per_sec = (input_tensor.shape[0] * iterations) / elapsed
    
    print(f"{name}:")
    print(f"  Average inference time: {avg_time:.2f} ms/batch")
    print(f"  Throughput: {images_per_sec:.2f} images/sec")
    
    return avg_time

# Test setup
batch_size = 5
model_type = "mobilenetv2"

print(f"Benchmarking {model_type} with batch_size={batch_size}")
print("=" * 60)

# Load models
print("\nLoading models...")
loader_224 = ModelLoader()
loader_32 = ModelLoaderCifarNative()

model_224 = loader_224.load_model(model_type, num_classes=10)
model_32 = loader_32.load_model(model_type, num_classes=10)

# Create test inputs
input_224 = torch.randn(batch_size, 3, 224, 224)
input_32 = torch.randn(batch_size, 3, 32, 32)

print(f"\nInput tensor sizes:")
print(f"  224x224: {input_224.shape} = {input_224.numel():,} elements")
print(f"  32x32:   {input_32.shape} = {input_32.numel():,} elements")
print(f"  Ratio: {input_224.numel() / input_32.numel():.1f}x more data with resize")

print(f"\n\nBenchmarking on CPU (like Raspberry Pi)...")
print("-" * 60)

time_224 = benchmark_inference(model_224, input_224, "224x224 (current)")
time_32 = benchmark_inference(model_32, input_32, "32x32 (native)")

print(f"\nSpeedup: {time_224/time_32:.1f}x faster with native resolution!")
print(f"\nProjected impact on distributed inference:")
print(f"  Current: ~1319ms worker computation")
print(f"  Projected: ~{1319/time_224*time_32:.0f}ms worker computation")
print(f"  Potential throughput: ~{2.19 * time_224/time_32:.1f} images/sec (vs current 2.19)")