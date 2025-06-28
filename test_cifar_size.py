#!/usr/bin/env python3
"""
Test if our CIFAR-10 trained models can handle 32x32 input directly.
"""

import torch
from core.model_loader import ModelLoader

def test_model_with_cifar_size(model_type):
    """Test if model works with 32x32 CIFAR-10 native size."""
    print(f"\nTesting {model_type} with 32x32 input:")
    
    loader = ModelLoader()
    model = loader.load_model(model_type, num_classes=10)
    
    # Test with CIFAR-10 native size
    test_input_32 = torch.randn(1, 3, 32, 32)
    
    try:
        with torch.no_grad():
            output = model(test_input_32)
        print(f"✓ {model_type} works with 32x32! Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ {model_type} failed with 32x32: {e}")
        return False

# Test each model
models = ["mobilenetv2", "resnet18", "alexnet", "squeezenet"]

print("Testing CIFAR-10 models with native 32x32 resolution...")
print("=" * 50)

working_models = []
for model_type in models:
    if test_model_with_cifar_size(model_type):
        working_models.append(model_type)

print(f"\n\nSummary:")
print(f"Models that work with 32x32: {working_models}")
print(f"Models that require resize: {[m for m in models if m not in working_models]}")

print("\n\nPerformance Impact:")
print("- Current: 224×224 = 50,176 pixels")
print("- Native:   32×32 =  1,024 pixels")
print("- Speedup potential: ~49x reduction in computation!")