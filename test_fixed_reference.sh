#!/bin/bash
# Script to test the fixed reference implementation

echo "=== Testing Fixed Reference Implementation ==="
echo "This implementation:"
echo "- Uses CIFAR-10 trained models (or modifies classifier to 10 classes)"
echo "- Measures accuracy"
echo "- Reports meaningful metrics"
echo ""

# Copy necessary model weights to References directory
echo "Setting up model weights..."
mkdir -p References/models
cp models/mobilenetv2_cifar10.pth References/models/ 2>/dev/null || echo "MobileNetV2 CIFAR-10 weights not found"
cp models/resnet18_cifar10.pth References/models/ 2>/dev/null || echo "ResNet18 CIFAR-10 weights not found"

echo ""
echo "To run the fixed implementation:"
echo ""
echo "1. On worker1-pi (server):"
echo "   python partition1_fixed.py --models MobileNetV2"
echo ""
echo "2. On worker2-pi (client):"
echo "   python partition2_fixed.py --models MobileNetV2 --host <worker1-pi-ip>"
echo ""
echo "The fixed version will:"
echo "- Load CIFAR-10 trained models (10 classes)"
echo "- Send labels from server to client"
echo "- Calculate and report accuracy"
echo "- Show meaningful metrics"
echo ""
echo "Expected output differences:"
echo "- Accuracy reported (should be ~80%+ with proper models)"
echo "- Slightly slower due to proper 10-class models"
echo "- Meaningful benchmark for CIFAR-10 distributed inference"