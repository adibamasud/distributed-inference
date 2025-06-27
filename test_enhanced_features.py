#!/usr/bin/env python3
"""
Test script for enhanced distributed inference features.
Tests individual components without requiring RPC setup.
"""

import sys
import os
sys.path.append('.')

# Add parent directory to path for model files
sys.path.append('..')

import torch
import logging
from profiling import LayerProfiler, IntelligentSplitter, split_model_intelligently
from metrics import EnhancedMetricsCollector
from core import ModelLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_model_loading():
    """Test model loading with enhanced system."""
    logger.info("=== Testing Model Loading ===")
    
    model_loader = ModelLoader("./models")  # Look for models in models directory
    
    # Test with dummy data first
    model = model_loader.load_model("mobilenetv2", num_classes=10)
    sample_input = model_loader.get_sample_input("mobilenetv2", batch_size=1)
    
    logger.info(f"Model loaded successfully: {type(model).__name__}")
    logger.info(f"Sample input shape: {sample_input.shape}")
    
    # Test forward pass
    with torch.no_grad():
        output = model(sample_input)
    logger.info(f"Model output shape: {output.shape}")
    
    return model, sample_input


def test_profiling(model, sample_input):
    """Test the profiling system."""
    logger.info("=== Testing Profiling System ===")
    
    # Profile the model
    profiler = LayerProfiler(device="cpu", warmup_iterations=1, profile_iterations=3)
    profile = profiler.profile_model(model, sample_input, "test_mobilenetv2")
    
    logger.info(f"Profiled {len(profile.layer_profiles)} layers")
    logger.info(f"Total execution time: {profile.total_time_ms:.2f}ms")
    logger.info(f"Total parameters: {profile.total_parameters:,}")
    
    # Find most expensive layers
    sorted_layers = sorted(profile.layer_profiles, key=lambda x: x.execution_time_ms, reverse=True)
    logger.info("Top 3 most expensive layers:")
    for i, layer in enumerate(sorted_layers[:3]):
        logger.info(f"  {i+1}. {layer.layer_name}: {layer.execution_time_ms:.2f}ms")
    
    return profile


def test_intelligent_splitting(model, profile):
    """Test intelligent splitting."""
    logger.info("=== Testing Intelligent Splitting ===")
    
    # Test intelligent splitting
    shards, split_config = split_model_intelligently(
        model, profile, num_splits=2,
        network_config={'communication_latency_ms': 5.0, 'network_bandwidth_mbps': 100.0}
    )
    
    logger.info(f"Created {len(shards)} shards")
    logger.info(f"Load balance score: {split_config.load_balance_score:.4f}")
    logger.info(f"Communication overhead: {split_config.estimated_communication_overhead_ms:.2f}ms")
    
    # Test each shard
    current_input = torch.randn(1, 3, 224, 224)
    for i, shard in enumerate(shards):
        logger.info(f"Testing shard {i} - input shape: {current_input.shape}")
        with torch.no_grad():
            current_input = shard(current_input)
        logger.info(f"Shard {i} output shape: {current_input.shape}")
    
    return shards


def test_enhanced_metrics():
    """Test enhanced metrics collection."""
    logger.info("=== Testing Enhanced Metrics ===")
    
    metrics = EnhancedMetricsCollector(rank=0, output_dir="./test_metrics")
    
    # Simulate some processing
    import time
    
    for batch_id in range(2):
        # Start batch
        metrics.start_batch(batch_id, batch_size=4)
        
        # Simulate processing stages
        for stage_id in range(2):
            stage_start = time.time()
            time.sleep(0.05)  # Simulate work
            stage_end = time.time()
            
            metrics.record_pipeline_stage(
                batch_id=batch_id,
                stage_id=stage_id,
                stage_name=f"test_stage_{stage_id}",
                start_time=stage_start,
                end_time=stage_end,
                input_size_bytes=1024,
                output_size_bytes=1024
            )
        
        # End batch
        metrics.end_batch(batch_id, accuracy=90.0 + batch_id * 2.0)
    
    # Get summary
    summary = metrics.get_device_summary()
    logger.info(f"Device summary: {summary['images_per_second']:.2f} IPS")
    
    # Finalize
    results = metrics.finalize("test_model")
    logger.info(f"Metrics saved to: {results['csv_files']['device_metrics']}")


def test_end_to_end_processing():
    """Test end-to-end processing without RPC."""
    logger.info("=== Testing End-to-End Processing ===")
    
    # Load model and create dummy data
    model_loader = ModelLoader("./models")
    model = model_loader.load_model("mobilenetv2", num_classes=10)
    
    # Create dummy dataset
    dummy_loader = model_loader.load_dataset("dummy", "mobilenetv2", batch_size=4)
    
    # Get a batch
    images, labels = next(iter(dummy_loader))
    logger.info(f"Processing batch: {images.shape}")
    
    # Run inference
    start_time = time.time()
    with torch.no_grad():
        output = model(images)
    end_time = time.time()
    
    # Calculate accuracy (random for dummy data)
    _, predicted = torch.max(output.data, 1)
    accuracy = (predicted == labels).sum().item() / len(labels) * 100.0
    
    logger.info(f"Inference completed in {(end_time - start_time)*1000:.2f}ms")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Accuracy: {accuracy:.2f}%")


def main():
    """Run all tests."""
    logger.info("Starting Enhanced Distributed Inference Tests")
    
    try:
        # Test 1: Model loading
        model, sample_input = test_model_loading()
        
        # Test 2: Profiling
        profile = test_profiling(model, sample_input)
        
        # Test 3: Intelligent splitting
        shards = test_intelligent_splitting(model, profile)
        
        # Test 4: Enhanced metrics
        test_enhanced_metrics()
        
        # Test 5: End-to-end processing
        test_end_to_end_processing()
        
        logger.info("=== All Tests Completed Successfully ===")
        logger.info("The enhanced distributed inference system is working correctly!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)