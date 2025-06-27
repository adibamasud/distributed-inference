#!/usr/bin/env python3
"""
Basic functionality test for enhanced distributed inference.
Tests core components without complex model splitting.
"""

import sys
import os
sys.path.append('.')
sys.path.append('..')

import torch
import logging
from profiling import LayerProfiler
from metrics import EnhancedMetricsCollector
from core import ModelLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_core_functionality():
    """Test core functionality without complex splitting."""
    logger.info("=== Testing Core Enhanced Functionality ===")
    
    # 1. Test Model Loading
    logger.info("1. Testing Model Loading...")
    model_loader = ModelLoader("./models")
    model = model_loader.load_model("mobilenetv2", num_classes=10)
    sample_input = model_loader.get_sample_input("mobilenetv2", batch_size=1)
    
    with torch.no_grad():
        output = model(sample_input)
    logger.info(f"✓ Model loading successful - Output shape: {output.shape}")
    
    # 2. Test Profiling (abbreviated)
    logger.info("2. Testing Layer Profiling...")
    profiler = LayerProfiler(device="cpu", warmup_iterations=1, profile_iterations=1)
    profile = profiler.profile_model(model, sample_input, "test_model")
    logger.info(f"✓ Profiling successful - {len(profile.layer_profiles)} layers, {profile.total_time_ms:.2f}ms total")
    
    # 3. Test Enhanced Metrics
    logger.info("3. Testing Enhanced Metrics...")
    metrics = EnhancedMetricsCollector(rank=0, output_dir="./test_basic_metrics")
    
    # Simulate processing
    import time
    metrics.start_batch(0, batch_size=1)
    time.sleep(0.01)
    metrics.record_pipeline_stage(0, 0, "test_stage", time.time()-0.01, time.time())
    metrics.end_batch(0, accuracy=95.0)
    
    summary = metrics.get_device_summary()
    logger.info(f"✓ Metrics collection successful - {summary['images_per_second']:.2f} IPS")
    
    metrics.finalize("test_model")
    
    # 4. Test Dataset Loading
    logger.info("4. Testing Dataset Loading...")
    dummy_loader = model_loader.load_dataset("dummy", "mobilenetv2", batch_size=2)
    images, labels = next(iter(dummy_loader))
    logger.info(f"✓ Dataset loading successful - Batch shape: {images.shape}")
    
    # 5. Test End-to-End Inference
    logger.info("5. Testing End-to-End Inference...")
    start_time = time.time()
    with torch.no_grad():
        output = model(images)
    end_time = time.time()
    
    _, predicted = torch.max(output.data, 1)
    accuracy = (predicted == labels).sum().item() / len(labels) * 100.0
    logger.info(f"✓ End-to-end inference successful - {(end_time-start_time)*1000:.2f}ms, {accuracy:.1f}% accuracy")
    
    return True


def test_traditional_splitting():
    """Test using the traditional splitting approach."""
    logger.info("=== Testing Traditional Model Splitting ===")
    
    # Load model
    model_loader = ModelLoader("./models")
    model = model_loader.load_model("mobilenetv2", num_classes=10)
    
    # Manually create simple splits for testing
    # Split into features and classifier
    features = model.features
    classifier = model.classifier
    
    # Test the split
    sample_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        # Process through features
        intermediate = features(sample_input)
        logger.info(f"Features output shape: {intermediate.shape}")
        
        # Process through classifier
        # Add necessary operations (adaptive pooling, flatten)
        pooled = torch.nn.functional.adaptive_avg_pool2d(intermediate, (1, 1))
        flattened = torch.flatten(pooled, 1)
        final_output = classifier(flattened)
        
        logger.info(f"Final output shape: {final_output.shape}")
    
    logger.info("✓ Traditional splitting test successful")
    return True


def test_metrics_csv_output():
    """Test that metrics are properly written to CSV files."""
    logger.info("=== Testing Metrics CSV Output ===")
    
    metrics = EnhancedMetricsCollector(rank=0, output_dir="./test_csv_metrics")
    
    # Simulate multiple batches
    import time
    for batch_id in range(3):
        metrics.start_batch(batch_id, batch_size=2)
        
        # Simulate pipeline stages
        for stage_id in range(2):
            stage_start = time.time()
            time.sleep(0.01)
            stage_end = time.time()
            metrics.record_pipeline_stage(
                batch_id, stage_id, f"stage_{stage_id}", 
                stage_start, stage_end, 1024, 1024
            )
        
        # Record network metrics
        metrics.record_network_metrics(5.0, 100.0)
        
        metrics.end_batch(batch_id, accuracy=85.0 + batch_id * 2.0)
    
    # Finalize and check files
    results = metrics.finalize("test_csv_model")
    
    # Check that CSV files exist
    for csv_type, csv_path in results['csv_files'].items():
        if os.path.exists(csv_path):
            logger.info(f"✓ {csv_type} CSV file created: {csv_path}")
            # Check file size
            size = os.path.getsize(csv_path)
            logger.info(f"  File size: {size} bytes")
        else:
            logger.error(f"✗ {csv_type} CSV file missing: {csv_path}")
            return False
    
    logger.info("✓ CSV output test successful")
    return True


def main():
    """Run all basic tests."""
    logger.info("Starting Basic Functionality Tests for Enhanced Distributed Inference")
    
    tests = [
        ("Core Functionality", test_core_functionality),
        ("Traditional Splitting", test_traditional_splitting),
        ("CSV Metrics Output", test_metrics_csv_output),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running Test: {test_name}")
        logger.info('='*50)
        
        try:
            success = test_func()
            if success:
                logger.info(f"✓ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"✗ {test_name} FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
            failed += 1
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed} PASSED, {failed} FAILED")
    logger.info('='*50)
    
    if failed == 0:
        logger.info("🎉 All basic functionality tests PASSED!")
        logger.info("The enhanced distributed inference system core components are working correctly.")
        return True
    else:
        logger.error(f"❌ {failed} test(s) failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)