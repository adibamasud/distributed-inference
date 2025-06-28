#!/usr/bin/env python3
"""
Enhanced Distributed DNN Inference with Intelligent Splitting and Pipelining

This script provides advanced distributed deep neural network inference capabilities:
- Layer-by-layer profiling for optimal split point detection
- Intelligent model splitting based on computational costs
- True sequential pipelining for overlapping inference stages
- Comprehensive metrics collection including per-device IPS and pipeline efficiency
"""

# Disable PyTorch's advanced CPU optimizations for Raspberry Pi compatibility
import os
os.environ['ATEN_CPU_CAPABILITY'] = ''

import time
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from dotenv import load_dotenv
import argparse
import logging
import socket
import sys
from typing import List, Dict, Any, Optional
import json

# Import our enhanced modules
from profiling import LayerProfiler, IntelligentSplitter, split_model_intelligently
from metrics import EnhancedMetricsCollector
from pipelining import PipelineManager, DistributedPipelineWorker
from core import ModelLoader


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


class EnhancedShardWrapper(nn.Module):
    """Enhanced shard wrapper with metrics integration."""
    
    def __init__(self, submodule: nn.Module, shard_id: int, 
                 metrics_collector: Optional[EnhancedMetricsCollector] = None):
        super().__init__()
        self.module = submodule.to("cpu")
        self.shard_id = shard_id
        self.metrics_collector = metrics_collector
        
        # Create metrics collector if none provided (for workers)
        if self.metrics_collector is None:
            rank = rpc.get_worker_info().id if rpc.is_available() else 0
            self.metrics_collector = EnhancedMetricsCollector(rank)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with enhanced metrics collection."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor but got {type(x)}")
        
        logging.info(f"[{socket.gethostname()}] Shard {self.shard_id} processing tensor shape: {x.shape}")
        
        x = x.to("cpu")
        start_time = time.time()
        
        # Forward pass
        with torch.no_grad():
            output = self.module(x).cpu()
        
        end_time = time.time()
        
        # Record stage metrics
        if self.metrics_collector:
            self.metrics_collector.record_pipeline_stage(
                batch_id=0,  # This would need to be passed from the pipeline
                stage_id=self.shard_id,
                stage_name=f"shard_{self.shard_id}",
                start_time=start_time,
                end_time=end_time,
                input_size_bytes=x.numel() * x.element_size(),
                output_size_bytes=output.numel() * output.element_size()
            )
        
        logging.info(f"[{socket.gethostname()}] Shard {self.shard_id} completed: {output.shape}")
        return output
    
    def parameter_rrefs(self):
        """Get parameter RRefs for distributed training (if needed)."""
        return [RRef(p) for p in self.parameters()]


class EnhancedDistributedModel(nn.Module):
    """Enhanced distributed model with intelligent splitting and pipelining."""
    
    def __init__(self, model_type: str, num_splits: int, workers: List[str],
                 num_classes: int = 10, metrics_collector: Optional[EnhancedMetricsCollector] = None,
                 use_intelligent_splitting: bool = True, use_pipelining: bool = False,
                 models_dir: str = ".", split_block: Optional[int] = None):
        super().__init__()
        self.model_type = model_type
        self.num_splits = num_splits
        self.workers = workers
        self.num_classes = num_classes
        self.metrics_collector = metrics_collector
        self.use_intelligent_splitting = use_intelligent_splitting
        self.use_pipelining = use_pipelining
        self.models_dir = models_dir
        self.split_block = split_block
        
        self.logger = logging.getLogger(__name__)
        
        # Load model
        model_loader = ModelLoader(models_dir)
        self.original_model = model_loader.load_model(model_type, num_classes)
        
        # Profile model if using intelligent splitting
        self.model_profile = None
        if self.use_intelligent_splitting:
            self._profile_model(model_loader)
        
        # Split model
        self.shards = self._split_model()
        
        # Deploy shards to workers
        self.worker_rrefs = self._deploy_shards()
        
        # Setup pipeline if enabled
        self.pipeline_manager = None
        if self.use_pipelining:
            self._setup_pipeline()
    
    def _profile_model(self, model_loader: ModelLoader):
        """Profile the model for intelligent splitting."""
        self.logger.info(f"Profiling model: {self.model_type}")
        
        # Get sample input
        sample_input = model_loader.get_sample_input(self.model_type, batch_size=1)
        
        # Profile the model
        profiler = LayerProfiler(device="cpu", warmup_iterations=2, profile_iterations=5)
        self.model_profile = profiler.profile_model(self.original_model, sample_input, self.model_type)
        
        # Save profile for analysis
        profile_path = f"./profiles/{self.model_type}_profile.json"
        os.makedirs("./profiles", exist_ok=True)
        self.model_profile.save_to_file(profile_path)
        
        self.logger.info(f"Model profile saved to: {profile_path}")
        self.logger.info(f"Total model execution time: {self.model_profile.total_time_ms:.2f}ms")
        self.logger.info(f"Total model parameters: {self.model_profile.total_parameters:,}")
    
    def _split_model(self) -> List[nn.Module]:
        """Split the model using intelligent, block-level, or traditional methods."""
        if self.use_intelligent_splitting and self.model_profile:
            # Try block-level splitting first (like reference implementation)
            if hasattr(self.original_model, 'features') and hasattr(self.original_model, 'classifier'):
                self.logger.info("Using block-level splitting (reference implementation style)")
                return self._split_model_block_level()
            else:
                self.logger.info("Using intelligent splitting based on profiling data")
                
                # Use intelligent splitter
                shards, self.split_config = split_model_intelligently(
                    self.original_model, 
                    self.model_profile, 
                    self.num_splits,
                    network_config={
                        'communication_latency_ms': 200.0,  # Realistic RPC + serialization latency
                        'network_bandwidth_mbps': 3.5       # Measured effective bandwidth between Pis
                    }
                )
                
                self.logger.info(f"Intelligent split created {len(shards)} shards")
                self.logger.info(f"Load balance score: {self.split_config.load_balance_score:.4f}")
                self.logger.info(f"Estimated communication overhead: {self.split_config.estimated_communication_overhead_ms:.2f}ms")
                
                return shards
        else:
            self.logger.info("Using traditional manual splitting")
            # Fall back to the original manual splitting from the base script
            return self._split_model_traditional()
    
    def _split_model_traditional(self) -> List[nn.Module]:
        """Traditional model splitting (from original script)."""
        # Import the original splitting function
        import sys
        sys.path.append('..')
        
        try:
            # This imports the function from the original script
            exec(open('../rpc_layer_split_with_metrics.py').read(), globals())
            return split_model_into_n_shards(self.original_model, self.num_splits)
        except Exception as e:
            self.logger.error(f"Failed to use traditional splitting: {e}")
            # Fallback: just return the whole model as one shard
            return [self.original_model]
    
    def _split_model_block_level(self) -> List[nn.Module]:
        """Block-level model splitting (like reference implementation)."""
        feature_blocks = list(self.original_model.features.children())
        total_blocks = len(feature_blocks)
        
        self.logger.info(f"Model has {total_blocks} feature blocks")
        
        # Calculate split point based on number of splits requested
        if self.split_block is not None:
            # Use user-specified split block
            split_at_block = self.split_block
            self.logger.info(f"Using user-specified split block: {split_at_block}")
        elif self.num_splits == 1:
            # For MobileNetV2: split at block 8 (like reference implementation)
            if self.model_type.lower() == 'mobilenetv2':
                split_at_block = 8
            else:
                # For other models, split roughly in the middle
                split_at_block = total_blocks // 2
        else:
            # For multiple splits, distribute blocks evenly
            split_at_block = total_blocks // (self.num_splits + 1)
        
        self.logger.info(f"Splitting at block {split_at_block} (reference style)")
        
        # Create shard 1: first part of features
        shard1_modules = feature_blocks[:split_at_block]
        shard1 = nn.Sequential(*shard1_modules)
        
        # Create shard 2: remaining features + pooling + classifier
        shard2_modules = feature_blocks[split_at_block:]
        shard2_modules.append(nn.AdaptiveAvgPool2d((1, 1)))
        shard2_modules.append(nn.Flatten())
        shard2_modules.append(self.original_model.classifier)
        shard2 = nn.Sequential(*shard2_modules)
        
        # Log partition details (for TODO item #2)
        shard1_params = sum(p.numel() for p in shard1.parameters())
        shard2_params = sum(p.numel() for p in shard2.parameters())
        total_params = shard1_params + shard2_params
        
        self.logger.info(f"Shard 1: Adding explicit AdaptiveAvgPool2d and Flatten for features→classifier transition")
        self.logger.info(f"Created 2 shards from block-level split")
        self.logger.info(f"Shard 1 parameters: {shard1_params:,} ({shard1_params/total_params*100:.1f}%)")
        self.logger.info(f"Shard 2 parameters: {shard2_params:,} ({shard2_params/total_params*100:.1f}%)")
        self.logger.info(f"Split ratio: Shard1={shard1_params/total_params*100:.1f}%, Shard2={shard2_params/total_params*100:.1f}%")
        
        return [shard1, shard2]
    
    def _deploy_shards(self) -> List[RRef]:
        """Deploy shards to worker nodes."""
        worker_rrefs = []
        
        for i, shard in enumerate(self.shards):
            worker_name = self.workers[i % len(self.workers)]
            
            # Create remote shard wrapper
            rref = rpc.remote(
                worker_name,
                EnhancedShardWrapper,
                args=(shard, i, None)  # Workers create their own metrics collectors
            )
            worker_rrefs.append(rref)
            
            self.logger.info(f"Deployed shard {i} to worker {worker_name}")
        
        return worker_rrefs
    
    def _setup_pipeline(self):
        """Setup pipeline manager for pipelined execution."""
        self.logger.info("Setting up pipeline for pipelined execution")
        
        # Create pipeline manager
        metrics_callback = None
        if self.metrics_collector:
            metrics_callback = self.metrics_collector.record_pipeline_stage
        
        self.pipeline_manager = PipelineManager(
            shards=self.shards,
            workers=self.workers,
            metrics_callback=metrics_callback,
            use_local_pipeline=False,  # Use RPC-based pipeline
            max_concurrent_batches=4
        )
    
    def forward(self, x: torch.Tensor, batch_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass through the distributed model."""
        if self.use_pipelining and self.pipeline_manager:
            # Use pipelined execution
            return self.pipeline_manager.process_batch_rpc_pipelined(x)
        else:
            # Sequential execution (original approach)
            return self._forward_sequential(x, batch_id)
    
    def _forward_sequential(self, x: torch.Tensor, batch_id: Optional[int] = None) -> torch.Tensor:
        """Sequential forward pass (non-pipelined)."""
        current_tensor = x
        
        for i, shard_rref in enumerate(self.worker_rrefs):
            self.logger.debug(f"Processing through shard {i}")
            
            # Measure RPC latency (includes computation)
            start_time = time.time()
            current_tensor = shard_rref.rpc_sync().forward(current_tensor)
            end_time = time.time()
            
            # Record RPC metrics (computation + network)
            if self.metrics_collector:
                rpc_total_ms = (end_time - start_time) * 1000
                
                # Estimate network overhead (serialization + transfer)
                # Based on analysis: ~14ms for 0.77MB on gigabit network
                tensor_size_mb = (current_tensor.numel() * current_tensor.element_size()) / (1024 * 1024)
                estimated_network_ms = 0.5 + (tensor_size_mb * 0.3) + (tensor_size_mb * 8 / 940) * 1000 * 2  # RTT + serialize + transfer
                
                # The rest is computation time
                estimated_computation_ms = max(0, rpc_total_ms - estimated_network_ms)
                
                # Record as "RPC latency" not "network latency" 
                self.metrics_collector.record_network_metrics(rpc_total_ms, estimated_network_ms)
        
        return current_tensor
    
    def parameter_rrefs(self):
        """Get parameter RRefs from all shards."""
        remote_params = []
        for rref in self.worker_rrefs:
            remote_params.extend(rref.remote().parameter_rrefs().to_here())
        return remote_params


# Global metrics collector for RPC access
global_metrics_collector = None

def collect_worker_summary(model_name: str, batch_size: int, num_parameters: int = 0) -> Dict[str, Any]:
    """RPC function to collect summary from workers."""
    global global_metrics_collector
    if global_metrics_collector:
        return global_metrics_collector.get_device_summary()
    return {}


def run_enhanced_inference(rank: int, world_size: int, model_type: str, batch_size: int,
                          num_classes: int, dataset: str, num_test_samples: int,
                          num_splits: int, metrics_dir: str, use_intelligent_splitting: bool = True,
                          use_pipelining: bool = False, num_threads: int = 4,
                          models_dir: str = ".", split_block: Optional[int] = None):
    """
    Run enhanced distributed inference with profiling and pipelining.
    """
    # Initialize enhanced metrics collector
    metrics_collector = EnhancedMetricsCollector(rank, metrics_dir, enable_realtime=True)
    
    # Make globally accessible for RPC
    global global_metrics_collector
    global_metrics_collector = metrics_collector
    
    # Setup logging with hostname and rank
    hostname = socket.gethostname()
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.hostname = hostname
        record.rank = rank
        return record
    logging.setLogRecordFactory(record_factory)
    
    for handler in logging.root.handlers:
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(hostname)s:rank%(rank)s] - %(message)s'
        ))
    
    logger = logging.getLogger(__name__)
    logger.info("Starting enhanced distributed inference")
    
    # Load environment variables
    load_dotenv()
    master_addr = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '29555')
    
    # Setup RPC environment
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Use network interface from .env file or defaults
    gloo_ifname = os.getenv('GLOO_SOCKET_IFNAME', 'eth0' if rank == 0 else 'eth0')
    os.environ['GLOO_SOCKET_IFNAME'] = gloo_ifname
    
    if rank == 0:
        os.environ['TENSORPIPE_SOCKET_IFADDR'] = '0.0.0.0'
    
    rpc_initialized = False
    
    if rank == 0:  # Master node
        logger.info("Initializing master node with enhanced features")
        logger.info(f"Loading dataset: {dataset}")
        try:
            # Initialize RPC
            rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
                num_worker_threads=num_threads,
                rpc_timeout=3600,
                init_method=f"tcp://0.0.0.0:{master_port}"
            )
            
            rpc.init_rpc("master", rank=rank, world_size=world_size, 
                        rpc_backend_options=rpc_backend_options)
            rpc_initialized = True
            logger.info("Master RPC initialized successfully")
            
            # Define workers
            workers = [f"worker{i}" for i in range(1, world_size)]
            
            # Create enhanced distributed model
            model = EnhancedDistributedModel(
                model_type=model_type,
                num_splits=num_splits,
                workers=workers,
                num_classes=num_classes,
                metrics_collector=metrics_collector,
                use_intelligent_splitting=use_intelligent_splitting,
                use_pipelining=use_pipelining,
                models_dir=models_dir,
                split_block=split_block
            )
            
            logger.info("Enhanced distributed model created successfully")
            
            # Load dataset
            model_loader = ModelLoader(models_dir)
            test_loader = model_loader.load_dataset(dataset, model_type, batch_size)
            
            logger.info(f"Dataset successfully loaded: {dataset} (batch_size={batch_size})")
            
            # Run inference with enhanced metrics
            logger.info("Starting inference with enhanced metrics collection...")
            start_time = time.time()
            
            total_images = 0
            num_correct = 0
            batch_count = 0
            
            if use_pipelining and model.use_pipelining and model.pipeline_manager:
                # Pipelined inference with multiple batches in flight
                logger.info("Using PIPELINED inference for maximum throughput")
                
                # Configuration for pipelining
                max_batches_in_flight = 3  # Limit memory usage
                active_batches = {}  # batch_id -> (images, labels, start_time)
                
                with torch.no_grad():
                    data_iter = iter(test_loader)
                    batch_id = 0
                    
                    # Keep pipeline full
                    while total_images < num_test_samples:
                        # Start new batches if we have capacity
                        while len(active_batches) < max_batches_in_flight and total_images < num_test_samples:
                            try:
                                images, labels = next(data_iter)
                            except StopIteration:
                                break
                            
                            # Trim batch if necessary
                            remaining = num_test_samples - total_images
                            if images.size(0) > remaining:
                                images = images[:remaining]
                                labels = labels[:remaining]
                            
                            # Explicit device placement
                            images = images.to("cpu")
                            labels = labels.to("cpu")
                            
                            # Start batch tracking
                            batch_start_time = metrics_collector.start_batch(batch_id, len(images))
                            
                            logger.info(f"Starting batch {batch_id + 1} with {len(images)} images (pipeline)")
                            
                            # Start batch processing asynchronously
                            pipeline_batch_id = model.pipeline_manager.start_batch_rpc_pipelined(images, labels)
                            active_batches[pipeline_batch_id] = (images, labels, batch_start_time, batch_id)
                            
                            total_images += len(images)
                            batch_id += 1
                        
                        # Collect completed batches
                        completed_ids = []
                        for pid, (orig_images, orig_labels, start_time, tracking_id) in list(active_batches.items()):
                            # Check if batch is complete (non-blocking)
                            result = model.pipeline_manager.get_completed_batch(pid, timeout=0.001)
                            if result is not None:
                                # Calculate accuracy
                                _, predicted = torch.max(result.data, 1)
                                batch_correct = (predicted == orig_labels).sum().item()
                                num_correct += batch_correct
                                
                                batch_accuracy = (batch_correct / len(orig_labels)) * 100.0
                                
                                # End batch tracking
                                metrics_collector.end_batch(tracking_id, accuracy=batch_accuracy)
                                
                                logger.info(f"Completed batch {tracking_id + 1} accuracy: {batch_accuracy:.2f}%")
                                completed_ids.append(pid)
                                batch_count += 1
                        
                        # Remove completed batches
                        for pid in completed_ids:
                            del active_batches[pid]
                        
                        # Small sleep if no completions to avoid busy waiting
                        if not completed_ids and len(active_batches) >= max_batches_in_flight:
                            time.sleep(0.01)
                    
                    # Wait for remaining batches
                    logger.info("Waiting for final batches to complete...")
                    while active_batches:
                        completed_ids = []
                        for pid, (orig_images, orig_labels, start_time, tracking_id) in list(active_batches.items()):
                            result = model.pipeline_manager.get_completed_batch(pid, timeout=0.1)
                            if result is not None:
                                # Calculate accuracy
                                _, predicted = torch.max(result.data, 1)
                                batch_correct = (predicted == orig_labels).sum().item()
                                num_correct += batch_correct
                                
                                batch_accuracy = (batch_correct / len(orig_labels)) * 100.0
                                
                                # End batch tracking
                                metrics_collector.end_batch(tracking_id, accuracy=batch_accuracy)
                                
                                logger.info(f"Completed batch {tracking_id + 1} accuracy: {batch_accuracy:.2f}%")
                                completed_ids.append(pid)
                                batch_count += 1
                        
                        for pid in completed_ids:
                            del active_batches[pid]
                
            else:
                # Original sequential inference
                logger.info("Using sequential inference")
                
                with torch.no_grad():
                    for i, (images, labels) in enumerate(test_loader):
                        if total_images >= num_test_samples:
                            break
                        
                        # Trim batch if necessary
                        remaining = num_test_samples - total_images
                        if images.size(0) > remaining:
                            images = images[:remaining]
                            labels = labels[:remaining]
                        
                        # Explicit device placement for consistency and predictability
                        images = images.to("cpu")
                        labels = labels.to("cpu")
                        
                        # Start batch tracking
                        batch_start_time = metrics_collector.start_batch(batch_count, len(images))
                        
                        logger.info(f"Processing batch {batch_count + 1} with {len(images)} images")
                        
                        # Run inference
                        output = model(images, batch_id=batch_count)
                        
                        # Calculate accuracy
                        _, predicted = torch.max(output.data, 1)
                        batch_correct = (predicted == labels).sum().item()
                        num_correct += batch_correct
                        total_images += len(images)
                        
                        batch_accuracy = (batch_correct / len(labels)) * 100.0
                        
                        # End batch tracking
                        metrics_collector.end_batch(batch_count, accuracy=batch_accuracy)
                        
                        logger.info(f"Batch {batch_count + 1} accuracy: {batch_accuracy:.2f}%")
                        batch_count += 1
            
            elapsed_time = time.time() - start_time
            final_accuracy = (num_correct / total_images) * 100.0 if total_images > 0 else 0.0
            overall_ips = total_images / elapsed_time if elapsed_time > 0 else 0.0
            
            logger.info(f"=== Enhanced Inference Results ===")
            logger.info(f"Total images processed: {total_images}")
            logger.info(f"Total time: {elapsed_time:.2f}s")
            logger.info(f"Final accuracy: {final_accuracy:.2f}%")
            logger.info(f"Overall throughput: {overall_ips:.2f} images/sec")
            
            # Collect worker summaries
            logger.info("Collecting enhanced metrics from workers...")
            worker_summaries = []
            
            for i in range(1, world_size):
                worker_name = f"worker{i}"
                try:
                    summary = rpc.rpc_sync(worker_name, collect_worker_summary,
                                         args=(model_type, batch_size, 0))
                    if summary:
                        worker_summaries.append(summary)
                        logger.info(f"Collected enhanced summary from {worker_name}")
                except Exception as e:
                    logger.warning(f"Failed to collect summary from {worker_name}: {e}")
            
            # Final metrics collection
            pipeline_stats = model.pipeline_manager.get_pipeline_stats() if model.pipeline_manager else {}
            
            logger.info("=== Pipeline Statistics ===")
            if pipeline_stats:
                logger.info(f"Pipeline utilization: {pipeline_stats.get('pipeline_utilization', 0):.2f}")
                logger.info(f"Active batches: {pipeline_stats.get('active_batches', 0)}")
            
        except Exception as e:
            logger.error(f"Error in enhanced master node: {e}", exc_info=True)
    
    else:  # Worker nodes
        logger.info(f"Initializing enhanced worker node with rank {rank}")
        retry_count = 0
        max_retries = 30
        connected = False
        
        while retry_count < max_retries and not connected:
            try:
                rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
                    num_worker_threads=num_threads,
                    rpc_timeout=3600,
                    init_method=f"tcp://{master_addr}:{master_port}"
                )
                
                rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size,
                           rpc_backend_options=rpc_backend_options)
                connected = rpc_initialized = True
                logger.info(f"Enhanced worker {rank} connected successfully")
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"Connection attempt {retry_count} failed: {e}")
                if retry_count >= max_retries:
                    logger.error(f"Worker {rank} failed to connect after {max_retries} attempts")
                    sys.exit(1)
                time.sleep(10 + (retry_count % 5))
    
    # Cleanup
    if rpc_initialized:
        logger.info("Shutting down RPC")
        try:
            rpc.shutdown()
            logger.info("RPC shutdown complete")
        except Exception as e:
            logger.error(f"Error during RPC shutdown: {e}")
    
    # Finalize metrics
    final_results = metrics_collector.finalize(model_type)
    
    logger.info("=== Final Enhanced Metrics Summary ===")
    device_summary = final_results['device_summary']
    efficiency_stats = final_results['efficiency_stats']
    
    logger.info(f"Images per second: {device_summary.get('images_per_second', 0):.2f}")
    logger.info(f"NEW Throughput (inter-batch): {efficiency_stats.get('new_pipeline_throughput_ips', 0):.2f} images/sec")
    logger.info(f"Average processing time: {device_summary.get('average_processing_time_ms', 0):.2f}ms")
    logger.info(f"Pipeline utilization: {efficiency_stats.get('average_pipeline_utilization', 0):.2f}")
    rpc_total = device_summary.get('avg_network_latency_ms', 0)
    network_overhead = device_summary.get('avg_throughput_mbps', 0)
    computation_time = rpc_total - network_overhead
    
    logger.info(f"RPC total time: {rpc_total:.2f}ms")
    logger.info(f"  - Network overhead: {network_overhead:.2f}ms")
    logger.info(f"  - Worker computation: {computation_time:.2f}ms")


def main():
    """Main function with enhanced argument parsing."""
    parser = argparse.ArgumentParser(description="Enhanced Distributed DNN Inference with Profiling and Pipelining")
    parser.add_argument("--rank", type=int, default=0, help="Rank of current process")
    parser.add_argument("--world-size", type=int, default=3, help="World size (1 master + N workers)")
    parser.add_argument("--model", type=str, default="mobilenetv2", 
                       choices=ModelLoader.list_supported_models(),
                       help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of output classes")
    parser.add_argument("--dataset", type=str, default="cifar10", 
                       choices=["cifar10", "dummy"], help="Dataset to use")
    parser.add_argument("--num-test-samples", type=int, default=64, help="Number of images to test")
    parser.add_argument("--num-partitions", type=int, default=2, help="Number of model partitions")
    parser.add_argument("--metrics-dir", type=str, default="./enhanced_metrics", 
                       help="Directory for enhanced metrics")
    parser.add_argument("--models-dir", type=str, default="./models", 
                       help="Directory containing model weight files")
    
    # Enhanced features
    parser.add_argument("--use-intelligent-splitting", action="store_true", default=True,
                       help="Use intelligent splitting based on profiling")
    parser.add_argument("--use-pipelining", action="store_true", default=False,
                       help="Enable pipelined execution")
    parser.add_argument("--num-threads", type=int, default=4, 
                       help="Number of RPC threads")
    parser.add_argument("--disable-intelligent-splitting", action="store_true",
                       help="Disable intelligent splitting (use traditional method)")
    parser.add_argument("--split-block", type=int, default=None,
                       help="Specific block number to split at (for MobileNetV2)")
    
    args = parser.parse_args()
    
    # Handle intelligent splitting flag
    if args.disable_intelligent_splitting:
        args.use_intelligent_splitting = False
    
    # Validation
    if args.world_size > 1 and args.num_partitions > args.world_size - 1:
        raise ValueError(f"Partitions ({args.num_partitions}) cannot exceed workers ({args.world_size - 1})")
    
    # Run enhanced inference
    run_enhanced_inference(
        rank=args.rank,
        world_size=args.world_size,
        model_type=args.model,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        dataset=args.dataset,
        num_test_samples=args.num_test_samples,
        num_splits=args.num_partitions - 1,  # Convert partitions to split points
        metrics_dir=args.metrics_dir,
        use_intelligent_splitting=args.use_intelligent_splitting,
        use_pipelining=args.use_pipelining,
        num_threads=args.num_threads,
        models_dir=args.models_dir,
        split_block=args.split_block
    )


if __name__ == "__main__":
    main()