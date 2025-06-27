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
                 models_dir: str = "."):
        super().__init__()
        self.model_type = model_type
        self.num_splits = num_splits
        self.workers = workers
        self.num_classes = num_classes
        self.metrics_collector = metrics_collector
        self.use_intelligent_splitting = use_intelligent_splitting
        self.use_pipelining = use_pipelining
        self.models_dir = models_dir
        
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
        self.model_profile.save_to_json(profile_path)
        
        self.logger.info(f"Model profile saved to: {profile_path}")
        self.logger.info(f"Total model execution time: {self.model_profile.total_time_ms:.2f}ms")
        self.logger.info(f"Total model parameters: {self.model_profile.total_parameters:,}")
    
    def _split_model(self) -> List[nn.Module]:
        """Split the model using intelligent or traditional methods."""
        if self.use_intelligent_splitting and self.model_profile:
            self.logger.info("Using intelligent splitting based on profiling data")
            
            # Use intelligent splitter
            shards, self.split_config = split_model_intelligently(
                self.original_model, 
                self.model_profile, 
                self.num_splits,
                network_config={
                    'communication_latency_ms': 5.0,  # Estimate for Pi network
                    'network_bandwidth_mbps': 100.0   # Estimate for WiFi
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
            return self.pipeline_manager.process_batch_rpc(x)
        else:
            # Sequential execution (original approach)
            return self._forward_sequential(x, batch_id)
    
    def _forward_sequential(self, x: torch.Tensor, batch_id: Optional[int] = None) -> torch.Tensor:
        """Sequential forward pass (non-pipelined)."""
        current_tensor = x
        
        for i, shard_rref in enumerate(self.worker_rrefs):
            self.logger.debug(f"Processing through shard {i}")
            
            # Measure RPC latency and throughput
            start_time = time.time()
            current_tensor = shard_rref.rpc_sync().forward(current_tensor)
            end_time = time.time()
            
            # Record network metrics
            if self.metrics_collector:
                rpc_latency_ms = (end_time - start_time) * 1000
                data_size_bytes = current_tensor.numel() * current_tensor.element_size()
                throughput_mbps = (data_size_bytes / (1024 * 1024)) / (rpc_latency_ms / 1000) if rpc_latency_ms > 0 else 0
                
                self.metrics_collector.record_network_metrics(rpc_latency_ms, throughput_mbps)
        
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
                          models_dir: str = "."):
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
                models_dir=models_dir
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
    logger.info(f"Network latency: {device_summary.get('avg_network_latency_ms', 0):.2f}ms")
    logger.info(f"Network throughput: {device_summary.get('avg_throughput_mbps', 0):.2f}Mbps")


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
        num_splits=args.num_partitions,
        metrics_dir=args.metrics_dir,
        use_intelligent_splitting=args.use_intelligent_splitting,
        use_pipelining=args.use_pipelining,
        num_threads=args.num_threads,
        models_dir=args.models_dir
    )


if __name__ == "__main__":
    main()