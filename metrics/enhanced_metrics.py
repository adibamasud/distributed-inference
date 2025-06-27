#!/usr/bin/env python3
"""
Enhanced metrics collection system for distributed DNN inference.
Collects comprehensive metrics including per-device IPS, pipeline efficiency,
network latency, and throughput measurements.
"""

import time
import psutil
import socket
import csv
import os
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json


@dataclass
class PipelineStageMetrics:
    """Metrics for a single pipeline stage."""
    stage_id: int
    stage_name: str
    device_id: str
    start_time: float
    end_time: float
    processing_time_ms: float
    queue_wait_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    input_size_bytes: int
    output_size_bytes: int
    batch_id: Optional[int] = None


@dataclass  
class BatchMetrics:
    """Metrics for a complete batch processed through the pipeline."""
    batch_id: int
    batch_size: int
    start_time: float
    end_time: float
    total_time_ms: float
    pipeline_stages: List[PipelineStageMetrics] = field(default_factory=list)
    accuracy: Optional[float] = None
    
    def add_stage(self, stage_metrics: PipelineStageMetrics):
        """Add stage metrics to this batch."""
        self.pipeline_stages.append(stage_metrics)
    
    def get_pipeline_utilization(self) -> float:
        """Calculate pipeline utilization efficiency."""
        if not self.pipeline_stages:
            return 0.0
        
        total_processing_time = sum(stage.processing_time_ms for stage in self.pipeline_stages)
        return total_processing_time / self.total_time_ms if self.total_time_ms > 0 else 0.0


@dataclass
class DeviceMetrics:
    """Metrics for a specific device in the distributed system."""
    device_id: str
    hostname: str
    rank: int
    total_images_processed: int = 0
    total_processing_time_ms: float = 0.0
    images_per_second: float = 0.0
    average_processing_time_ms: float = 0.0
    memory_usage_history: List[float] = field(default_factory=list)
    cpu_usage_history: List[float] = field(default_factory=list)
    network_latency_history: List[float] = field(default_factory=list)
    throughput_history: List[float] = field(default_factory=list)
    
    def update_performance(self, processing_time_ms: float, images_count: int = 1):
        """Update performance metrics with new data."""
        self.total_images_processed += images_count
        self.total_processing_time_ms += processing_time_ms
        
        if self.total_processing_time_ms > 0:
            self.images_per_second = (self.total_images_processed * 1000.0) / self.total_processing_time_ms
            self.average_processing_time_ms = self.total_processing_time_ms / self.total_images_processed


class EnhancedMetricsCollector:
    """Enhanced metrics collector with comprehensive pipeline and device metrics."""
    
    # CSV headers for comprehensive metrics
    _BATCH_CSV_HEADERS = [
        'batch_id', 'batch_size', 'start_time', 'end_time', 'total_time_ms',
        'pipeline_utilization', 'accuracy', 'images_per_second', 'model_name',
        'num_devices', 'timestamp'
    ]
    
    _DEVICE_CSV_HEADERS = [
        'device_id', 'hostname', 'rank', 'total_images_processed', 
        'total_processing_time_ms', 'images_per_second', 'average_processing_time_ms',
        'avg_memory_usage_mb', 'avg_cpu_usage_percent', 'avg_network_latency_ms',
        'avg_throughput_mbps', 'model_name', 'timestamp'
    ]
    
    _PIPELINE_CSV_HEADERS = [
        'batch_id', 'stage_id', 'stage_name', 'device_id', 'start_time', 'end_time',
        'processing_time_ms', 'queue_wait_time_ms', 'memory_usage_mb', 
        'cpu_usage_percent', 'input_size_bytes', 'output_size_bytes', 'timestamp'
    ]
    
    def __init__(self, rank: int, output_dir: str = "./metrics", enable_realtime: bool = True):
        self.rank = rank
        self.hostname = socket.gethostname()
        self.device_id = f"{self.hostname}_rank_{rank}"
        self.output_dir = output_dir
        self.enable_realtime = enable_realtime
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize storage
        self.batch_metrics: List[BatchMetrics] = []
        self.device_metrics: Dict[str, DeviceMetrics] = {}
        self.pipeline_metrics: List[PipelineStageMetrics] = []
        
        # Initialize device metrics for this node
        self.device_metrics[self.device_id] = DeviceMetrics(
            device_id=self.device_id,
            hostname=self.hostname,
            rank=rank
        )
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.system_metrics_history = deque(maxlen=1000)
        
        # CSV file paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batch_csv_file = os.path.join(output_dir, f"batch_metrics_{self.device_id}_{timestamp}.csv")
        self.device_csv_file = os.path.join(output_dir, f"device_metrics_{self.device_id}_{timestamp}.csv")
        self.pipeline_csv_file = os.path.join(output_dir, f"pipeline_metrics_{self.device_id}_{timestamp}.csv")
        
        # Initialize CSV files with headers
        self._init_csv_files()
        
        self.logger = logging.getLogger(__name__)
        
        if self.enable_realtime:
            self.start_monitoring()
    
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        with open(self.batch_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self._BATCH_CSV_HEADERS)
        
        with open(self.device_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self._DEVICE_CSV_HEADERS)
        
        with open(self.pipeline_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self._PIPELINE_CSV_HEADERS)
    
    def start_monitoring(self):
        """Start real-time system monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitoring_thread.start()
        self.logger.info(f"Started real-time monitoring for {self.device_id}")
    
    def stop_monitoring(self):
        """Stop real-time system monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        self.logger.info(f"Stopped real-time monitoring for {self.device_id}")
    
    def _monitor_system(self):
        """Monitor system metrics in real-time."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                memory = psutil.virtual_memory()
                memory_used_mb = (memory.total - memory.available) / (1024 * 1024)
                cpu_percent = psutil.cpu_percent()
                
                # Store in history
                self.system_metrics_history.append({
                    'timestamp': time.time(),
                    'memory_mb': memory_used_mb,
                    'cpu_percent': cpu_percent
                })
                
                # Update device metrics
                if self.device_id in self.device_metrics:
                    device = self.device_metrics[self.device_id]
                    device.memory_usage_history.append(memory_used_mb)
                    device.cpu_usage_history.append(cpu_percent)
                
                time.sleep(0.5)  # Monitor every 500ms
                
            except Exception as e:
                self.logger.warning(f"Error in system monitoring: {e}")
    
    def start_batch(self, batch_id: int, batch_size: int) -> float:
        """Start tracking a new batch."""
        start_time = time.time()
        batch_metrics = BatchMetrics(
            batch_id=batch_id,
            batch_size=batch_size,
            start_time=start_time,
            end_time=0.0,
            total_time_ms=0.0
        )
        self.batch_metrics.append(batch_metrics)
        self.logger.debug(f"Started batch {batch_id} with {batch_size} images")
        return start_time
    
    def end_batch(self, batch_id: int, accuracy: Optional[float] = None) -> float:
        """End tracking for a batch."""
        end_time = time.time()
        
        # Find the batch
        batch = None
        for b in self.batch_metrics:
            if b.batch_id == batch_id:
                batch = b
                break
        
        if batch is None:
            self.logger.warning(f"Batch {batch_id} not found for end tracking")
            return end_time
        
        batch.end_time = end_time
        batch.total_time_ms = (end_time - batch.start_time) * 1000
        batch.accuracy = accuracy
        
        # Calculate images per second for this batch
        if batch.total_time_ms > 0:
            batch_ips = (batch.batch_size * 1000.0) / batch.total_time_ms
        else:
            batch_ips = 0.0
        
        # Update device metrics
        device = self.device_metrics[self.device_id]
        device.update_performance(batch.total_time_ms, batch.batch_size)
        
        self.logger.info(f"Completed batch {batch_id}: {batch.total_time_ms:.2f}ms, {batch_ips:.2f} IPS")
        
        # Write to CSV immediately
        self._write_batch_to_csv(batch, batch_ips)
        
        return end_time
    
    def record_pipeline_stage(self, batch_id: int, stage_id: int, stage_name: str, 
                            start_time: float, end_time: float, 
                            input_size_bytes: int = 0, output_size_bytes: int = 0,
                            queue_wait_time_ms: float = 0.0):
        """Record metrics for a pipeline stage."""
        processing_time_ms = (end_time - start_time) * 1000
        
        # Get current system metrics
        memory = psutil.virtual_memory()
        memory_used_mb = (memory.total - memory.available) / (1024 * 1024)
        cpu_percent = psutil.cpu_percent()
        
        stage_metrics = PipelineStageMetrics(
            stage_id=stage_id,
            stage_name=stage_name,
            device_id=self.device_id,
            start_time=start_time,
            end_time=end_time,
            processing_time_ms=processing_time_ms,
            queue_wait_time_ms=queue_wait_time_ms,
            memory_usage_mb=memory_used_mb,
            cpu_usage_percent=cpu_percent,
            input_size_bytes=input_size_bytes,
            output_size_bytes=output_size_bytes,
            batch_id=batch_id
        )
        
        self.pipeline_metrics.append(stage_metrics)
        
        # Add to corresponding batch
        for batch in self.batch_metrics:
            if batch.batch_id == batch_id:
                batch.add_stage(stage_metrics)
                break
        
        # Write to CSV immediately
        self._write_pipeline_stage_to_csv(stage_metrics)
        
        self.logger.debug(f"Recorded stage {stage_name} for batch {batch_id}: {processing_time_ms:.2f}ms")
    
    def record_network_metrics(self, latency_ms: float, throughput_mbps: float, 
                             device_id: Optional[str] = None):
        """Record network communication metrics."""
        target_device = device_id or self.device_id
        
        if target_device not in self.device_metrics:
            self.device_metrics[target_device] = DeviceMetrics(
                device_id=target_device,
                hostname="unknown",
                rank=-1
            )
        
        device = self.device_metrics[target_device]
        device.network_latency_history.append(latency_ms)
        device.throughput_history.append(throughput_mbps)
        
        self.logger.debug(f"Network metrics - Latency: {latency_ms:.2f}ms, Throughput: {throughput_mbps:.2f}Mbps")
    
    def _write_batch_to_csv(self, batch: BatchMetrics, ips: float):
        """Write batch metrics to CSV."""
        try:
            with open(self.batch_csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    batch.batch_id,
                    batch.batch_size,
                    batch.start_time,
                    batch.end_time,
                    batch.total_time_ms,
                    batch.get_pipeline_utilization(),
                    batch.accuracy or 0.0,
                    ips,
                    "",  # model_name - to be filled by caller
                    len(self.device_metrics),
                    datetime.now().isoformat()
                ])
        except Exception as e:
            self.logger.error(f"Error writing batch metrics: {e}")
    
    def _write_pipeline_stage_to_csv(self, stage: PipelineStageMetrics):
        """Write pipeline stage metrics to CSV."""
        try:
            with open(self.pipeline_csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    stage.batch_id,
                    stage.stage_id,
                    stage.stage_name,
                    stage.device_id,
                    stage.start_time,
                    stage.end_time,
                    stage.processing_time_ms,
                    stage.queue_wait_time_ms,
                    stage.memory_usage_mb,
                    stage.cpu_usage_percent,
                    stage.input_size_bytes,
                    stage.output_size_bytes,
                    datetime.now().isoformat()
                ])
        except Exception as e:
            self.logger.error(f"Error writing pipeline stage metrics: {e}")
    
    def get_device_summary(self, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive summary for a device."""
        target_device = device_id or self.device_id
        
        if target_device not in self.device_metrics:
            return {}
        
        device = self.device_metrics[target_device]
        
        # Calculate averages
        avg_memory = sum(device.memory_usage_history) / len(device.memory_usage_history) if device.memory_usage_history else 0.0
        avg_cpu = sum(device.cpu_usage_history) / len(device.cpu_usage_history) if device.cpu_usage_history else 0.0
        avg_latency = sum(device.network_latency_history) / len(device.network_latency_history) if device.network_latency_history else 0.0
        avg_throughput = sum(device.throughput_history) / len(device.throughput_history) if device.throughput_history else 0.0
        
        return {
            'device_id': device.device_id,
            'hostname': device.hostname,
            'rank': device.rank,
            'total_images_processed': device.total_images_processed,
            'total_processing_time_ms': device.total_processing_time_ms,
            'images_per_second': device.images_per_second,
            'average_processing_time_ms': device.average_processing_time_ms,
            'avg_memory_usage_mb': avg_memory,
            'avg_cpu_usage_percent': avg_cpu,
            'avg_network_latency_ms': avg_latency,
            'avg_throughput_mbps': avg_throughput
        }
    
    def write_device_summary_to_csv(self, model_name: str = ""):
        """Write device summary to CSV."""
        summary = self.get_device_summary()
        if not summary:
            return
        
        try:
            with open(self.device_csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    summary['device_id'],
                    summary['hostname'],
                    summary['rank'],
                    summary['total_images_processed'],
                    summary['total_processing_time_ms'],
                    summary['images_per_second'],
                    summary['average_processing_time_ms'],
                    summary['avg_memory_usage_mb'],
                    summary['avg_cpu_usage_percent'],
                    summary['avg_network_latency_ms'],
                    summary['avg_throughput_mbps'],
                    model_name,
                    datetime.now().isoformat()
                ])
        except Exception as e:
            self.logger.error(f"Error writing device summary: {e}")
    
    def get_pipeline_efficiency_stats(self) -> Dict[str, Any]:
        """Calculate pipeline efficiency statistics."""
        if not self.batch_metrics:
            return {}
        
        utilizations = [batch.get_pipeline_utilization() for batch in self.batch_metrics]
        avg_utilization = sum(utilizations) / len(utilizations)
        
        total_batches = len(self.batch_metrics)
        total_images = sum(batch.batch_size for batch in self.batch_metrics)
        total_time_ms = sum(batch.total_time_ms for batch in self.batch_metrics)
        
        overall_ips = (total_images * 1000.0) / total_time_ms if total_time_ms > 0 else 0.0
        
        # NEW: Calculate pipeline throughput using inter-batch completion times
        new_pipeline_throughput_ips, new_batch_throughput = self._calculate_new_pipeline_throughput()
        
        return {
            'total_batches': total_batches,
            'total_images': total_images,
            'total_time_ms': total_time_ms,
            'overall_images_per_second': overall_ips,
            'new_pipeline_throughput_ips': new_pipeline_throughput_ips,
            'new_pipeline_batch_throughput': new_batch_throughput,
            'average_pipeline_utilization': avg_utilization,
            'pipeline_stages_recorded': len(self.pipeline_metrics)
        }
    
    def _calculate_new_pipeline_throughput(self) -> tuple[float, float]:
        """
        Calculate NEW pipeline throughput using inter-batch completion times.
        Based on formula from image.png:
        - average inter-batch time = [(t2-t1) + (t3-t2) + (t4-t3)] / 3
        - batch throughput = 1 / average inter-batch time
        - image inference throughput = batch size * batch throughput
        """
        if len(self.batch_metrics) < 2:
            return 0.0, 0.0
        
        # Sort batches by end time to get completion order
        sorted_batches = sorted(self.batch_metrics, key=lambda b: b.end_time)
        
        # Calculate inter-batch completion times
        inter_batch_times = []
        for i in range(1, len(sorted_batches)):
            prev_batch = sorted_batches[i-1]
            curr_batch = sorted_batches[i]
            inter_batch_time_s = curr_batch.end_time - prev_batch.end_time
            inter_batch_times.append(inter_batch_time_s)
        
        if not inter_batch_times:
            return 0.0, 0.0
        
        # Calculate average inter-batch time
        avg_inter_batch_time_s = sum(inter_batch_times) / len(inter_batch_times)
        
        if avg_inter_batch_time_s <= 0:
            return 0.0, 0.0
        
        # Calculate NEW batch throughput (batches per second)
        new_batch_throughput = 1.0 / avg_inter_batch_time_s
        
        # Calculate NEW image inference throughput (images per second)
        # Use average batch size in case batch sizes vary
        avg_batch_size = sum(b.batch_size for b in self.batch_metrics) / len(self.batch_metrics)
        new_pipeline_throughput_ips = avg_batch_size * new_batch_throughput
        
        return new_pipeline_throughput_ips, new_batch_throughput
    
    def finalize(self, model_name: str = ""):
        """Finalize metrics collection and write summaries."""
        self.stop_monitoring()
        
        # Write final device summary
        self.write_device_summary_to_csv(model_name)
        
        # Log final statistics
        efficiency_stats = self.get_pipeline_efficiency_stats()
        device_summary = self.get_device_summary()
        
        self.logger.info(f"=== Final Metrics Summary for {self.device_id} ===")
        self.logger.info(f"Total images processed: {device_summary.get('total_images_processed', 0)}")
        self.logger.info(f"Images per second: {device_summary.get('images_per_second', 0):.2f}")
        self.logger.info(f"NEW Throughput (inter-batch): {efficiency_stats.get('new_pipeline_throughput_ips', 0):.2f} images/sec")
        self.logger.info(f"Average processing time: {device_summary.get('average_processing_time_ms', 0):.2f}ms")
        self.logger.info(f"Pipeline utilization: {efficiency_stats.get('average_pipeline_utilization', 0):.2f}")
        self.logger.info(f"Metrics saved to: {self.output_dir}")
        
        return {
            'device_summary': device_summary,
            'efficiency_stats': efficiency_stats,
            'csv_files': {
                'batch_metrics': self.batch_csv_file,
                'device_metrics': self.device_csv_file,
                'pipeline_metrics': self.pipeline_csv_file
            }
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    print("Enhanced metrics collector module loaded successfully")