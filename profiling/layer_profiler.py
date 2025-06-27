#!/usr/bin/env python3
"""
Layer-by-layer profiling system for optimal model splitting.
This module profiles individual layers to determine computational costs,
memory usage, and optimal split points for distributed inference.
"""

import time
import torch
import torch.nn as nn
import torch.profiler
import psutil
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class LayerProfile:
    """Profile data for a single layer."""
    layer_name: str
    layer_type: str
    execution_time_ms: float
    memory_usage_mb: float
    flops: int
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    parameters: int
    cpu_utilization: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'layer_name': self.layer_name,
            'layer_type': self.layer_type,
            'execution_time_ms': self.execution_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'flops': self.flops,
            'input_shape': list(self.input_shape),
            'output_shape': list(self.output_shape),
            'parameters': self.parameters,
            'cpu_utilization': self.cpu_utilization
        }


@dataclass
class ModelProfile:
    """Complete profile for a model."""
    model_name: str
    layer_profiles: List[LayerProfile]
    total_time_ms: float
    total_memory_mb: float
    total_flops: int
    total_parameters: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'layer_profiles': [lp.to_dict() for lp in self.layer_profiles],
            'total_time_ms': self.total_time_ms,
            'total_memory_mb': self.total_memory_mb,
            'total_flops': self.total_flops,
            'total_parameters': self.total_parameters
        }
    
    def save_to_json(self, filepath: str):
        """Save profile to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'ModelProfile':
        """Load profile from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        layer_profiles = []
        for lp_data in data['layer_profiles']:
            layer_profiles.append(LayerProfile(
                layer_name=lp_data['layer_name'],
                layer_type=lp_data['layer_type'],
                execution_time_ms=lp_data['execution_time_ms'],
                memory_usage_mb=lp_data['memory_usage_mb'],
                flops=lp_data['flops'],
                input_shape=tuple(lp_data['input_shape']),
                output_shape=tuple(lp_data['output_shape']),
                parameters=lp_data['parameters'],
                cpu_utilization=lp_data['cpu_utilization']
            ))
        
        return cls(
            model_name=data['model_name'],
            layer_profiles=layer_profiles,
            total_time_ms=data['total_time_ms'],
            total_memory_mb=data['total_memory_mb'],
            total_flops=data['total_flops'],
            total_parameters=data['total_parameters']
        )


class LayerProfiler:
    """Profiles individual layers to determine computational costs."""
    
    def __init__(self, device: str = "cpu", warmup_iterations: int = 3, profile_iterations: int = 10):
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.profile_iterations = profile_iterations
        self.logger = logging.getLogger(__name__)
        
    def _estimate_flops(self, module: nn.Module, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> int:
        """Estimate FLOPs for a layer based on its type and tensor shapes."""
        if isinstance(module, nn.Conv2d):
            # For Conv2d: FLOPs = output_elements * (kernel_size^2 * input_channels + bias_term)
            batch_size, out_channels, out_h, out_w = output_tensor.shape
            kernel_size = module.kernel_size[0] * module.kernel_size[1]
            in_channels = module.in_channels
            flops = batch_size * out_channels * out_h * out_w * (kernel_size * in_channels + (1 if module.bias is not None else 0))
            return int(flops)
            
        elif isinstance(module, nn.Linear):
            # For Linear: FLOPs = batch_size * input_features * output_features + bias_term
            batch_size = input_tensor.shape[0]
            flops = batch_size * module.in_features * module.out_features + (module.out_features if module.bias is not None else 0)
            return int(flops)
            
        elif isinstance(module, nn.BatchNorm2d):
            # BatchNorm: FLOPs ≈ 2 * num_elements (normalize + scale+shift)
            return int(2 * output_tensor.numel())
            
        elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.ELU, nn.LeakyReLU)):
            # Activation functions: FLOPs ≈ num_elements
            return int(output_tensor.numel())
            
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            # Pooling: FLOPs ≈ num_output_elements * kernel_area
            if hasattr(module, 'kernel_size'):
                if isinstance(module.kernel_size, int):
                    kernel_area = module.kernel_size ** 2
                else:
                    kernel_area = module.kernel_size[0] * module.kernel_size[1]
                return int(output_tensor.numel() * kernel_area)
            else:
                # AdaptiveAvgPool
                return int(output_tensor.numel())
        else:
            # Conservative estimate for unknown layers
            return int(output_tensor.numel())
    
    def profile_layer(self, module: nn.Module, input_tensor: torch.Tensor, layer_name: str) -> LayerProfile:
        """Profile a single layer."""
        module.eval()
        module = module.to(self.device)
        input_tensor = input_tensor.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = module(input_tensor)
        
        # Clear cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Profile execution time
        execution_times = []
        memory_usages = []
        cpu_utilizations = []
        
        for _ in range(self.profile_iterations):
            # Measure memory before
            memory_before = psutil.virtual_memory().used / (1024 * 1024)
            cpu_before = psutil.cpu_percent()
            
            # Time the execution
            if self.device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = module(input_tensor)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            # Measure memory after
            memory_after = psutil.virtual_memory().used / (1024 * 1024)
            cpu_after = psutil.cpu_percent()
            
            execution_times.append((end_time - start_time) * 1000)  # Convert to ms
            memory_usages.append(max(0, memory_after - memory_before))
            cpu_utilizations.append(max(0, cpu_after - cpu_before))
        
        # Calculate statistics
        avg_execution_time = sum(execution_times) / len(execution_times)
        avg_memory_usage = sum(memory_usages) / len(memory_usages)
        avg_cpu_utilization = sum(cpu_utilizations) / len(cpu_utilizations)
        
        # Estimate FLOPs
        with torch.no_grad():
            sample_output = module(input_tensor)
            flops = self._estimate_flops(module, input_tensor, sample_output)
        
        # Count parameters
        parameters = sum(p.numel() for p in module.parameters())
        
        return LayerProfile(
            layer_name=layer_name,
            layer_type=type(module).__name__,
            execution_time_ms=avg_execution_time,
            memory_usage_mb=avg_memory_usage,
            flops=flops,
            input_shape=tuple(input_tensor.shape),
            output_shape=tuple(sample_output.shape),
            parameters=parameters,
            cpu_utilization=avg_cpu_utilization
        )
    
    def profile_model(self, model: nn.Module, sample_input: torch.Tensor, model_name: str) -> ModelProfile:
        """Profile all layers in a model."""
        self.logger.info(f"Starting profiling for model: {model_name}")
        
        model.eval()
        layer_profiles = []
        
        # Register hooks to capture intermediate outputs
        intermediate_outputs = {}
        handles = []
        
        def make_hook(name):
            def hook_fn(module, input, output):
                intermediate_outputs[name] = (input[0].clone(), output.clone())
            return hook_fn
        
        # Register hooks for all named modules
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                handle = module.register_forward_hook(make_hook(name))
                handles.append(handle)
        
        # Run forward pass to collect intermediate outputs
        with torch.no_grad():
            _ = model(sample_input)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Profile each layer
        total_time = 0
        total_memory = 0
        total_flops = 0
        total_parameters = 0
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0 and name in intermediate_outputs:  # Leaf modules only
                input_tensor, _ = intermediate_outputs[name]
                
                try:
                    profile = self.profile_layer(module, input_tensor, name)
                    layer_profiles.append(profile)
                    
                    total_time += profile.execution_time_ms
                    total_memory += profile.memory_usage_mb
                    total_flops += profile.flops
                    total_parameters += profile.parameters
                    
                    self.logger.info(f"Profiled layer {name}: {profile.execution_time_ms:.2f}ms, {profile.memory_usage_mb:.2f}MB")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to profile layer {name}: {e}")
        
        self.logger.info(f"Completed profiling for {model_name}: {len(layer_profiles)} layers profiled")
        
        return ModelProfile(
            model_name=model_name,
            layer_profiles=layer_profiles,
            total_time_ms=total_time,
            total_memory_mb=total_memory,
            total_flops=total_flops,
            total_parameters=total_parameters
        )


def profile_model_architectures(models_info: List[Tuple[nn.Module, torch.Tensor, str]], 
                              output_dir: str = "./profiles",
                              device: str = "cpu") -> Dict[str, ModelProfile]:
    """Profile multiple model architectures and save results."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    profiler = LayerProfiler(device=device)
    profiles = {}
    
    for model, sample_input, model_name in models_info:
        try:
            profile = profiler.profile_model(model, sample_input, model_name)
            profiles[model_name] = profile
            
            # Save individual profile
            profile_path = os.path.join(output_dir, f"{model_name}_profile.json")
            profile.save_to_json(profile_path)
            logging.info(f"Saved profile for {model_name} to {profile_path}")
            
        except Exception as e:
            logging.error(f"Failed to profile {model_name}: {e}")
    
    return profiles


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # This would typically be called from the main inference script
    # with actual model instances
    print("Layer profiler module loaded successfully")