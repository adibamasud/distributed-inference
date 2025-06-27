#!/usr/bin/env python3
"""
Enhanced layer-by-layer profiling system that captures both named modules and functional operations.
This module profiles individual layers AND functional operations (like adaptive_avg_pool2d, flatten)
to determine accurate computational costs, memory usage, and optimal split points for distributed inference.
"""

import time
import torch
import torch.nn as nn
import torch.profiler
import psutil
import logging
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class LayerProfile:
    """Profile data for a single layer or functional operation."""
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
    """Complete profile for a model including functional operations."""
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
    
    def save_to_file(self, filepath: str):
        """Save profile to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class LayerProfiler:
    """Enhanced profiler that captures both named modules and functional operations."""
    
    def __init__(self, device: str = "cpu", warmup_iterations: int = 3, profile_iterations: int = 10):
        """
        Initialize the enhanced profiler.
        
        Args:
            device: Device to run profiling on ("cpu" or "cuda")
            warmup_iterations: Number of warmup iterations
            profile_iterations: Number of profiling iterations for averaging
        """
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.profile_iterations = profile_iterations
        self.logger = logging.getLogger(__name__)
        
        # Model-specific functional operation definitions
        self.model_functional_ops = {
            'mobilenetv2': self._get_mobilenetv2_functional_ops,
            'mobilenet_v2': self._get_mobilenetv2_functional_ops,
            'resnet18': self._get_resnet_functional_ops,
            'resnet': self._get_resnet_functional_ops,
        }
    
    def _get_mobilenetv2_functional_ops(self) -> List[Dict[str, Any]]:
        """Define functional operations for MobileNetV2."""
        return [
            {
                'name': 'adaptive_avg_pool2d',
                'layer_name': 'functional.adaptive_avg_pool2d',
                'operation': lambda x: torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)),
                'insert_after': 'features.18.2',  # Last ReLU6 in features
                'flops_calc': lambda inp, out: int(inp.numel())  # Read each input element once
            },
            {
                'name': 'flatten',
                'layer_name': 'functional.flatten',
                'operation': lambda x: torch.flatten(x, 1),
                'insert_after': 'functional.adaptive_avg_pool2d',
                'flops_calc': lambda inp, out: 0  # Reshape operation, no FLOPs
            }
        ]
    
    def _get_resnet_functional_ops(self) -> List[Dict[str, Any]]:
        """Define functional operations for ResNet models."""
        return [
            {
                'name': 'adaptive_avg_pool2d',
                'layer_name': 'functional.adaptive_avg_pool2d',
                'operation': lambda x: torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)),
                'insert_after': 'layer4.1.relu',  # Last ReLU in ResNet18
                'flops_calc': lambda inp, out: int(inp.numel())
            },
            {
                'name': 'flatten',
                'layer_name': 'functional.flatten', 
                'operation': lambda x: torch.flatten(x, 1),
                'insert_after': 'functional.adaptive_avg_pool2d',
                'flops_calc': lambda inp, out: 0
            }
        ]
    
    def _calculate_flops(self, module: nn.Module, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> int:
        """Calculate FLOPs for a given module."""
        if isinstance(module, nn.Conv2d):
            # Conv2D: FLOPs = batch_size * output_height * output_width * kernel_height * kernel_width * input_channels * output_channels
            batch_size, in_channels, in_height, in_width = input_tensor.shape
            out_channels, out_height, out_width = output_tensor.shape[1], output_tensor.shape[2], output_tensor.shape[3]
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * in_channels
            output_elements = batch_size * out_height * out_width * out_channels
            flops = kernel_ops * output_elements + (output_elements if module.bias is not None else 0)
            return int(flops)
            
        elif isinstance(module, nn.Linear):
            # Linear: FLOPs = batch_size * input_features * output_features + bias_term
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
        """Profile a single layer or module."""
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
        
        # Calculate averages
        avg_execution_time = sum(execution_times) / len(execution_times)
        avg_memory_usage = sum(memory_usages) / len(memory_usages)
        avg_cpu_utilization = sum(cpu_utilizations) / len(cpu_utilizations)
        
        # Calculate FLOPs
        with torch.no_grad():
            sample_output = module(input_tensor)
        flops = self._calculate_flops(module, input_tensor, sample_output)
        
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
    
    def profile_functional_operation(self, operation_func: Callable, input_tensor: torch.Tensor, 
                                   operation_name: str, layer_name: str, flops_calc: Callable = None) -> LayerProfile:
        """Profile a functional operation."""
        input_tensor = input_tensor.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = operation_func(input_tensor)
        
        # Profile execution time
        execution_times = []
        
        for _ in range(self.profile_iterations):
            if self.device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = operation_func(input_tensor)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            execution_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_execution_time = sum(execution_times) / len(execution_times)
        
        # Calculate FLOPs
        if flops_calc:
            flops = flops_calc(input_tensor, output)
        else:
            flops = int(output.numel())  # Conservative estimate
        
        return LayerProfile(
            layer_name=layer_name,
            layer_type=operation_name,
            execution_time_ms=avg_execution_time,
            memory_usage_mb=0.0,  # Functional operations don't have learnable parameters
            flops=flops,
            input_shape=tuple(input_tensor.shape),
            output_shape=tuple(output.shape),
            parameters=0,  # No learnable parameters
            cpu_utilization=0.0  # Skip CPU measurement for functional ops
        )
    
    def profile_model(self, model: nn.Module, sample_input: torch.Tensor, model_name: str) -> ModelProfile:
        """Profile all layers and functional operations in a model."""
        self.logger.info(f"Starting enhanced profiling for model: {model_name}")
        
        model.eval()
        layer_profiles = []
        
        # Step 1: Profile all named modules with hooks
        intermediate_outputs = self._collect_intermediate_outputs(model, sample_input)
        
        # Step 2: Profile each named module
        for name, module in model.named_modules():
            if len(list(module.children())) == 0 and name in intermediate_outputs:  # Leaf modules only
                input_tensor, output_tensor = intermediate_outputs[name]
                
                try:
                    profile = self.profile_layer(module, input_tensor, name)
                    layer_profiles.append(profile)
                    self.logger.info(f"Profiled layer {name}: {profile.execution_time_ms:.2f}ms, {profile.memory_usage_mb:.2f}MB")
                except Exception as e:
                    self.logger.warning(f"Failed to profile layer {name}: {e}")
        
        # Step 3: Add missing functional operations
        layer_profiles = self._add_functional_operations(model, layer_profiles, sample_input, model_name)
        
        # Calculate totals
        total_time = sum(lp.execution_time_ms for lp in layer_profiles)
        total_memory = sum(lp.memory_usage_mb for lp in layer_profiles)
        total_flops = sum(lp.flops for lp in layer_profiles)
        total_parameters = sum(lp.parameters for lp in layer_profiles)
        
        self.logger.info(f"Completed enhanced profiling for {model_name}: {len(layer_profiles)} operations profiled")
        
        return ModelProfile(
            model_name=model_name,
            layer_profiles=layer_profiles,
            total_time_ms=total_time,
            total_memory_mb=total_memory,
            total_flops=total_flops,
            total_parameters=total_parameters
        )
    
    def _collect_intermediate_outputs(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Collect intermediate outputs from all named modules."""
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
        
        return intermediate_outputs
    
    def _add_functional_operations(self, model: nn.Module, layer_profiles: List[LayerProfile], 
                                 sample_input: torch.Tensor, model_name: str) -> List[LayerProfile]:
        """Add missing functional operations based on model architecture."""
        model_key = model_name.lower()
        
        # Find matching functional operations definition
        functional_ops_getter = None
        for key, getter in self.model_functional_ops.items():
            if key in model_key:
                functional_ops_getter = getter
                break
        
        if not functional_ops_getter:
            self.logger.info(f"No functional operations defined for {model_name}")
            return layer_profiles
        
        functional_ops = functional_ops_getter()
        
        # Build a map of layer names to indices
        layer_name_to_idx = {profile.layer_name: i for i, profile in enumerate(layer_profiles)}
        
        # Insert functional operations in reverse order to maintain indices
        for func_op in reversed(functional_ops):
            insert_after = func_op['insert_after']
            
            # Find insertion point
            insert_idx = -1
            if insert_after in layer_name_to_idx:
                insert_idx = layer_name_to_idx[insert_after] + 1
            else:
                # Try to find by partial match
                for layer_name, idx in layer_name_to_idx.items():
                    if insert_after in layer_name:
                        insert_idx = idx + 1
                        break
            
            if insert_idx > 0:
                # Get input tensor for this operation
                if insert_after.startswith('functional.'):
                    # Previous functional operation - use its output shape
                    prev_layer = layer_profiles[insert_idx - 1]
                    input_tensor = torch.randn(*prev_layer.output_shape)
                else:
                    # Named module - use its output
                    prev_layer = layer_profiles[insert_idx - 1]
                    input_tensor = torch.randn(*prev_layer.output_shape)
                
                # Profile the functional operation
                try:
                    func_profile = self.profile_functional_operation(
                        operation_func=func_op['operation'],
                        input_tensor=input_tensor,
                        operation_name=func_op['name'],
                        layer_name=func_op['layer_name'],
                        flops_calc=func_op['flops_calc']
                    )
                    
                    # Insert the functional operation
                    layer_profiles.insert(insert_idx, func_profile)
                    
                    # Update the layer name to index mapping
                    layer_name_to_idx = {profile.layer_name: i for i, profile in enumerate(layer_profiles)}
                    
                    self.logger.info(f"Added functional operation: {func_op['layer_name']} after {insert_after}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to profile functional operation {func_op['name']}: {e}")
        
        return layer_profiles


def profile_model_enhanced(model: nn.Module, sample_input: torch.Tensor, model_name: str,
                         device: str = "cpu", save_path: str = None) -> ModelProfile:
    """
    Convenience function to profile a model with enhanced functional operation detection.
    
    Args:
        model: PyTorch model to profile
        sample_input: Sample input tensor
        model_name: Name of the model
        device: Device to run profiling on
        save_path: Optional path to save the profile
    
    Returns:
        ModelProfile with complete profiling data
    """
    profiler = LayerProfiler(device=device)
    profile = profiler.profile_model(model, sample_input, model_name)
    
    if save_path:
        profile.save_to_file(save_path)
    
    return profile


if __name__ == "__main__":
    # Example usage
    import torchvision.models as models
    
    logging.basicConfig(level=logging.INFO)
    
    # Test with MobileNetV2
    model = models.mobilenet_v2(weights=None)
    sample_input = torch.randn(1, 3, 224, 224)
    
    profile = profile_model_enhanced(model, sample_input, "mobilenetv2", save_path="test_profile.json")
    print(f"Enhanced profiling complete: {len(profile.layer_profiles)} operations profiled")