#!/usr/bin/env python3
"""
Detailed analysis of f16 model to identify remaining f32 parameters
"""

import torch
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

def analyze_model_dtypes(model_path):
    print(f"🔍 Analyzing model dtypes for: {model_path}")
    
    # Load model
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    
    dtype_stats = {}
    component_stats = {}
    f32_parameters = []
    
    total_params = 0
    total_memory = 0
    
    for name, param in model.named_parameters():
        dtype_str = str(param.dtype)
        component = name.split('.')[0] if '.' in name else name
        
        param_count = param.numel()
        param_memory = param.numel() * param.element_size()
        
        total_params += param_count
        total_memory += param_memory
        
        # Track by dtype
        if dtype_str not in dtype_stats:
            dtype_stats[dtype_str] = {'count': 0, 'memory': 0}
        dtype_stats[dtype_str]['count'] += param_count
        dtype_stats[dtype_str]['memory'] += param_memory
        
        # Track by component
        if component not in component_stats:
            component_stats[component] = {'dtypes': {}, 'total_params': 0, 'total_memory': 0}
        if dtype_str not in component_stats[component]['dtypes']:
            component_stats[component]['dtypes'][dtype_str] = {'count': 0, 'memory': 0}
        component_stats[component]['dtypes'][dtype_str]['count'] += param_count
        component_stats[component]['dtypes'][dtype_str]['memory'] += param_memory
        component_stats[component]['total_params'] += param_count
        component_stats[component]['total_memory'] += param_memory
        
        # Track f32 parameters specifically
        if param.dtype == torch.float32:
            f32_parameters.append((name, param.shape, param_count))
    
    # Also check buffers
    buffer_stats = {}
    f32_buffers = []
    
    for name, buffer in model.named_buffers():
        dtype_str = str(buffer.dtype)
        buffer_count = buffer.numel()
        buffer_memory = buffer.numel() * buffer.element_size()
        
        if dtype_str not in buffer_stats:
            buffer_stats[dtype_str] = {'count': 0, 'memory': 0}
        buffer_stats[dtype_str]['count'] += buffer_count
        buffer_stats[dtype_str]['memory'] += buffer_memory
        
        if buffer.dtype == torch.float32:
            f32_buffers.append((name, buffer.shape, buffer_count))
    
    # Print analysis
    print(f"\n📊 Overall Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Total memory: {total_memory / (1024**3):.2f} GB")
    
    print(f"\n📋 Parameter Dtypes:")
    for dtype, stats in sorted(dtype_stats.items()):
        percentage = (stats['count'] / total_params) * 100
        memory_gb = stats['memory'] / (1024**3)
        print(f"  {dtype}: {stats['count']:,} params ({percentage:.1f}%) - {memory_gb:.2f} GB")
    
    print(f"\n🏗️  Component Analysis:")
    for component, stats in sorted(component_stats.items()):
        print(f"  {component}:")
        print(f"    Total: {stats['total_params']:,} params - {stats['total_memory'] / (1024**3):.2f} GB")
        for dtype, dtype_stats in stats['dtypes'].items():
            percentage = (dtype_stats['count'] / stats['total_params']) * 100
            print(f"      {dtype}: {dtype_stats['count']:,} ({percentage:.1f}%)")
    
    if f32_parameters:
        print(f"\n🔍 F32 Parameters ({len(f32_parameters)} layers):")
        for name, shape, count in f32_parameters[:10]:  # Show first 10
            print(f"  {name}: {shape} - {count:,} params")
        if len(f32_parameters) > 10:
            print(f"  ... and {len(f32_parameters) - 10} more")
    else:
        print(f"\n✅ No F32 parameters found in model!")
    
    if buffer_stats:
        print(f"\n🗃️  Buffer Statistics:")
        for dtype, stats in sorted(buffer_stats.items()):
            memory_mb = stats['memory'] / (1024**2)
            print(f"  {dtype}: {stats['count']:,} buffers - {memory_mb:.2f} MB")
    
    if f32_buffers:
        print(f"\n🔍 F32 Buffers ({len(f32_buffers)} buffers):")
        for name, shape, count in f32_buffers:
            print(f"  {name}: {shape} - {count:,} elements")
    
    # Memory breakdown analysis
    print(f"\n💾 Expected vs Actual Memory:")
    expected_f16_memory = total_params * 2 / (1024**3)  # 2 bytes per f16 param
    actual_memory = total_memory / (1024**3)
    print(f"  Expected (pure f16): {expected_f16_memory:.2f} GB")
    print(f"  Actual: {actual_memory:.2f} GB")
    print(f"  Overhead: {actual_memory - expected_f16_memory:.2f} GB")
    
    # Recommendations
    print(f"\n💡 Analysis:")
    if actual_memory > expected_f16_memory * 1.1:  # More than 10% overhead
        print("  🔸 Model has significant f32 overhead")
        print("  🔸 Consider using --no_mixed_precision for maximum memory savings")
        print("  🔸 Some components may be keeping f32 for numerical stability")
    else:
        print("  ✅ Model is efficiently using f16 precision")
    
    return {
        'total_params': total_params,
        'total_memory_gb': actual_memory,
        'expected_f16_gb': expected_f16_memory,
        'f32_params': len(f32_parameters),
        'f32_buffers': len(f32_buffers)
    }

if __name__ == "__main__":
    analyze_model_dtypes("./VibeVoice-1.5B-f16")