#!/usr/bin/env python3
"""
Incremental VibeVoice f32 to f16 conversion script.

This script converts models shard-by-shard to avoid RAM exhaustion,
perfect for systems with limited memory.

Usage:
    python -m vibevoice.scripts.convert_f32_to_f16_incremental \
        --input_path microsoft/VibeVoice-1.5B \
        --output_path ./VibeVoice-1.5B-f16 \
        --max_shard_size 1GB
"""

import argparse
import gc
import json
import os
import psutil
import shutil
import time
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from transformers.utils import logging
from safetensors.torch import load_file, save_file

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig

logger = logging.get_logger(__name__)


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb / 1024


def parse_size(size_str: str) -> int:
    """Parse size string like '1GB', '500MB' to bytes."""
    size_str = size_str.upper().strip()
    
    if size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    elif size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith('KB'):
        return int(float(size_str[:-2]) * 1024)
    else:
        return int(size_str)


def should_keep_in_f32(param_name: str, mixed_precision: bool = True) -> bool:
    """Determine if a parameter should stay in f32 for numerical stability."""
    if not mixed_precision:
        return False
    
    # Keep these layers in f32 for stability
    keep_patterns = [
        'layernorm', 'norm', 'ln_', 'layer_norm',
        'embed_tokens', 'embed_positions', 'wte', 'wpe',
        'lm_head', 'classifier', 'score'
    ]
    
    param_lower = param_name.lower()
    return any(pattern in param_lower for pattern in keep_patterns)


def convert_shard_to_f16(
    shard_path: str, 
    output_path: str,
    mixed_precision: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Convert a single shard file to f16.
    
    Returns statistics about the conversion.
    """
    if verbose:
        logger.info(f"Processing shard: {shard_path}")
    
    start_memory = get_memory_usage()
    
    # Load the shard
    state_dict = load_file(shard_path)
    
    load_memory = get_memory_usage()
    if verbose:
        logger.info(f"  Loaded shard: {load_memory - start_memory:.2f}GB additional RAM")
    
    # Convert parameters
    converted_dict = {}
    stats = {
        'total_params': 0,
        'f16_params': 0,
        'f32_params': 0,
        'total_size_mb': 0,
        'f16_size_mb': 0,
        'f32_size_mb': 0
    }
    
    for name, param in state_dict.items():
        stats['total_params'] += param.numel()
        
        if param.dtype.is_floating_point and not should_keep_in_f32(name, mixed_precision):
            # Convert to f16
            converted_param = param.half()
            converted_dict[name] = converted_param
            stats['f16_params'] += param.numel()
            stats['f16_size_mb'] += param.numel() * 2 / (1024 * 1024)  # f16 = 2 bytes
        else:
            # Keep original precision
            converted_dict[name] = param
            if param.dtype == torch.float32:
                stats['f32_params'] += param.numel()
                stats['f32_size_mb'] += param.numel() * 4 / (1024 * 1024)  # f32 = 4 bytes
            elif param.dtype == torch.float16:
                stats['f16_params'] += param.numel()
                stats['f16_size_mb'] += param.numel() * 2 / (1024 * 1024)
        
        stats['total_size_mb'] = stats['f16_size_mb'] + stats['f32_size_mb']
    
    convert_memory = get_memory_usage()
    if verbose:
        logger.info(f"  Converted parameters: {convert_memory - load_memory:.2f}GB additional RAM")
    
    # Save the converted shard
    save_file(converted_dict, output_path)
    
    save_memory = get_memory_usage()
    if verbose:
        logger.info(f"  Saved shard: {save_memory:.2f}GB total RAM")
    
    # Clean up
    del state_dict
    del converted_dict
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    cleanup_memory = get_memory_usage()
    if verbose:
        logger.info(f"  After cleanup: {cleanup_memory:.2f}GB RAM")
    
    return stats


def incremental_convert_to_f16(
    input_path: str,
    output_path: str,
    mixed_precision: bool = True,
    max_shard_size: str = "2GB",
    verbose: bool = False
) -> bool:
    """
    Convert VibeVoice model to f16 incrementally to avoid RAM exhaustion.
    """
    try:
        logger.info(f"🔄 Starting incremental f16 conversion")
        logger.info(f"📥 Input: {input_path}")
        logger.info(f"📤 Output: {output_path}")
        logger.info(f"💾 Max shard size: {max_shard_size}")
        logger.info(f"🧮 Mixed precision: {mixed_precision}")
        
        start_memory = get_memory_usage()
        logger.info(f"🐏 Starting RAM usage: {start_memory:.2f}GB")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Load and update config
        logger.info("📋 Loading model configuration...")
        config = VibeVoiceConfig.from_pretrained(input_path)
        config.torch_dtype = "float16"
        config.save_pretrained(output_path)
        
        # Find model files
        input_dir = Path(input_path) if os.path.isdir(input_path) else Path.home() / ".cache/huggingface/hub" / f"models--{input_path.replace('/', '--')}"
        
        # Look for safetensors files
        if os.path.isdir(input_path):
            model_files = list(Path(input_path).glob("*.safetensors"))
            index_file = Path(input_path) / "model.safetensors.index.json"
        else:
            # Download from HF Hub
            from huggingface_hub import snapshot_download
            logger.info("📥 Downloading model from Hugging Face...")
            local_dir = snapshot_download(
                repo_id=input_path,
                allow_patterns=["*.safetensors", "*.json", "*.txt"],
                local_dir=None,
                local_dir_use_symlinks=False
            )
            model_files = list(Path(local_dir).glob("*.safetensors"))
            index_file = Path(local_dir) / "model.safetensors.index.json"
        
        if not model_files:
            raise FileNotFoundError("No safetensors files found in model directory")
        
        logger.info(f"📦 Found {len(model_files)} model files to convert")
        
        # Process each shard individually
        total_stats = {
            'total_params': 0,
            'f16_params': 0,
            'f32_params': 0,
            'total_size_mb': 0,
            'f16_size_mb': 0,
            'f32_size_mb': 0
        }
        
        converted_files = []
        
        for i, shard_file in enumerate(sorted(model_files)):
            logger.info(f"🔄 Processing shard {i+1}/{len(model_files)}: {shard_file.name}")
            
            output_shard = Path(output_path) / shard_file.name
            
            # Convert this shard
            shard_stats = convert_shard_to_f16(
                str(shard_file),
                str(output_shard),
                mixed_precision=mixed_precision,
                verbose=verbose
            )
            
            # Update totals
            for key in total_stats:
                total_stats[key] += shard_stats[key]
            
            converted_files.append(shard_file.name)
            
            current_memory = get_memory_usage()
            logger.info(f"  ✅ Shard {i+1} complete. RAM: {current_memory:.2f}GB")
            
            # Brief pause to let system recover
            time.sleep(1)
        
        # Update index file if it exists
        if index_file.exists():
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            # Update weight map to point to converted files
            if 'weight_map' in index_data:
                new_weight_map = {}
                for param_name, file_name in index_data['weight_map'].items():
                    new_weight_map[param_name] = file_name  # File names stay the same
                
                index_data['weight_map'] = new_weight_map
            
            # Save updated index
            output_index = Path(output_path) / "model.safetensors.index.json"
            with open(output_index, 'w') as f:
                json.dump(index_data, f, indent=2)
            
            logger.info("📋 Updated model index file")
        
        # Copy other necessary files
        other_files = ["preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", 
                      "vocab.json", "special_tokens_map.json", "README.md"]
        
        source_dir = Path(input_path) if os.path.isdir(input_path) else Path(local_dir)
        for file_name in other_files:
            src_file = source_dir / file_name
            if src_file.exists():
                dst_file = Path(output_path) / file_name
                shutil.copy2(src_file, dst_file)
                if verbose:
                    logger.info(f"📋 Copied {file_name}")
        
        # Calculate savings
        memory_savings = (total_stats['f32_size_mb'] - total_stats['f16_size_mb']) / total_stats['f32_size_mb'] * 100 if total_stats['f32_size_mb'] > 0 else 0
        total_size_gb = total_stats['total_size_mb'] / 1024
        
        logger.info("🎉 Incremental conversion completed!")
        logger.info(f"📊 Conversion Statistics:")
        logger.info(f"  Total parameters: {total_stats['total_params']:,}")
        logger.info(f"  F16 parameters: {total_stats['f16_params']:,} ({total_stats['f16_params']/total_stats['total_params']*100:.1f}%)")
        logger.info(f"  F32 parameters: {total_stats['f32_params']:,} ({total_stats['f32_params']/total_stats['total_params']*100:.1f}%)")
        logger.info(f"  Model size: {total_size_gb:.2f}GB")
        logger.info(f"  Memory savings: ~{memory_savings:.1f}%")
        
        final_memory = get_memory_usage()
        logger.info(f"🐏 Peak RAM usage: {final_memory:.2f}GB")
        
        # Create README
        readme_content = f"""# VibeVoice F16 Model (Incremental Conversion)

This model has been converted to float16 (f16) precision using incremental processing to avoid RAM exhaustion.

## Conversion Details
- **Original model**: {input_path}
- **Conversion method**: Incremental shard-by-shard processing
- **Mixed precision**: {mixed_precision}
- **Total parameters**: {total_stats['total_params']:,}
- **F16 parameters**: {total_stats['f16_params']:,} ({total_stats['f16_params']/total_stats['total_params']*100:.1f}%)
- **F32 parameters**: {total_stats['f32_params']:,} ({total_stats['f32_params']/total_stats['total_params']*100:.1f}%)
- **Model size**: {total_size_gb:.2f}GB
- **Memory savings**: ~{memory_savings:.1f}%

## Usage

```python
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

# Load with f16 precision
model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    "{output_path}",
    torch_dtype=torch.float16,
    device_map="cpu"
)

processor = VibeVoiceProcessor.from_pretrained("{output_path}")

# Use --use_f16 flag with demo scripts
python demo/inference_from_file.py --model_path {output_path} --use_f16 --device cpu
```

## RAM-Friendly Conversion
This model was converted using incremental processing, making it possible to convert large models on systems with limited RAM.
"""
        
        with open(Path(output_path) / "README.md", "w") as f:
            f.write(readme_content)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Incremental conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert VibeVoice model to f16 precision incrementally (RAM-friendly)"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input model (local path or HuggingFace model ID)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where to save the converted f16 model"
    )
    parser.add_argument(
        "--no_mixed_precision",
        action="store_true",
        help="Disable mixed precision (convert ALL layers to f16)"
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="2GB",
        help="Maximum size per shard (e.g., '1GB', '500MB')"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging with memory usage"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.set_verbosity_info()
    else:
        logging.set_verbosity_warning()
    
    success = incremental_convert_to_f16(
        input_path=args.input_path,
        output_path=args.output_path,
        mixed_precision=not args.no_mixed_precision,
        max_shard_size=args.max_shard_size,
        verbose=args.verbose
    )
    
    if success:
        print(f"\n🎉 Incremental f16 conversion completed successfully!")
        print(f"📁 Converted model saved to: {args.output_path}")
        print(f"💾 RAM-friendly conversion completed without exhausting system memory")
        print(f"🚀 Test your model with: python demo/inference_from_file.py --model_path {args.output_path} --use_f16 --device cpu")
    else:
        print(f"\n❌ Incremental f16 conversion failed. Check logs for details.")
        exit(1)


if __name__ == "__main__":
    main()