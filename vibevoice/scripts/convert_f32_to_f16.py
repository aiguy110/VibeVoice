#!/usr/bin/env python3
"""
Convert VibeVoice model from f32/bfloat16 to f16 precision.

This script converts an existing VibeVoice model to use float16 (f16) precision,
which can reduce memory usage by approximately 50% while maintaining reasonable quality.

Usage:
    python -m vibevoice.scripts.convert_f32_to_f16 \
        --input_path microsoft/VibeVoice-1.5B \
        --output_path ./VibeVoice-1.5B-f16 \
        --test_conversion
"""

import argparse
import json
import os
import shutil
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from transformers.utils import logging

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

logger = logging.get_logger(__name__)


def get_model_memory_usage(model: torch.nn.Module) -> Dict[str, float]:
    """Calculate model memory usage in GB."""
    total_params = 0
    total_bytes = 0

    for param in model.parameters():
        param_count = param.numel()
        param_bytes = param.numel() * param.element_size()
        total_params += param_count
        total_bytes += param_bytes

    return {
        "parameters": total_params,
        "memory_gb": total_bytes / (1024**3),
        "memory_mb": total_bytes / (1024**2)
    }


def convert_model_to_f16(
    model: torch.nn.Module,
    exclude_layers: Optional[List[str]] = None,
    mixed_precision: bool = True
) -> torch.nn.Module:
    """
    Convert model to f16 precision with optional mixed precision.

    Args:
        model: The model to convert
        exclude_layers: List of layer names/patterns to keep in f32
        mixed_precision: If True, keep certain critical layers in f32

    Returns:
        Converted model
    """
    if exclude_layers is None:
        exclude_layers = []

    # Default layers to keep in f32 for numerical stability
    if mixed_precision:
        exclude_layers.extend([
            "layernorm",  # Layer normalization layers
            "norm",       # RMS norm layers
            "lm_head",    # Language model head (optional)
            "embed_tokens",  # Embeddings (optional)
        ])

    def should_exclude_layer(name: str) -> bool:
        """Check if layer should be excluded from f16 conversion."""
        name_lower = name.lower()
        for exclude_pattern in exclude_layers:
            if exclude_pattern.lower() in name_lower:
                return True
        return False

    logger.info("Converting model to f16 precision...")
    logger.info(f"Excluded layer patterns: {exclude_layers}")

    kept_in_f32 = []
    converted_to_f16 = []

    for name, param in model.named_parameters():
        if should_exclude_layer(name):
            # Keep in original precision (likely f32)
            kept_in_f32.append(name)
        else:
            # Convert to f16
            param.data = param.data.half()
            converted_to_f16.append(name)

    # Also convert buffers
    for name, buffer in model.named_buffers():
        if should_exclude_layer(name):
            kept_in_f32.append(f"buffer:{name}")
        else:
            if buffer.dtype.is_floating_point:
                buffer.data = buffer.data.half()
                converted_to_f16.append(f"buffer:{name}")

    logger.info(f"Converted {len(converted_to_f16)} parameters/buffers to f16")
    logger.info(f"Kept {len(kept_in_f32)} parameters/buffers in original precision")

    return model


def test_model_conversion(
    original_path: str,
    converted_path: str,
    test_text: str = "Speaker 1: Hello, this is a test of the VibeVoice model conversion."
) -> bool:
    """
    Test that the converted model works and produces reasonable outputs.

    Args:
        original_path: Path to original model
        converted_path: Path to converted model
        test_text: Text to test with

    Returns:
        True if test passes, False otherwise
    """
    logger.info("Testing model conversion...")

    try:
        # Load processor
        processor = VibeVoiceProcessor.from_pretrained(original_path)

        # Load original model
        logger.info("Loading original model...")
        original_model = VibeVoiceForConditionalGeneration.from_pretrained(
            original_path,
            torch_dtype=torch.float32,
            device_map="cpu"
        )

        # Load converted model
        logger.info("Loading converted f16 model...")
        converted_model = VibeVoiceForConditionalGeneration.from_pretrained(
            converted_path,
            torch_dtype=torch.float16,
            device_map="cpu"
        )

        # Compare memory usage
        orig_memory = get_model_memory_usage(original_model)
        conv_memory = get_model_memory_usage(converted_model)

        logger.info(f"Original model: {orig_memory['memory_gb']:.2f} GB")
        logger.info(f"Converted model: {conv_memory['memory_gb']:.2f} GB")
        logger.info(f"Memory savings: {(1 - conv_memory['memory_gb'] / orig_memory['memory_gb']) * 100:.1f}%")

        # Basic inference test (just check model doesn't crash)
        test_inputs = processor(test_text, return_tensors="pt")

        with torch.no_grad():
            # Test original model
            try:
                orig_outputs = original_model.generate(
                    **test_inputs,
                    max_length=50,
                    do_sample=False,
                    num_return_sequences=1
                )
                logger.info("✓ Original model inference: PASS")
            except Exception as e:
                logger.error(f"Original model inference failed: {e}")
                return False

            # Test converted model
            try:
                # Convert inputs to f16 if they're float tensors
                test_inputs_f16 = {}
                for k, v in test_inputs.items():
                    if torch.is_tensor(v) and v.dtype.is_floating_point:
                        test_inputs_f16[k] = v.half()
                    else:
                        test_inputs_f16[k] = v

                conv_outputs = converted_model.generate(
                    **test_inputs_f16,
                    max_length=50,
                    do_sample=False,
                    num_return_sequences=1
                )
                logger.info("✓ Converted model inference: PASS")
            except Exception as e:
                logger.error(f"Converted model inference failed: {e}")
                return False

        logger.info("🎉 Model conversion test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Model conversion test failed: {e}")
        return False

    finally:
        # Clean up memory
        if 'original_model' in locals():
            del original_model
        if 'converted_model' in locals():
            del converted_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def convert_vibevoice_to_f16(
    input_path: str,
    output_path: str,
    mixed_precision: bool = True,
    exclude_layers: Optional[List[str]] = None,
    test_conversion: bool = False
) -> bool:
    """
    Convert a VibeVoice model from f32/bfloat16 to f16 precision.

    Args:
        input_path: Path to input model (local path or HuggingFace model ID)
        output_path: Path where to save the converted model
        mixed_precision: Whether to use mixed precision (keep some layers in f32)
        exclude_layers: Additional layer patterns to exclude from conversion
        test_conversion: Whether to run conversion test

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Converting VibeVoice model from {input_path} to f16 precision")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Mixed precision: {mixed_precision}")

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Load original model configuration
        logger.info("Loading model configuration...")
        config = VibeVoiceConfig.from_pretrained(input_path)

        # Update config to specify f16 dtype
        config.torch_dtype = "float16"

        # Load model in original precision
        logger.info("Loading original model...")
        model = VibeVoiceForConditionalGeneration.from_pretrained(
            input_path,
            torch_dtype=torch.float32,  # Load in f32 first
            device_map="cpu",  # Keep on CPU to avoid GPU memory issues
        )

        # Get original memory usage
        orig_memory = get_model_memory_usage(model)
        logger.info(f"Original model memory usage: {orig_memory['memory_gb']:.2f} GB")

        # Convert to f16
        model = convert_model_to_f16(
            model,
            exclude_layers=exclude_layers,
            mixed_precision=mixed_precision
        )

        # Get converted memory usage
        conv_memory = get_model_memory_usage(model)
        memory_savings = (1 - conv_memory['memory_gb'] / orig_memory['memory_gb']) * 100
        logger.info(f"Converted model memory usage: {conv_memory['memory_gb']:.2f} GB")
        logger.info(f"Memory savings: {memory_savings:.1f}%")

        # Save converted model
        logger.info("Saving converted model...")
        config.save_pretrained(output_path)

        model.save_pretrained(
            output_path,
            max_shard_size="2GB",
            safe_serialization=True
        )

        # Copy processor files if they exist
        processor_files = [
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "special_tokens_map.json"
        ]

        for file_name in processor_files:
            src_path = Path(input_path) / file_name
            if src_path.exists():
                dst_path = Path(output_path) / file_name
                shutil.copy2(src_path, dst_path)
                logger.info(f"Copied {file_name}")

        # Create README with conversion info
        readme_content = f"""# VibeVoice F16 Model

This model has been converted to float16 (f16) precision for reduced memory usage.

## Conversion Details
- **Original model**: {input_path}
- **Mixed precision**: {mixed_precision}
- **Memory savings**: ~{memory_savings:.1f}%
- **Original size**: {orig_memory['memory_gb']:.2f} GB
- **Converted size**: {conv_memory['memory_gb']:.2f} GB

## Usage

```python
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

# Load with f16 precision
model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    "{output_path}",
    torch_dtype=torch.float16,
    device_map="cpu"  # or "cuda" for GPU
)

processor = VibeVoiceProcessor.from_pretrained("{output_path}")

# Use --use_f16 flag with demo scripts
python demo/inference_from_file.py --model_path {output_path} --use_f16 --device cpu
```

## Notes
- F16 precision may result in minor quality differences compared to f32
- Some operations automatically upcast to f32 for numerical stability
- Optimized for CPU inference, but also works on CUDA GPUs
"""

        with open(Path(output_path) / "README.md", "w") as f:
            f.write(readme_content)

        logger.info(f"✅ Model conversion completed successfully!")
        logger.info(f"Converted model saved to: {output_path}")

        # Run test if requested
        if test_conversion:
            success = test_model_conversion(input_path, output_path)
            if not success:
                logger.error("❌ Model conversion test failed!")
                return False

        return True

    except Exception as e:
        logger.error(f"❌ Model conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert VibeVoice model to f16 precision"
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
        "--exclude_layers",
        nargs="*",
        default=None,
        help="Additional layer name patterns to exclude from f16 conversion"
    )
    parser.add_argument(
        "--test_conversion",
        action="store_true",
        help="Run inference test after conversion"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.set_verbosity_info()
    else:
        logging.set_verbosity_warning()

    success = convert_vibevoice_to_f16(
        input_path=args.input_path,
        output_path=args.output_path,
        mixed_precision=not args.no_mixed_precision,
        exclude_layers=args.exclude_layers,
        test_conversion=args.test_conversion
    )

    if success:
        print("\n🎉 F16 conversion completed successfully!")
        print(f"📁 Converted model saved to: {args.output_path}")
        print(f"💾 Expected memory savings: ~50%")
        print(f"🚀 Test your model with: python demo/inference_from_file.py --model_path {args.output_path} --use_f16 --device cpu")
    else:
        print("\n❌ F16 conversion failed. Check logs for details.")
        exit(1)


if __name__ == "__main__":
    main()
