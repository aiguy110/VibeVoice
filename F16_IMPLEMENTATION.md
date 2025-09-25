# VibeVoice F16 Implementation Guide

This document describes the complete implementation of float16 (f16) precision support for VibeVoice, enabling ~50% memory reduction while maintaining audio quality.

## 🎯 Overview

The f16 implementation includes:
- ✅ **CPU f16 Support**: Fixed DPM solver restrictions for CPU f16 operations
- ✅ **Demo Script Integration**: Added `--use_f16` flags to all demo scripts  
- ✅ **Smart Dtype Handling**: Automatic fallback to f32 when needed for numerical stability
- ✅ **Audio Pipeline Compatibility**: Updated audio processing to handle f16 tensors
- ✅ **Model Conversion Tool**: Automated script to convert existing models to f16
- ✅ **Comprehensive Testing**: Validation suite for all f16 components

## 🔧 Environment Setup

### Option 1: Use HuggingFace Models (Recommended)
The easiest way to get started is using pre-converted F16 models from HuggingFace Hub (see [HuggingFace section](#-using-f16-models-from-huggingface-hub)).

### Option 2: Create F16 Development Environment

If you need to convert models locally or modify the F16 implementation, create a dedicated environment:

```bash
# Create Python virtual environment for F16 work
python3 -m venv f16_env
source f16_env/bin/activate  # On Windows: f16_env\Scripts\activate

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate
pip install gradio soundfile librosa
pip install huggingface_hub

# Install VibeVoice in editable mode
pip install -e .

# Optional: Install additional dependencies for full F16 testing
pip install diffusers fastapi uvicorn
```

### Environment Validation

```bash
# Test F16 environment setup
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'F16 supported: {torch.cuda.is_available() or hasattr(torch, \"float16\")}')
print('Environment ready for F16 operations!')
"
```

> **Note**: The `f16_env/` directory is not included in version control and should be recreated locally as needed. Use the instructions above to set up your own F16 development environment.

## 🚀 Quick Start

### 1. Convert an Existing Model to F16

```bash
# Convert VibeVoice-1.5B to f16
python -m vibevoice.scripts.convert_f32_to_f16 \
    --input_path microsoft/VibeVoice-1.5B \
    --output_path ./VibeVoice-1.5B-f16 \
    --test_conversion

# Convert VibeVoice-Large to f16  
python -m vibevoice.scripts.convert_f32_to_f16 \
    --input_path microsoft/VibeVoice-Large \
    --output_path ./VibeVoice-Large-f16 \
    --test_conversion
```

### 2. Run Inference with F16

```bash
# File-based inference with f16
python demo/inference_from_file.py \
    --model_path ./VibeVoice-1.5B-f16 \
    --use_f16 \
    --device cpu \
    --txt_path demo/text_examples/1p_abs.txt \
    --speaker_names Alice

# Interactive Gradio demo with f16
python demo/gradio_demo.py \
    --model_path ./VibeVoice-1.5B-f16 \
    --use_f16 \
    --device cpu
```

### 3. Validate Your Installation

```bash
# Run comprehensive f16 tests
python test_f16_implementation.py --model_path microsoft/VibeVoice-1.5B

# Quick CPU f16 capability test
python test_cpu_f16.py
```

## 📊 Memory & Performance Benefits

| Model Size | Original (f32) | F16 Version | Memory Savings |
|------------|---------------|-------------|----------------|
| VibeVoice-1.5B | ~6 GB | ~3 GB | ~50% |
| VibeVoice-Large | ~13 GB | ~6.5 GB | ~50% |

**Performance Notes:**
- CPU inference may be slightly slower due to f16→f32 conversions
- GPU inference (CUDA) benefits from native f16 acceleration
- Audio quality remains very close to original f32 models

## 🔧 Implementation Details

### Key Changes Made

1. **DPM Solver Fix** (`vibevoice/schedule/dmp_solver.py`)
   - Removed hard-coded CPU f16 restriction
   - Added dynamic capability testing for f16 operations
   - Intelligent fallback to f32 when needed

2. **Demo Scripts** (`demo/inference_from_file.py`, `demo/gradio_demo.py`)
   - Added `--use_f16` command-line flag
   - Updated dtype selection logic for all devices
   - Enhanced audio tensor conversion for f16 compatibility

3. **Audio Processing** (`demo/gradio_demo.py`)
   - Updated tensor-to-numpy conversion to handle f16
   - Added automatic dtype detection and conversion

4. **Conversion Tool** (`vibevoice/scripts/convert_f32_to_f16.py`)
   - Comprehensive f32→f16 model conversion
   - Mixed precision support (keeps critical layers in f32)
   - Built-in testing and validation
   - Memory usage analysis

### Mixed Precision Strategy

The conversion uses **mixed precision** by default, keeping these layers in f32 for numerical stability:
- Layer normalization (`layernorm`, `norm`)
- Embeddings (`embed_tokens`) - optional
- Language model head (`lm_head`) - optional

This approach maintains quality while still achieving significant memory savings.

## 🛠️ Advanced Usage

### Custom Layer Exclusions

```bash
# Exclude additional layers from f16 conversion
python -m vibevoice.scripts.convert_f32_to_f16 \
    --input_path microsoft/VibeVoice-1.5B \
    --output_path ./VibeVoice-1.5B-custom-f16 \
    --exclude_layers "attention" "mlp.gate_proj" \
    --test_conversion
```

### Disable Mixed Precision

```bash
# Convert ALL layers to f16 (may reduce quality)
python -m vibevoice.scripts.convert_f32_to_f16 \
    --input_path microsoft/VibeVoice-1.5B \
    --output_path ./VibeVoice-1.5B-pure-f16 \
    --no_mixed_precision \
    --test_conversion
```

### Programmatic Usage

```python
from vibevoice.scripts.convert_f32_to_f16 import convert_vibevoice_to_f16

# Convert model programmatically
success = convert_vibevoice_to_f16(
    input_path="microsoft/VibeVoice-1.5B",
    output_path="./VibeVoice-1.5B-f16",
    mixed_precision=True,
    test_conversion=True
)
```

## 🧪 Testing & Validation

### Comprehensive Test Suite

The implementation includes a full test suite (`test_f16_implementation.py`):

1. **CPU F16 Operations**: Validates basic f16 math operations
2. **DPM Solver Compatibility**: Tests diffusion sampling with f16
3. **Demo Script Integration**: Verifies command-line flag parsing
4. **Conversion Script**: Validates model conversion workflow
5. **Audio Processing**: Tests audio pipeline f16 compatibility

### Running Tests

```bash
# Run all tests
python test_f16_implementation.py

# Test with specific model
python test_f16_implementation.py --model_path ./VibeVoice-1.5B-f16

# Test just CPU f16 capabilities
python test_cpu_f16.py
```

## ⚠️ Important Notes

### Device Compatibility
- **CPU**: Full f16 support with automatic fallback
- **CUDA**: Native f16 acceleration available
- **MPS (Apple)**: Limited f16 support, falls back to f32

### Quality Considerations
- F16 may introduce minor numerical differences
- Critical operations automatically upcast to f32 when needed
- Mixed precision minimizes quality impact while maximizing memory savings

### Troubleshooting

**Issue**: Model fails to load with f16
```bash
# Solution: Check PyTorch version and device compatibility
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Issue**: Poor audio quality with f16
```bash
# Solution: Use mixed precision (default) or exclude more layers
python -m vibevoice.scripts.convert_f32_to_f16 \
    --input_path your_model \
    --output_path your_f16_model \
    --exclude_layers "layernorm" "norm" "lm_head" "embed_tokens" "attention"
```

## 🌐 Using F16 Models from HuggingFace Hub

Pre-converted F16 models are available on HuggingFace Hub for immediate use:

### Available Models
- **VibeVoice-1.5B-f16**: `aiguy110/VibeVoice-1.5B-f16`
- **VibeVoice-Large-f16**: `aiguy110/VibeVoice-Large-f16`

### Quick Start with HuggingFace Models

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Use F16 models directly in demo scripts
python demo/gradio_demo.py --model_path aiguy110/VibeVoice-1.5B-f16 --use_f16
python demo/inference_from_file.py --model_path aiguy110/VibeVoice-1.5B-f16 --txt_path demo/text_examples/1p_abs.txt --speaker_names Alice
```

### Programmatic Usage

```python
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
import torch

# Load F16 model from HuggingFace
processor = VibeVoiceProcessor.from_pretrained("aiguy110/VibeVoice-1.5B-f16")
model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    "aiguy110/VibeVoice-1.5B-f16",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Models will be automatically downloaded and cached locally
```

### Benefits of HuggingFace F16 Models
- **Instant availability** - No local conversion needed
- **Automatic caching** - Downloaded once, reused thereafter
- **Easy sharing** - Use model names instead of local paths
- **Version control** - Consistent model versions across environments

## 🎉 Summary

This f16 implementation provides:
- **50% memory reduction** for CPU and GPU inference
- **Maintained audio quality** through mixed precision
- **Full backward compatibility** with existing code
- **Easy conversion** of existing models
- **HuggingFace Hub integration** for instant access
- **Comprehensive testing** and validation

The implementation is production-ready and extensively tested across different devices and model sizes.

---

**Need Help?** Check the test output for specific issues or run the validation scripts to ensure your setup is working correctly.