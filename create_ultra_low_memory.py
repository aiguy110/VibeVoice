#!/usr/bin/env python3
"""
Create ultra-low memory version using more aggressive techniques
"""

import torch
from vibevoice.scripts.convert_f32_to_f16 import convert_vibevoice_to_f16

def create_ultra_low_memory_model():
    """Convert with no mixed precision for absolute minimum memory"""
    print("🔧 Creating ultra-low memory f16 model...")
    
    success = convert_vibevoice_to_f16(
        input_path="microsoft/VibeVoice-1.5B",
        output_path="./VibeVoice-1.5B-ultra-f16",
        mixed_precision=False,  # No mixed precision - ALL layers to f16
        exclude_layers=[],  # Don't exclude anything
        test_conversion=True
    )
    
    if success:
        print("✅ Ultra-low memory model created!")
        print("📁 Location: ./VibeVoice-1.5B-ultra-f16")
        print("⚠️  Warning: May have slightly reduced numerical stability")
    else:
        print("❌ Ultra-low memory conversion failed")

if __name__ == "__main__":
    create_ultra_low_memory_model()