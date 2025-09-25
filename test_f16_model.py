#!/usr/bin/env python3
"""Quick test of the converted f16 model"""

import torch
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

def test_f16_model():
    print("🧪 Testing f16 converted model...")

    try:
        # Load processor
        processor = VibeVoiceProcessor.from_pretrained("./VibeVoice-1.5B-f16")
        print("✅ Processor loaded successfully")

        # Load f16 model
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            "./VibeVoice-1.5B-f16",
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        print("✅ F16 model loaded successfully")

        # Test text processing
        test_text = "Speaker 1: Hello, this is a test of the converted model."
        inputs = processor(test_text, return_tensors="pt")
        print("✅ Text processing successful")

        # Check model memory usage
        total_params = sum(p.numel() for p in model.parameters())
        f16_params = sum(p.numel() for p in model.parameters() if p.dtype == torch.float16)
        f32_params = sum(p.numel() for p in model.parameters() if p.dtype == torch.float32)

        print(f"📊 Model Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  F16 parameters: {f16_params:,} ({f16_params/total_params*100:.1f}%)")
        print(f"  F32 parameters: {f32_params:,} ({f32_params/total_params*100:.1f}%)")

        # Calculate memory usage
        memory_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        memory_gb = memory_bytes / (1024**3)
        print(f"  Memory usage: {memory_gb:.2f} GB")

        print("\n🎉 F16 model test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ F16 model test failed: {e}")
        return False

if __name__ == "__main__":
    test_f16_model()
