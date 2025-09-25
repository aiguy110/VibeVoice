#!/usr/bin/env python3

import torch
import numpy as np

def test_cpu_f16_operations():
    """Test CPU f16 operations that were problematic in the codebase"""
    print("Testing CPU f16 capabilities...")
    print(f"PyTorch version: {torch.__version__}")

    # Test basic f16 tensor creation
    try:
        x = torch.randn(10, 20, dtype=torch.float16, device='cpu')
        print("✓ CPU f16 tensor creation: PASS")
    except Exception as e:
        print(f"✗ CPU f16 tensor creation: FAIL - {e}")
        return False

    # Test clamp operation (mentioned as problematic in DPM solver)
    try:
        x = torch.randn(10, 20, dtype=torch.float16, device='cpu')
        clamped = torch.clamp(x, min=-1.0, max=1.0)
        print("✓ CPU f16 clamp operation: PASS")
    except Exception as e:
        print(f"✗ CPU f16 clamp operation: FAIL - {e}")
        return False

    # Test quantile calculation (used in DPM solver)
    try:
        x = torch.randn(100, dtype=torch.float16, device='cpu')
        q = torch.quantile(x.abs(), 0.995)
        print("✓ CPU f16 quantile operation: PASS")
    except Exception as e:
        print(f"✗ CPU f16 quantile operation: FAIL - {e}")
        print("  Note: May need to upcast for quantile operations")

    # Test common math operations
    try:
        x = torch.randn(10, 10, dtype=torch.float16, device='cpu')
        y = torch.randn(10, 10, dtype=torch.float16, device='cpu')

        # Addition
        z = x + y
        # Multiplication
        z = x * y
        # Matrix multiplication
        z = torch.mm(x, y)
        # Activations
        z = torch.relu(x)
        z = torch.sigmoid(x)

        print("✓ CPU f16 basic math operations: PASS")
    except Exception as e:
        print(f"✗ CPU f16 basic math operations: FAIL - {e}")
        return False

    # Test conversions
    try:
        x_f32 = torch.randn(10, 10, dtype=torch.float32, device='cpu')
        x_f16 = x_f32.half()
        x_back = x_f16.float()
        print("✓ CPU f16 conversions: PASS")
    except Exception as e:
        print(f"✗ CPU f16 conversions: FAIL - {e}")
        return False

    # Test numpy conversion
    try:
        x = torch.randn(10, dtype=torch.float16, device='cpu')
        x_np = x.numpy()
        print(f"✓ CPU f16 to numpy conversion: PASS (dtype: {x_np.dtype})")
    except Exception as e:
        print(f"✗ CPU f16 to numpy conversion: FAIL - {e}")
        return False

    print("\n🎉 CPU f16 support looks good for modern PyTorch!")
    return True

if __name__ == "__main__":
    test_cpu_f16_operations()
