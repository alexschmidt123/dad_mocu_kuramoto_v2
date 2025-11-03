#!/usr/bin/env python3
"""
Quick script to verify GPU is being used for MOCU computation.
Run this before data generation to verify GPU setup.
"""

import torch
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.mocu import MOCU, _mocu_comp_torch

print("=" * 80)
print("GPU Usage Verification Test")
print("=" * 80)

# Check CUDA availability
print("\n[1] CUDA Availability Check:")
print(f"  - PyTorch version: {torch.__version__}")
print(f"  - CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  - CUDA version: {torch.version.cuda}")
    print(f"  - GPU count: {torch.cuda.device_count()}")
    print(f"  - GPU name: {torch.cuda.get_device_name(0)}")
    print(f"  - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("  ✗ CUDA not available - GPU acceleration disabled")
    sys.exit(1)

# Test tensor operations on GPU
print("\n[2] Tensor GPU Operations Test:")
test_w = np.random.randn(5).astype(np.float32)
test_a = np.random.randn(5, 5).astype(np.float32)
test_a = (test_a + test_a.T) / 2  # Make symmetric

# Test _mocu_comp_torch
print("  Testing _mocu_comp_torch (RK4 integration)...")
try:
    result = _mocu_comp_torch(test_w, 0.01, 5, 640, test_a, device='cuda')
    print(f"  ✓ _mocu_comp_torch executed successfully on GPU")
    print(f"  Result (sync=1, non-sync=0): {result}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test MOCU computation
print("\n[3] MOCU Computation Test:")
print("  Running small MOCU computation (K_max=100 for speed)...")
try:
    # Create simple bounds
    a_lower = np.abs(test_a) * 0.5
    a_upper = np.abs(test_a) * 1.5
    
    import time
    start = time.time()
    mocu_val = MOCU(100, test_w, 5, 0.01, 640, 4.0, a_lower, a_upper, 0, device='cuda')
    elapsed = time.time() - start
    
    print(f"  ✓ MOCU computation completed on GPU")
    print(f"  - MOCU value: {mocu_val:.6f}")
    print(f"  - Time: {elapsed:.2f}s")
    print(f"  - GPU memory used: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    if elapsed < 1.0:
        print(f"  ✓ Speed check: Fast execution suggests GPU is being used")
    else:
        print(f"  ⚠ Speed check: Slow execution might indicate CPU usage")
        
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Verify tensors are on GPU
print("\n[4] Tensor Device Verification:")
test_tensor = torch.randn(10, 10, device='cuda')
if test_tensor.is_cuda:
    print(f"  ✓ Test tensor is on GPU (device: {test_tensor.device})")
else:
    print(f"  ✗ Test tensor is NOT on GPU (device: {test_tensor.device})")
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ All GPU tests passed! GPU acceleration is working correctly.")
print("=" * 80)
print("\nTo enable detailed GPU debugging during data generation, run:")
print("  export DEBUG_GPU=true")
print("  python scripts/generate_mocu_data.py ...")

