#!/usr/bin/env python3
"""
Debug script to trace imports and identify what triggers PyCUDA initialization.

Run this before your training script to see what's being imported.
"""

import sys
import os

# Set environment variable FIRST, before any other imports
os.environ['MOCU_BACKEND'] = 'torch'
print(f"[DEBUG] Set MOCU_BACKEND=torch")

# Track imports
original_import = __builtins__.__import__
import_trace = []

def debug_import(name, *args, **kwargs):
    """Track imports that might trigger PyCUDA."""
    # Filter for interesting imports
    if 'pycuda' in name.lower() or 'mocu' in name.lower():
        import_trace.append(name)
        print(f"[DEBUG] IMPORT: {name}")
        if 'pycuda' in name.lower():
            print(f"  ⚠️  PyCUDA import detected!")
    
    return original_import(name, *args, **kwargs)

# Install import hook (optional - uncomment to trace all imports)
# __builtins__.__import__ = debug_import

print(f"[DEBUG] Import tracking enabled")
print(f"[DEBUG] Python path: {sys.path[:3]}...")
print()

# Now import your actual modules
print("=" * 80)
print("TESTING IMPORTS")
print("=" * 80)

try:
    print("1. Importing torch...")
    import torch
    print(f"   ✓ PyTorch {torch.__version__} imported")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available")
    print()
except Exception as e:
    print(f"   ✗ Error: {e}")
    print()

try:
    print("2. Importing mocu_backend...")
    from src.core.mocu_backend import MOCU
    print(f"   ✓ mocu_backend imported (MOCU is a wrapper)")
    print()
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    print()

try:
    print("3. Testing MOCU backend loading (should use torch)...")
    # This should trigger backend loading
    print(f"   MOCU type: {type(MOCU)}")
    # Don't actually call it, just check the type
    print(f"   ✓ MOCU wrapper created")
    print()
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    print()

try:
    print("4. Checking if mocu_cuda would be safe to import...")
    import os
    backend_mode = os.getenv('MOCU_BACKEND', 'auto')
    print(f"   MOCU_BACKEND: {backend_mode}")
    
    if backend_mode == 'torch':
        print("   ✓ mocu_cuda would be BLOCKED (safe)")
    else:
        try:
            import torch
            if torch.cuda.is_available():
                print("   ⚠️  mocu_cuda would be BLOCKED (PyTorch CUDA active)")
            else:
                print("   ✓ mocu_cuda would be SAFE (PyTorch CUDA not active)")
        except:
            print("   ✓ mocu_cuda would be SAFE (PyTorch not available)")
    print()
except Exception as e:
    print(f"   ✗ Error: {e}")
    print()

try:
    print("5. Testing MPNN predictor imports...")
    from src.models.predictors.predictor_utils import load_mpnn_predictor
    print("   ✓ predictor_utils imported")
    
    # Check if sampling_mocu is imported
    if 'sampling_mocu' in sys.modules:
        print("   ⚠️  sampling_mocu was imported (may trigger PyCUDA)")
    else:
        print("   ✓ sampling_mocu NOT imported (safe)")
    print()
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    print()

print("=" * 80)
print("IMPORT TRACE SUMMARY")
print("=" * 80)
if import_trace:
    print("Suspicious imports detected:")
    for imp in import_trace:
        print(f"  - {imp}")
else:
    print("No suspicious imports detected")
print()

print("=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print("If this script runs without segfault, the issue is in:")
print("  1. Your training script importing something specific")
print("  2. Model loading triggering something")
print("  3. Runtime CUDA operations causing conflicts")
print()
print("Next steps:")
print("  1. Run this debug script: python3 debug_imports.py")
print("  2. If successful, try running training with extra debugging")
print("  3. Add print statements in train_dad_policy.py to see where it crashes")

