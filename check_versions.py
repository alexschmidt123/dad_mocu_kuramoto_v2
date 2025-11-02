#!/usr/bin/env python3
"""
Diagnostic script to check software versions and GPU compatibility.

This helps identify potential causes of segmentation faults in PyTorch/PyCUDA workloads.
"""

import sys
import platform

print("=" * 80)
print("SOFTWARE VERSIONS AND COMPATIBILITY CHECK")
print("=" * 80)
print()

# Python version
print("1. PYTHON VERSION")
print("-" * 80)
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print()

# PyTorch version and CUDA support
print("2. PYTORCH VERSION AND CUDA")
print("-" * 80)
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA available: YES")
        print(f"CUDA version (PyTorch): {torch.version.cuda}")
        print(f"cuDNN version (PyTorch): {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
            
            # Memory info
            torch.cuda.set_device(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
            free = total_memory - reserved
            
            print(f"  Total Memory: {total_memory:.2f} GB")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            print(f"  Free: {free:.2f} GB")
    else:
        print("CUDA available: NO")
        print("CUDA version (PyTorch): N/A")
        print("cuDNN version (PyTorch): N/A")
except ImportError:
    print("PyTorch: NOT INSTALLED")
except Exception as e:
    print(f"PyTorch: ERROR - {e}")
print()

# PyCUDA version
print("3. PYCUDA VERSION")
print("-" * 80)
try:
    import pycuda
    print(f"PyCUDA version: {pycuda.VERSION}")
    
    try:
        import pycuda.driver as drv
        drv.init()
        print(f"PyCUDA driver version: {drv.get_version()}")
        
        device_count = drv.Device.count()
        print(f"PyCUDA detected GPUs: {device_count}")
        
        for i in range(device_count):
            dev = drv.Device(i)
            print(f"\nGPU {i} (PyCUDA):")
            attrs = dev.get_attributes()
            print(f"  Name: {dev.name()}")
            print(f"  Compute Capability: {attrs[drv.device_attribute.COMPUTE_CAPABILITY_MAJOR]}.{attrs[drv.device_attribute.COMPUTE_CAPABILITY_MINOR]}")
            mem_total = attrs[drv.device_attribute.TOTAL_MEMORY] / (1024**3)  # GB
            print(f"  Total Memory: {mem_total:.2f} GB")
    except Exception as e:
        print(f"PyCUDA driver info: ERROR - {e}")
        print("  (PyCUDA may not be able to access GPU if PyTorch CUDA context is active)")
except ImportError:
    print("PyCUDA: NOT INSTALLED")
except Exception as e:
    print(f"PyCUDA: ERROR - {e}")
print()

# System CUDA version (from nvidia-smi)
print("4. SYSTEM CUDA AND DRIVER")
print("-" * 80)
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,cuda_version', '--format=csv,noheader'], 
                           capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for i, line in enumerate(lines):
            parts = line.split(', ')
            if len(parts) >= 2:
                print(f"GPU {i}:")
                print(f"  Driver Version: {parts[0].strip()}")
                print(f"  CUDA Version: {parts[1].strip()}")
    else:
        print("nvidia-smi: Command failed")
except FileNotFoundError:
    print("nvidia-smi: NOT FOUND (NVIDIA driver may not be installed)")
except subprocess.TimeoutExpired:
    print("nvidia-smi: TIMEOUT")
except Exception as e:
    print(f"nvidia-smi: ERROR - {e}")

# Try to get more detailed GPU info
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', '--format=csv,noheader'], 
                           capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("\nGPU Memory Status:")
        lines = result.stdout.strip().split('\n')
        for i, line in enumerate(lines):
            parts = line.split(', ')
            if len(parts) >= 4:
                print(f"GPU {i}: {parts[0].strip()}")
                print(f"  Total: {parts[1].strip()}")
                print(f"  Used: {parts[2].strip()}")
                print(f"  Free: {parts[3].strip()}")
except Exception:
    pass
print()

# PyTorch Geometric version
print("5. PYTORCH GEOMETRIC")
print("-" * 80)
try:
    import torch_geometric
    print(f"torch-geometric version: {torch_geometric.__version__}")
except ImportError:
    print("torch-geometric: NOT INSTALLED")
except Exception as e:
    print(f"torch-geometric: ERROR - {e}")
print()

# NumPy version
print("6. NUMPY VERSION")
print("-" * 80)
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError:
    print("NumPy: NOT INSTALLED")
except Exception as e:
    print(f"NumPy: ERROR - {e}")
print()

# Compatibility check
print("7. COMPATIBILITY WARNINGS")
print("-" * 80)
warnings = []

try:
    import torch
    if torch.cuda.is_available():
        pytorch_cuda = torch.version.cuda
        
        # Check PyTorch CUDA vs system CUDA
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=cuda_version', '--format=csv,noheader'], 
                                   capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                system_cuda = result.stdout.strip().split('\n')[0].strip()
                # CUDA versions should be compatible (PyTorch built with older CUDA can run on newer driver)
                # But we can check if they're wildly different
                try:
                    pytorch_cuda_major = int(pytorch_cuda.split('.')[0])
                    system_cuda_major = int(system_cuda.split('.')[0])
                    if abs(pytorch_cuda_major - system_cuda_major) > 1:
                        warnings.append(f"CUDA version mismatch: PyTorch built with CUDA {pytorch_cuda}, system has CUDA {system_cuda}")
                except:
                    pass
        except:
            pass
        
        # Check if PyCUDA can access GPU (may fail if PyTorch CUDA context is active)
        try:
            import pycuda.driver as drv
            drv.init()
            # Just check if we can get device count (won't create context)
            _ = drv.Device.count()
        except Exception as e:
            warnings.append(f"PyCUDA cannot access GPU (this is expected if PyTorch CUDA context is active): {e}")
        
        # Check GPU memory
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
            free_mem = total_mem - reserved
            
            if free_mem < 0.5:  # Less than 500MB free
                warnings.append(f"GPU {i} has very little free memory: {free_mem:.2f} GB free out of {total_mem:.2f} GB total")
            
            if reserved > 0 and free_mem < 1.0:
                warnings.append(f"GPU {i} memory may be fragmented or leaked: {reserved:.2f} GB reserved, only {free_mem:.2f} GB free")
except Exception as e:
    warnings.append(f"Error checking compatibility: {e}")

if warnings:
    for warning in warnings:
        print(f"⚠️  {warning}")
else:
    print("✓ No compatibility warnings detected")
print()

print("=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print()
print("If you're experiencing segmentation faults:")
print("1. Check GPU memory: Free GPU memory and try again")
print("2. Restart Python: Close all Python processes using CUDA")
print("3. Check PyTorch/CUDA compatibility: Ensure PyTorch CUDA version matches your system")
print("4. Use MOCU_BACKEND=torch: Set this environment variable to avoid PyCUDA conflicts")
print("5. Check for memory leaks: Monitor GPU memory usage during training")
print()

print("=" * 80)
print("MOCU BACKEND SETTINGS")
print("=" * 80)
import os
backend_mode = os.getenv('MOCU_BACKEND', 'auto')
print(f"Current MOCU_BACKEND: {backend_mode}")
if backend_mode == 'auto':
    try:
        import torch
        if torch.cuda.is_available():
            print("→ Auto-detection will use: torch (PyTorch CUDA is active)")
        else:
            print("→ Auto-detection will use: pycuda (PyTorch CUDA not active)")
    except:
        print("→ Auto-detection will use: pycuda")
elif backend_mode == 'torch':
    print("→ Forced to use: torch backend (safe with PyTorch)")
elif backend_mode == 'pycuda':
    print("→ Forced to use: pycuda backend (fast but may conflict with PyTorch)")
print()

