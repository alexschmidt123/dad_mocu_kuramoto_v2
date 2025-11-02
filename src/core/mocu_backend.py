"""
Smart backend selector for MOCU computation.

This module prevents PyCUDA/PyTorch conflicts by:
1. Checking if PyTorch CUDA is already active
2. If YES → use PyTorch implementation (safe)
3. If NO → use PyCUDA implementation (fast)
"""

import os

def get_backend_mode():
    """
    Decide which backend to use.
    
    Returns:
        'pycuda' or 'torch'
    """
    # Option 1: User explicitly sets environment variable
    mode = os.getenv('MOCU_BACKEND', 'auto')
    
    if mode != 'auto':
        return mode  # User choice overrides
    
    # Option 2: Auto-detect what's safe
    try:
        import torch
        # Check if PyTorch has been imported and CUDA is available
        # If PyTorch is imported, there's a risk CUDA context is active
        # (even if not explicitly initialized yet, importing torch may initialize it)
        if torch.cuda.is_available():
            # PyTorch CUDA is AVAILABLE → use torch backend to be safe
            # This prevents PyCUDA from creating a conflicting context
            return 'torch'
        else:
            # PyTorch CUDA not available → safe to use PyCUDA
            return 'pycuda'
    except ImportError:
        # PyTorch not installed → use PyCUDA
        return 'pycuda'


def load_mocu_backend(mode=None):
    """
    Load the appropriate MOCU implementation.
    
    Returns:
        MOCU function (either from mocu_cuda.py or mocu_torch.py)
    """
    if mode is None:
        mode = get_backend_mode()
    
    if mode == 'pycuda':
        try:
            from .mocu_cuda import MOCU
            print("[MOCU] Using PyCUDA backend (fast)")
            return MOCU
        except ImportError:
            print("[MOCU] PyCUDA not available, using PyTorch")
            from .mocu_torch import MOCU_torch
            return MOCU_torch
    
    else:  # mode == 'torch'
        from .mocu_torch import MOCU_torch
        print("[MOCU] Using PyTorch backend (safe)")
        return MOCU_torch


# This is what other files import
MOCU = load_mocu_backend()
