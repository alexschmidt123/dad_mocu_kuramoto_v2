"""
Smart backend selector for MOCU computation.

This module prevents PyCUDA/PyTorch conflicts by:
1. Checking if PyTorch CUDA is already active
2. If YES → use PyTorch implementation (safe)
3. If NO → use PyCUDA implementation (fast)

CRITICAL: When MOCU_BACKEND=torch is set, PyCUDA will NEVER be imported,
even if someone tries to import mocu_cuda directly.
"""

import os

def get_backend_mode():
    """
    Decide which backend to use.
    
    Returns:
        'pycuda' or 'torch'
    """
    # Option 1: User explicitly sets environment variable (CHECKED FIRST!)
    mode = os.getenv('MOCU_BACKEND', 'auto')
    
    if mode != 'auto':
        return mode  # User choice overrides everything
    
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


# Lazy loading: only load backend when MOCU is actually called
_MOCU_func = None

def _get_mocu():
    """
    Lazy loader for MOCU function.
    Only imports and loads the backend when MOCU is actually accessed.
    """
    global _MOCU_func
    if _MOCU_func is None:
        mode = get_backend_mode()
        
        if mode == 'pycuda':
            try:
                from .mocu_cuda import MOCU
                print("[MOCU] Using PyCUDA backend (fast)")
                _MOCU_func = MOCU
            except ImportError:
                print("[MOCU] PyCUDA not available, using PyTorch")
                from .mocu_torch import MOCU_torch
                _MOCU_func = MOCU_torch
        else:  # mode == 'torch'
            from .mocu_torch import MOCU_torch
            print("[MOCU] Using PyTorch backend (safe)")
            _MOCU_func = MOCU_torch
    
    return _MOCU_func


class MOCUWrapper:
    """
    Wrapper class that lazy-loads the actual MOCU function.
    This prevents importing PyCUDA at module import time.
    """
    def __call__(self, *args, **kwargs):
        func = _get_mocu()
        return func(*args, **kwargs)


# This is what other files import - it's a callable that lazy-loads the backend
MOCU = MOCUWrapper()
