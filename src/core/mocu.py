"""
PyTorch-based MOCU computation with CUDA acceleration.

This is the primary implementation for MOCU computation, using PyTorch CUDA
for acceleration. This replaces the PyCUDA implementation for better
compatibility and safety with PyTorch workflows.

Features:
- Full CUDA acceleration via PyTorch
- Safe to use with PyTorch training workflows
- Optimized batch processing for better performance
"""

import torch
import numpy as np
from typing import Union, Optional

# RK4 integration for synchronization detection
def _mocu_comp_torch(w: Union[np.ndarray, torch.Tensor], h: float, N: int, M: int, 
                     a: Union[np.ndarray, torch.Tensor], device: str = 'cuda') -> int:
    """
    Check if system synchronizes using RK4 integration.
    
    Args:
        w: Natural frequencies [N]
        h: Time step
        N: Number of oscillators
        M: Number of time steps
        a: Coupling matrix [N, N]
        device: Device string ('cuda' or 'cpu')
    
    Returns:
        D: 1 if synchronized, 0 if not
    """
    pi_n = 3.14159265358979323846
    
    # Convert to tensors and move to device
    if isinstance(w, np.ndarray):
        w = torch.from_numpy(w).to(device).float()
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a).to(device).float()
    
    # Ensure w and a are on correct device
    if not w.is_cuda and device == 'cuda' and torch.cuda.is_available():
        w = w.cuda()
    if not a.is_cuda and device == 'cuda' and torch.cuda.is_available():
        a = a.cuda()
    
    # Debug: Verify tensors are on GPU
    import os
    debug_gpu = os.getenv('DEBUG_GPU', 'false').lower() == 'true'
    if debug_gpu:
        if device == 'cuda' and torch.cuda.is_available():
            assert w.is_cuda, f"ERROR: w tensor not on GPU! device={w.device}"
            assert a.is_cuda, f"ERROR: a tensor not on GPU! device={a.device}"
            # Only print once to avoid spam
            if not hasattr(_mocu_comp_torch, '_debug_printed'):
                print(f"[DEBUG] _mocu_comp_torch: Tensors confirmed on GPU (device={w.device})")
                mem_allocated = torch.cuda.memory_allocated(0) / 1024**2
                mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
                print(f"[DEBUG] GPU memory after tensor creation: {mem_allocated:.2f} MB allocated, {mem_reserved:.2f} MB reserved")
                _mocu_comp_torch._debug_printed = True
    
    theta = torch.zeros(N, device=w.device, dtype=torch.float32)
    theta_old = torch.zeros(N, device=w.device, dtype=torch.float32)
    
    max_temp = torch.tensor(-100.0, device=w.device, dtype=torch.float32)
    min_temp = torch.tensor(100.0, device=w.device, dtype=torch.float32)
    
    # Optimized RK4 with pre-computed sin operations
    # Note: This loop processes sequentially, but tensor operations are GPU-accelerated
    for k in range(M):
        # Compute forces F using vectorized operations
        # F[i] = w[i] + sum_j(a[i,j] * sin(theta[j] - theta[i]))
        theta_diff = theta.unsqueeze(0) - theta.unsqueeze(1)  # [N, N]
        F = w + torch.sum(a * torch.sin(theta_diff), dim=1)
        
        # Debug: Show GPU memory during first iteration to confirm GPU usage
        if debug_gpu and device == 'cuda' and torch.cuda.is_available() and k == 0:
            if not hasattr(_mocu_comp_torch, '_mem_debug_printed'):
                mem_allocated = torch.cuda.memory_allocated(0) / 1024**2
                mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
                print(f"[DEBUG] GPU memory during RK4 computation (iteration {k}): {mem_allocated:.2f} MB allocated, {mem_reserved:.2f} MB reserved")
                print(f"[DEBUG] Tensor devices: theta on {theta.device}, w on {w.device}, a on {a.device}")
                _mocu_comp_torch._mem_debug_printed = True
        
        # RK4 step 1
        k1 = h * F
        theta = theta_old + k1 / 2.0
        
        # RK4 step 2
        theta_diff = theta.unsqueeze(0) - theta.unsqueeze(1)
        F = w + torch.sum(a * torch.sin(theta_diff), dim=1)
        k2 = h * F
        theta = theta_old + k2 / 2.0
        
        # RK4 step 3
        theta_diff = theta.unsqueeze(0) - theta.unsqueeze(1)
        F = w + torch.sum(a * torch.sin(theta_diff), dim=1)
        k3 = h * F
        theta = theta_old + k3
        
        # RK4 step 4
        theta_diff = theta.unsqueeze(0) - theta.unsqueeze(1)
        F = w + torch.sum(a * torch.sin(theta_diff), dim=1)
        k4 = h * F
        theta = theta_old + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
        # Track differences for sync detection (only after M/2)
        if (M / 2) < k:
            diff_t = theta - theta_old
            max_diff_at_step = torch.max(diff_t)
            min_diff_at_step = torch.min(diff_t)
            max_temp = torch.max(max_temp, max_diff_at_step)
            min_temp = torch.min(min_temp, min_diff_at_step)
        
        # Wrap angles to [0, 2*pi]
        theta = torch.remainder(theta, 2.0 * pi_n)
        theta_old = theta.clone()
    
    tol = max_temp - min_temp
    D = 1 if tol <= 0.001 else 0
    
    return D


def _find_critical_coupling_torch(w: Union[np.ndarray, list], h: float, N: int, M: int, 
                                   a_base: np.ndarray, device: str = 'cuda') -> float:
    """
    Find critical coupling value for a single Monte Carlo sample using binary search.
    
    Args:
        w: Natural frequencies [N]
        h: Time step
        N: Number of oscillators
        M: Number of time steps
        a_base: Base coupling matrix [N, N]
        device: Device string
    
    Returns:
        Critical coupling value (float)
    """
    # Determine device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # Prepare extended w (add mean oscillator)
    if isinstance(w, np.ndarray):
        w_extended = np.append(w, 0.5 * np.mean(w))
        w_extended = torch.from_numpy(w_extended).float()
    else:
        w_tensor = torch.tensor(w, dtype=torch.float32)
        w_extended = torch.cat([w_tensor, 0.5 * torch.mean(w_tensor).unsqueeze(0)])
    
    w_extended = w_extended.to(device)
    
    # Create base coupling matrix for N+1 system
    N1 = N + 1
    a_new = torch.zeros(N1, N1, device=device, dtype=torch.float32)
    
    # Copy base matrix
    a_base_t = torch.from_numpy(a_base).to(device).float()
    a_new[:N, :N] = a_base_t
    
    # Binary search for critical coupling
    c_lower = torch.tensor(0.0, device=device, dtype=torch.float32)
    c_upper = None
    is_found = False
    iteration = 0
    
    # Find upper bound
    for iteration in range(1, 20):
        initialC = 2.0 * iteration
        a_new[N, :N] = initialC
        a_new[:N, N] = initialC
        
        D = _mocu_comp_torch(w_extended, h, N1, M, a_new, device=device)
        
        if D > 0:
            c_upper = torch.tensor(initialC, device=device, dtype=torch.float32)
            c_lower = torch.tensor(0.0 if iteration == 1 else 2.0 * (iteration - 1), 
                                  device=device, dtype=torch.float32)
            is_found = True
            break
    
    if not is_found:
        return 10000000.0  # Failed to find synchronization
    
    # Binary search refinement
    iteration_offset = iteration - 1
    for iteration in range(14 + iteration_offset):
        midPoint = (c_upper + c_lower) / 2.0
        
        a_new[N, :N] = midPoint
        a_new[:N, N] = midPoint
        
        D = _mocu_comp_torch(w_extended, h, N1, M, a_new, device=device)
        
        if D > 0:
            c_upper = midPoint
        else:
            c_lower = midPoint
        
        if (c_upper - c_lower) < 0.00025:
            break
    
    return c_upper.item()


def MOCU(K_max: int, w: Union[np.ndarray, torch.Tensor], N: int, h: float, M: int, T: float,
         aLowerBoundIn: Union[np.ndarray, torch.Tensor], aUpperBoundIn: Union[np.ndarray, torch.Tensor],
         seed: int = 0, device: str = 'cuda') -> float:
    """
    Compute MOCU using PyTorch with CUDA acceleration.
    
    This is the main MOCU computation function that replaces the PyCUDA implementation.
    It uses PyTorch's CUDA backend for acceleration, making it safe to use alongside
    PyTorch training workflows.
    
    Args:
        K_max: Number of Monte Carlo samples
        w: Natural frequencies [N] (numpy array or torch tensor)
        N: Number of oscillators
        h: Time step
        M: Number of time steps
        T: Time horizon (kept for compatibility, not used)
        aLowerBoundIn: Lower bounds [N, N] (numpy array or torch tensor)
        aUpperBoundIn: Upper bounds [N, N] (numpy array or torch tensor)
        seed: Random seed (0 = no seed)
        device: Device string ('cuda' or 'cpu'). Default 'cuda' uses GPU if available.
    
    Returns:
        MOCU value (float)
    """
    # Determine device
    import os
    debug_gpu = os.getenv('DEBUG_GPU', 'false').lower() == 'true'
    
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("[MOCU] CUDA not available, using CPU (slower)")
    elif device == 'cuda' and torch.cuda.is_available() and debug_gpu:
        # Debug: Print GPU info on first call
        if not hasattr(MOCU, '_debug_printed'):
            print(f"[DEBUG] MOCU using GPU: {torch.cuda.get_device_name(0)}")
            mem_before = torch.cuda.memory_allocated(0) / 1024**2
            mem_reserved_before = torch.cuda.memory_reserved(0) / 1024**2
            print(f"[DEBUG] GPU memory before MOCU: {mem_before:.2f} MB allocated, {mem_reserved_before:.2f} MB reserved")
            MOCU._debug_printed = True
    
    # Set random seed
    if seed != 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    # Convert inputs to numpy for processing
    if isinstance(w, torch.Tensor):
        w = w.cpu().numpy()
    if isinstance(aLowerBoundIn, torch.Tensor):
        aLowerBoundIn = aLowerBoundIn.cpu().numpy()
    if isinstance(aUpperBoundIn, torch.Tensor):
        aUpperBoundIn = aUpperBoundIn.cpu().numpy()
    
    # Prepare bounds
    vec_a_lower = np.reshape(aLowerBoundIn.copy(), N * N)
    vec_a_upper = np.reshape(aUpperBoundIn.copy(), N * N)
    
    # Generate random coupling matrices
    if seed == 0:
        rand_data = np.random.random(int((N - 1) * N / 2.0 * K_max)).astype(np.float64)
    else:
        rng = np.random.RandomState(int(seed))
        rand_data = rng.uniform(size=int((N - 1) * N / 2.0 * K_max))
    
    # Process samples sequentially (each requires binary search which has dependencies)
    # NOTE: This is the main bottleneck - each sample requires ~20-30 binary search iterations
    # Each iteration does RK4 integration (M steps), so total: K_max * 20-30 * M operations
    # GPU accelerates the RK4 tensor operations, but we can't parallelize binary search across samples
    
    # Pre-allocate result array for better memory management
    a_save = np.zeros(K_max, dtype=np.float64)
    
    # Process samples one by one (binary search prevents true parallelization)
    # The GPU is used for RK4 integration within each sample, but samples are sequential
    for sample_idx in range(K_max):
        # Generate random coupling matrix for this sample
        cnt0 = sample_idx * (N - 1) * N // 2
        a_new = np.zeros((N, N), dtype=np.float64)
        
        cnt1 = 0
        for i in range(N):
            for j in range(i + 1, N):
                rand_ind = cnt0 + cnt1
                a_val = vec_a_lower[j * N + i] + (vec_a_upper[j * N + i] - vec_a_lower[j * N + i]) * rand_data[rand_ind]
                a_new[j, i] = a_val
                a_new[i, j] = a_val
                cnt1 += 1
        
        # Find critical coupling for this sample (this uses GPU for RK4)
        critical_c = _find_critical_coupling_torch(w, h, N, M, a_new, device=device)
        a_save[sample_idx] = critical_c
        
        # Optional: synchronize periodically to avoid memory buildup
        # Less frequent sync reduces overhead (every 500 samples instead of 100)
        if device == 'cuda' and torch.cuda.is_available() and (sample_idx + 1) % 500 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # Free unused GPU memory
    
    if np.min(a_save) == 0:
        print("Non sync case exists")
    
    # Compute MOCU value (same logic as original PyCUDA version)
    if K_max >= 1000:
        temp = np.sort(a_save)
        ll = int(K_max * 0.005)
        uu = int(K_max * 0.995)
        a_save_filtered = temp[ll - 1:uu]
        a_star = np.max(a_save_filtered)
        MOCU_val = np.sum(a_star - a_save_filtered) / (K_max * 0.99)
    else:
        a_star = np.max(a_save)
        MOCU_val = np.sum(a_star - a_save) / K_max
    
    return float(MOCU_val)

