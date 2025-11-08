"""
torchdiffeq-based MOCU computation for optimal experimental design.

This implementation uses torchdiffeq to solve the Kuramoto ODE system,
replacing PyCUDA for better compatibility with PyTorch CUDA.

Based on the PyCUDA implementation but using torchdiffeq for ODE solving.
"""

import numpy as np
import torch
from typing import Union, Optional

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("[WARNING] torchdiffeq not available. Install with: pip install torchdiffeq")


class KuramotoODE(torch.nn.Module):
    """
    Kuramoto oscillator system: dθ/dt = w + Σ a[i,j] * sin(θ[j] - θ[i])
    
    This is a PyTorch module that can be used with torchdiffeq for ODE solving.
    """
    
    def __init__(self, w, a, device='cuda'):
        """
        Args:
            w: Natural frequencies [N] (numpy array or torch tensor)
            a: Coupling matrix [N, N] (numpy array or torch tensor)
            device: 'cuda' or 'cpu'
        """
        super().__init__()
        if isinstance(w, np.ndarray):
            w = torch.tensor(w, dtype=torch.float32)
        if isinstance(a, np.ndarray):
            a = torch.tensor(a, dtype=torch.float32)
        
        self.register_buffer('w', w.to(device))
        self.register_buffer('a', a.to(device))
        self.N = len(w)
        self.device = device
    
    def forward(self, t, theta):
        """
        Compute dθ/dt for Kuramoto system.
        
        Args:
            t: Time (scalar, not used but required by torchdiffeq)
            theta: Phase angles [N] (torch tensor)
        
        Returns:
            dtheta_dt: Phase derivatives [N]
        """
        # Expand theta for broadcasting: [N] -> [N, 1] and [1, N]
        theta_i = theta.unsqueeze(1)  # [N, 1]
        theta_j = theta.unsqueeze(0)   # [1, N]
        
        # Compute coupling term: Σ a[i,j] * sin(θ[j] - θ[i])
        # theta_j - theta_i: [1, N] - [N, 1] = [N, N] (broadcasting)
        coupling = torch.sum(self.a * torch.sin(theta_j - theta_i), dim=1)  # [N]
        
        # dθ/dt = w + coupling
        dtheta_dt = self.w + coupling
        
        return dtheta_dt


def solve_kuramoto_ode(w, a, h, M, device='cuda', method='rk4', timeout=5.0):
    """
    Solve Kuramoto ODE using torchdiffeq.
    
    Args:
        w: Natural frequencies [N] (numpy array)
        a: Coupling matrix [N, N] (numpy array)
        h: Time step
        M: Number of time steps
        device: 'cuda' or 'cpu'
        method: ODE solver method ('rk4', 'euler', 'adaptive_heun', etc.)
        timeout: Maximum time in seconds for ODE solving (default: 5.0)
    
    Returns:
        theta_trajectory: [M, N] phase angles over time (numpy array)
    
    Raises:
        RuntimeError: If ODE solving fails or times out
    """
    if not TORCHDIFFEQ_AVAILABLE:
        raise RuntimeError("torchdiffeq not available. Install with: pip install torchdiffeq")
    
    import time
    
    # Convert to tensors
    if isinstance(w, np.ndarray):
        w_tensor = torch.tensor(w, dtype=torch.float32, device=device)
    else:
        w_tensor = w.to(device)
    
    if isinstance(a, np.ndarray):
        a_tensor = torch.tensor(a, dtype=torch.float32, device=device)
    else:
        a_tensor = a.to(device)
    
    # Create ODE system
    ode_system = KuramotoODE(w_tensor, a_tensor, device=device)
    
    # Initial condition: all phases start at 0
    theta0 = torch.zeros(len(w), dtype=torch.float32, device=device)
    
    # Time points
    t = torch.linspace(0, M * h, M, dtype=torch.float32, device=device)
    
    # Solve ODE with timeout protection
    start_time = time.time()
    try:
        with torch.no_grad():  # No gradients needed for MOCU computation
            theta_trajectory = odeint(ode_system, theta0, t, method=method)
            
            # Explicit synchronization before CPU transfer to avoid hangs
            if device == 'cuda' and torch.cuda.is_available():
                # Don't use torch.cuda.synchronize() - it can hang
                # Instead, force completion by accessing tensor data
                _ = theta_trajectory[0, 0].item()  # Force computation to complete
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise RuntimeError(f"ODE solving exceeded timeout ({timeout}s): {elapsed:.2f}s")
            
            # Convert to numpy and return
            result = theta_trajectory.cpu().numpy()
            
            # Verify result is valid
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                raise RuntimeError("ODE solution contains NaN or Inf values")
            
            return result
            
    except RuntimeError as e:
        # Re-raise timeout or validation errors
        if "timeout" in str(e).lower() or "nan" in str(e).lower() or "inf" in str(e).lower():
            raise
        # For other RuntimeErrors, check if it's a CUDA issue
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise RuntimeError(f"ODE solving exceeded timeout ({timeout}s): {elapsed:.2f}s") from e
        raise RuntimeError(f"ODE solving failed after {elapsed:.2f}s: {e}") from e
    except Exception as e:
        elapsed = time.time() - start_time
        raise RuntimeError(f"ODE solving failed after {elapsed:.2f}s: {e}") from e


def check_synchronization(theta_trajectory, M):
    """
    Check if system is synchronized based on phase trajectory.
    
    Uses the same logic as the original CPU implementation:
    - Check phase differences in second half of trajectory
    - If max - min <= 1e-3, system is synchronized
    
    Args:
        theta_trajectory: [M, N] phase angles (numpy array)
        M: Number of time steps
    
    Returns:
        is_synchronized: 1 if synchronized, 0 if not
    """
    # Use second half of trajectory to check stability
    second_half = theta_trajectory[M//2:, :]
    
    # Compute phase differences (theta - theta_old)
    # For each time step, compute difference from previous
    diff_t = np.diff(second_half, axis=0)
    
    # Check if all differences are small (synchronized)
    # Original logic: max - min <= 1e-3
    tol = np.max(diff_t) - np.min(diff_t)
    
    return 1 if tol <= 1e-3 else 0


def binary_search_critical_coupling(w, a_lower, a_upper, a_sample, h, M, device='cuda'):
    """
    Binary search for critical coupling strength using torchdiffeq.
    
    This implements the same binary search logic as the PyCUDA version:
    1. Create extended system with N+1 oscillators (add mean oscillator)
    2. Use binary search to find critical coupling c such that system synchronizes
    3. Return the critical coupling value
    
    Args:
        w: Natural frequencies [N] (numpy array)
        a_lower: Lower bounds [N, N] (numpy array)
        a_upper: Upper bounds [N, N] (numpy array)
        a_sample: Sample coupling matrix [N, N] (numpy array)
        h: Time step
        M: Number of time steps
        device: 'cuda' or 'cpu'
    
    Returns:
        critical_coupling: Critical coupling strength (float)
    """
    N = len(w)
    N_extended = N + 1
    
    # Create extended w (add mean oscillator)
    w_extended = np.append(w, 0.5 * np.mean(w))
    
    # Create extended coupling matrix
    # Start with sample values for N×N submatrix
    a_extended = np.zeros((N_extended, N_extended))
    a_extended[:N, :N] = a_sample.copy()
    
    # Find initial coupling value where system synchronizes
    is_found = False
    initial_c = 0
    
    # Try increasing coupling values until synchronization is found
    for iteration in range(1, 20):
        initial_c = 2 * iteration
        
        # Set coupling to mean oscillator
        for i in range(N):
            a_extended[i, N] = initial_c
            a_extended[N, i] = initial_c
        
        # Solve ODE and check synchronization
        theta_traj = solve_kuramoto_ode(w_extended, a_extended, h, M, device=device)
        is_sync = check_synchronization(theta_traj, M)
        
        if is_sync == 1:
            is_found = True
            break
    
    if not is_found:
        # System never synchronizes with this sample
        return 10000000.0
    
    # Binary search for precise critical coupling
    c_lower = 0.0
    c_upper = initial_c
    iteration_offset = iteration - 1
    
    for iteration in range(14 + iteration_offset):
        mid_point = (c_upper + c_lower) / 2.0
        
        # Set coupling to mean oscillator
        for i in range(N):
            a_extended[i, N] = mid_point
            a_extended[N, i] = mid_point
        
        # Solve ODE and check synchronization
        theta_traj = solve_kuramoto_ode(w_extended, a_extended, h, M, device=device)
        is_sync = check_synchronization(theta_traj, M)
        
        if is_sync == 1:
            c_upper = mid_point
        else:
            c_lower = mid_point
        
        # Convergence check
        if (c_upper - c_lower) < 0.00025:
            break
    
    return c_upper


def MOCU_torchdiffeq(K_max: int, w: np.ndarray, N: int, h: float, M: int, T: float,
                     aLowerBoundIn: np.ndarray, aUpperBoundIn: np.ndarray,
                     seed: int = 0, device: str = 'cuda') -> float:
    """
    Compute MOCU using torchdiffeq for ODE solving.
    
    This replaces PyCUDA-based MOCU computation with torchdiffeq, which
    is fully compatible with PyTorch CUDA and avoids context conflicts.
    
    Args:
        K_max: Number of Monte Carlo samples
        w: Natural frequencies [N]
        N: Number of oscillators
        h: Time step
        M: Number of time steps
        T: Time horizon (kept for compatibility, not used)
        aLowerBoundIn: Lower bounds [N, N]
        aUpperBoundIn: Upper bounds [N, N]
        seed: Random seed (0 = no seed)
        device: 'cuda' or 'cpu'
    
    Returns:
        MOCU value (float)
    """
    if not TORCHDIFFEQ_AVAILABLE:
        raise RuntimeError("torchdiffeq not available. Install with: pip install torchdiffeq")
    
    # Set random seed
    if seed != 0:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    # Generate random coupling matrices and compute critical couplings
    a_star_values = []
    
    # Process samples (can be batched for better GPU utilization)
    # For now, process sequentially to match PyCUDA behavior
    for k in range(K_max):
        # Generate random coupling matrix
        a_sample = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                a_ij = np.random.uniform(aLowerBoundIn[i, j], aUpperBoundIn[i, j])
                a_sample[i, j] = a_ij
                a_sample[j, i] = a_ij
        
        # Binary search for critical coupling
        a_star = binary_search_critical_coupling(
            w, aLowerBoundIn, aUpperBoundIn, a_sample, h, M, device=device
        )
        a_star_values.append(a_star)
    
    a_star_values = np.array(a_star_values)
    
    # Check for non-synchronized cases
    if np.min(a_star_values) == 0:
        print("Non sync case exists")
    
    # Compute MOCU value (same logic as PyCUDA version)
    if K_max >= 1000:
        # Filter outliers (keep 99% of data, remove 0.5% from each end)
        temp = np.sort(a_star_values)
        ll = int(K_max * 0.005)
        uu = int(K_max * 0.995)
        a_save_filtered = temp[ll - 1:uu]
        a_star = np.max(a_save_filtered)
        MOCU_val = np.sum(a_star - a_save_filtered) / (K_max * 0.99)
    else:
        a_star = np.max(a_star_values)
        MOCU_val = np.sum(a_star - a_star_values) / K_max
    
    return float(MOCU_val)

