"""
CPU-based MOCU computation for evaluation (small data sizes).

This implementation uses NumPy for MOCU computation on CPU.
It's slower than PyCUDA but avoids CUDA context conflicts with PyTorch.

Use this for evaluation when:
- Small data sizes (RANDOM, ENTROPY methods)
- PyCUDA conflicts with PyTorch CUDA
- CPU is fast enough for small computations

For large-scale data generation, use PyCUDA (mocu_pycuda.py).
"""

import numpy as np
from src.core.sync_detection import determineSyncN


def MOCU_cpu(K_max: int, w: np.ndarray, N: int, h: float, M: int, T: float,
             aLowerBoundIn: np.ndarray, aUpperBoundIn: np.ndarray,
             seed: int = 0) -> float:
    """
    Compute MOCU using CPU (NumPy) implementation.
    
    This is a CPU-based version suitable for evaluation with small data sizes.
    Based on the same algorithm as PyCUDA version but runs on CPU.
    
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
    
    Returns:
        MOCU value (float)
    """
    # Set random seed
    if seed != 0:
        np.random.seed(seed)
    
    # Prepare extended w (add mean oscillator for N+1 system)
    w_extended = np.append(w, 0.5 * np.mean(w))
    
    # Prepare bounds (reshape to vector for easier indexing)
    a_lower_vec = aLowerBoundIn.flatten()
    a_upper_vec = aUpperBoundIn.flatten()
    
    # Generate random coupling matrices
    num_pairs = (N - 1) * N // 2
    if seed == 0:
        rand_data = np.random.random(num_pairs * K_max)
    else:
        rng = np.random.RandomState(int(seed))
        rand_data = rng.uniform(size=num_pairs * K_max)
    
    # Pre-allocate result array
    a_save = np.zeros(K_max)
    
    # Extended system size (N+1)
    N_extended = N + 1
    
    # Process each Monte Carlo sample
    for k_idx in range(K_max):
        # Build coupling matrix for extended system
        a_new = np.zeros((N_extended, N_extended))
        
        # Fill in original N x N part
        rand_offset = k_idx * num_pairs
        pair_idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                rand_val = rand_data[rand_offset + pair_idx]
                # Sample from bounds
                a_val = a_lower_vec[j * N + i] + (
                    a_upper_vec[j * N + i] - a_lower_vec[j * N + i]
                ) * rand_val
                a_new[j, i] = a_val
                a_new[i, j] = a_val
                pair_idx += 1
        
        # Find critical coupling using binary search
        # Add mean oscillator with coupling strength c
        is_found = False
        initial_c = 0
        
        # First: Find initial C that causes synchronization
        for iteration in range(1, 20):
            initial_c = 2 * iteration
            # Add coupling to mean oscillator
            a_new_candidate = a_new.copy()
            for i in range(N):
                a_new_candidate[i, N] = initial_c
                a_new_candidate[N, i] = initial_c
            
            # Check synchronization
            D = determineSyncN(w_extended, h, N_extended, M, a_new_candidate)
            if D > 0:
                is_found = True
                break
        
        if not is_found:
            # No synchronization found even at high coupling
            a_save[k_idx] = 10000000.0
            continue
        
        # Binary search for critical coupling
        c_lower = 0.0
        c_upper = initial_c
        iteration_offset = iteration - 1
        
        for iteration in range(14 + iteration_offset):
            mid_point = (c_upper + c_lower) / 2.0
            
            # Test at midpoint
            a_new_test = a_new.copy()
            for i in range(N):
                a_new_test[i, N] = mid_point
                a_new_test[N, i] = mid_point
            
            D = determineSyncN(w_extended, h, N_extended, M, a_new_test)
            
            if D > 0:
                c_upper = mid_point
            else:
                c_lower = mid_point
            
            # Early termination if converged
            if (c_upper - c_lower) < 0.00025:
                break
        
        a_save[k_idx] = c_upper
    
    # Check for non-sync cases
    if np.min(a_save) == 0:
        print("Non sync case exists")
    
    # Compute MOCU value (same logic as PyCUDA version)
    if K_max >= 1000:
        # Filter outliers (0.5% on each side)
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

