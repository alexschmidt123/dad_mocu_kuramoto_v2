"""
Sampling-Based MOCU Computation (Ground Truth).

This is NOT a learned predictor - it's the exact (but slow) computation
using PyCUDA-accelerated Monte Carlo sampling.

NOTE: This class uses PyCUDA MOCU computation (mocu_pycuda.py), consistent
with the reference implementation. This class is primarily used for 
predictor evaluation/comparison in compare_predictors.py as a ground 
truth baseline.

This is what the paper calls "Sampling-based" in Table 1.
"""

import numpy as np


class SamplingBasedMOCU:
    """
    Ground truth MOCU computation using Monte Carlo sampling.
    
        This is NOT a learned predictor - it's the exact (but slow) computation
        using PyTorch CUDA-accelerated integration.
    
    This class uses PyCUDA MOCU computation (mocu_pycuda.py), consistent
    with the reference implementation. This class is primarily used for 
    predictor evaluation/comparison in compare_predictors.py as a ground 
    truth baseline.
    
    This is what the paper calls "Sampling-based" in Table 1.
    """
    
    def __init__(self, K_max=20480, h=1.0/160.0, T=5.0):
        """
        Args:
            K_max: Number of Monte Carlo samples
            h: Time step for RK4 integration
            T: Time horizon
        
        NOTE: MOCU is imported lazily here (inside __init__) to avoid
        importing when this class is defined (module import).
        """
        # LAZY IMPORT: Only import MOCU_pycuda when an instance is created
        # Import happens here, not at module level
        from ...core.mocu_pycuda import MOCU_pycuda
        self.MOCU = MOCU_pycuda
        self.K_max = K_max
        self.h = h
        self.M = int(T / h)
        self.T = T
    
    def compute(self, w, a_lower, a_upper, num_iterations=10):
        """
        Compute MOCU using Monte Carlo sampling (ground truth).
        
        Args:
            w: Natural frequencies [N]
            a_lower: Lower bounds [N, N]
            a_upper: Upper bounds [N, N]
            num_iterations: Number of times to compute and average
        
        Returns:
            mocu: Ground truth MOCU value
        """
        N = len(w)
        mocu_vals = np.zeros(num_iterations)
        
        for i in range(num_iterations):
            mocu_vals[i] = self.MOCU(
                self.K_max, w, N, self.h, self.M, self.T,
                a_lower, a_upper, 0
            )
        
        return np.mean(mocu_vals)
    
    def __call__(self, w, a_lower, a_upper):
        """Allow calling as a function."""
        return self.compute(w, a_lower, a_upper)

