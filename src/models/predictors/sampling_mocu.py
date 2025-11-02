"""
Sampling-Based MOCU Computation (Ground Truth).

This is NOT a learned predictor - it's the exact (but slow) computation
using CUDA-accelerated Monte Carlo sampling.

NOTE: ODE method uses MOCU() directly from mocu_cuda.py, NOT this class.
This class is primarily used for predictor evaluation/comparison in 
compare_predictors.py as a ground truth baseline.

This is what the paper calls "Sampling-based" in Table 1.

CRITICAL: This file is separate from predictors.py to ensure that when
MPNN predictors are imported, this class (which imports mocu_cuda) is
never loaded. This prevents PyCUDA context initialization during DAD training.
"""

import numpy as np


class SamplingBasedMOCU:
    """
    Ground truth MOCU computation using Monte Carlo sampling.
    
    This is NOT a learned predictor - it's the exact (but slow) computation
    using CUDA-accelerated integration.
    
    NOTE: ODE method uses MOCU() directly from mocu_cuda.py, NOT this class.
    This class is primarily used for predictor evaluation/comparison in 
    compare_predictors.py as a ground truth baseline.
    
    This is what the paper calls "Sampling-based" in Table 1.
    """
    
    def __init__(self, K_max=20480, h=1.0/160.0, T=5.0):
        """
        Args:
            K_max: Number of Monte Carlo samples
            h: Time step for RK4 integration
            T: Time horizon
        
        NOTE: MOCU is imported lazily here (inside __init__) to avoid
        initializing PyCUDA context when this class is defined (module import).
        PyCUDA context is only created when an instance is actually created.
        """
        # LAZY IMPORT: Only import MOCU when an instance is created
        # This prevents PyCUDA context initialization during module import
        # Import happens here, not at module level
        from ...core.mocu_backend import MOCU
        self.MOCU = MOCU
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
                a_lower, a_upper, seed=0
            )
        
        return np.mean(mocu_vals)
    
    def __call__(self, w, a_lower, a_upper):
        """Allow calling as a function."""
        return self.compute(w, a_lower, a_upper)

