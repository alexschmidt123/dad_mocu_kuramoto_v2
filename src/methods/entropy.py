"""
ENTROPY OED Method

Simple heuristic-based method that selects experiments with maximum uncertainty.
Selects the pair (i, j) with the largest uncertainty bandwidth: max(a_upper - a_lower)

This method does NOT use any prediction model - it's purely based on current bounds.

In the paper: "ENTROPY" method
"""

import time
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.methods.base import OEDMethod
from src.core.mocu import MOCU


class ENTROPY_Method(OEDMethod):
    """
    Entropy-based (uncertainty-based) method for OED.
    
    Greedy heuristic: always select the pair with maximum uncertainty.
    No prediction model needed - purely based on current bounds.
    
    This is the fastest method but not necessarily the most effective.
    """
    
    def __init__(self, N, K_max, deltaT, MReal, TReal, it_idx):
        """
        Args:
            N: Number of oscillators
            K_max: Number of Monte Carlo samples for MOCU
            deltaT: Time step
            MReal: Number of time steps
            TReal: Time horizon
            it_idx: Number of MOCU averaging iterations
        """
        super().__init__(N, K_max, deltaT, MReal, TReal, it_idx)
        print(f"[ENTROPY] Initialized (greedy uncertainty)")
    
    def select_experiment(self, w, a_lower_bounds, a_upper_bounds, criticalK, isSynchronized, history):
        """
        Select next experiment using entropy (uncertainty) strategy.
        
        Selects the pair (i, j) with maximum uncertainty:
        argmax_{i,j} (a_upper[i,j] - a_lower[i,j])
        
        This is a simple greedy heuristic that prioritizes reducing
        the largest uncertainty first.
        """
        # Compute uncertainty (bandwidth) for each pair
        a_diff = np.triu(a_upper_bounds - a_lower_bounds, 1)
        
        # Mask out already selected experiments
        for (i, j), _ in history:
            a_diff[i, j] = 0.0
            if i > j:  # Ensure upper triangular
                a_diff[j, i] = 0.0
        
        # Find pair with maximum uncertainty
        valid_diff_values = a_diff[np.nonzero(a_diff)]
        
        if valid_diff_values.size == 0:
            print("[ENTROPY] Warning: No valid experiments left!")
            return -1, -1
        
        max_val = np.max(valid_diff_values)
        max_indices = np.where(a_diff == max_val)
        
        if len(max_indices[0]) > 1:
            # If multiple maximums, pick the first one
            max_i = int(max_indices[0][0])
            max_j = int(max_indices[1][0])
        else:
            max_i = int(max_indices[0])
            max_j = int(max_indices[1])
        
        print(f"[ENTROPY] Selected pair ({max_i}, {max_j}) with uncertainty {max_val:.4f}")
        
        return max_i, max_j

