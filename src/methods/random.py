"""
RANDOM OED Method

Baseline method that selects experiments uniformly at random.
Used as a comparison baseline to show the value of intelligent selection.

In the paper: "RANDOM" method
"""

import time
import numpy as np
import random
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.methods.base import OEDMethod
# MOCU computation handled by base class via torchdiffeq


class RANDOM_Method(OEDMethod):
    """
    Random baseline method for OED.
    
    Selects experiments uniformly at random from available pairs.
    This provides a lower bound on performance - any intelligent
    method should outperform random selection.
    """
    
    def __init__(self, N, K_max, deltaT, MReal, TReal, it_idx, seed=None):
        """
        Args:
            N: Number of oscillators
            K_max: Number of Monte Carlo samples for MOCU
            deltaT: Time step
            MReal: Number of time steps
            TReal: Time horizon
            it_idx: Number of MOCU averaging iterations
            seed: Random seed for reproducibility (optional)
        """
        super().__init__(N, K_max, deltaT, MReal, TReal, it_idx)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Pre-generate random sequence of all possible pairs
        self.experiment_sequence = []
        for i in range(N):
            for j in range(i + 1, N):
                self.experiment_sequence.append((i, j))
        
        random.shuffle(self.experiment_sequence)
        self.current_index = 0
        
        print(f"[RANDOM] Initialized (seed={seed})")
    
    def select_experiment(self, w, a_lower_bounds, a_upper_bounds, criticalK, isSynchronized, history):
        """
        Select next experiment randomly.
        
        Uses a pre-shuffled sequence to ensure all pairs are selected
        without replacement.
        """
        if self.current_index >= len(self.experiment_sequence):
            print("[RANDOM] Warning: All experiments have been selected!")
            return -1, -1
        
        selected_pair = self.experiment_sequence[self.current_index]
        self.current_index += 1
        
        print(f"[RANDOM] Selected pair {selected_pair} (index {self.current_index}/{len(self.experiment_sequence)})")
        
        return selected_pair

