"""
ODE-based OED Method (Sampling-based, Ground Truth)

Uses exact MOCU computation (Monte Carlo sampling with CUDA acceleration)
to greedily select experiments. This is the most accurate but slowest method.

In the paper:
- "ODE" = static version (compute once)
- "iODE" = iterative version (re-compute at each step)
"""

import time
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.methods.base import OEDMethod

# Use PyCUDA for ODE methods (as in original paper 2023)
try:
    from src.core.mocu_pycuda import MOCU_pycuda as MOCU_pycuda_impl
    PYCUDA_AVAILABLE = True
except (ImportError, RuntimeError):
    PYCUDA_AVAILABLE = False
    print("[WARNING] PyCUDA not available for ODE methods. ODE methods will fail.")


class ODE_Method(OEDMethod):
    """
    ODE-based method using sampling (ground truth MOCU computation).
    
    This method computes the expected MOCU for all possible experiments
    using Monte Carlo sampling. It's the most accurate but computationally expensive.
    
    Static version: computes R matrix once and reuses it.
    """
    
    def __init__(self, N, K_max, deltaT, MReal, TReal, it_idx, 
                 MVirtual=None, TVirtual=None, gpu_id=0):
        """
        Args:
            N: Number of oscillators
            K_max: Number of Monte Carlo samples for MOCU
            deltaT: Time step
            MReal: Number of time steps for MOCU evaluation
            TReal: Time horizon for MOCU evaluation
            it_idx: Number of MOCU averaging iterations
            MVirtual: Number of time steps for prediction (defaults to MReal)
            TVirtual: Time horizon for prediction (defaults to TReal)
            gpu_id: GPU device ID (for explicit CUDA usage, avoids PyCUDA conflicts)
        """
        super().__init__(N, K_max, deltaT, MReal, TReal, it_idx)
        self.MVirtual = MVirtual if MVirtual is not None else MReal
        self.TVirtual = TVirtual if TVirtual is not None else TReal
        self.R_matrix = np.zeros((N, N))
        
        if not PYCUDA_AVAILABLE:
            raise RuntimeError("PyCUDA is REQUIRED for ODE methods but not available. Install with: pip install pycuda")
        
        print(f"[ODE] Initialized (static version, using PyCUDA)")
    
    def _compute_expected_mocu_matrix(self, w, a_lower_bounds, a_upper_bounds):
        """
        Compute R matrix using ground truth MOCU (sampling-based).
        
        For each possible experiment (i, j):
        - Simulate synchronized observation → compute MOCU_syn
        - Simulate non-synchronized observation → compute MOCU_nonsyn
        - Expected MOCU = P_syn * MOCU_syn + P_nonsyn * MOCU_nonsyn
        
        This is VERY slow but gives exact expected values.
        """
        R_matrix = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            for j in range(i + 1, self.N):
                # Compute f_inv (critical coupling for this pair)
                w_i = w[i]
                w_j = w[j]
                f_inv = 0.5 * np.abs(w_i - w_j)
                
                # Scenario 1: Synchronized observation
                a_upper_syn = a_upper_bounds.copy()
                a_lower_syn = a_lower_bounds.copy()
                
                a_tilde = min(max(f_inv, a_lower_bounds[i, j]), a_upper_bounds[i, j])
                a_lower_syn[j, i] = a_tilde
                a_lower_syn[i, j] = a_tilde
                
                P_syn = (a_upper_bounds[i, j] - a_tilde) / (
                    a_upper_bounds[i, j] - a_lower_bounds[i, j] + 1e-10
                )
                
                # Compute MOCU for synchronized scenario (using PyCUDA)
                mocu_vals_syn = np.zeros(self.it_idx)
                for l in range(self.it_idx):
                    mocu_vals_syn[l] = MOCU_pycuda_impl(
                        self.K_max, w, self.N, self.deltaT, 
                        self.MVirtual, self.TVirtual,
                        a_lower_syn, a_upper_syn, seed=0
                    )
                MOCU_syn = np.mean(mocu_vals_syn)
                
                # Scenario 2: Non-synchronized observation
                a_upper_nonsyn = a_upper_bounds.copy()
                a_lower_nonsyn = a_lower_bounds.copy()
                
                a_upper_nonsyn[i, j] = a_tilde
                a_upper_nonsyn[j, i] = a_tilde
                
                P_nonsyn = (a_tilde - a_lower_bounds[i, j]) / (
                    a_upper_bounds[i, j] - a_lower_bounds[i, j] + 1e-10
                )
                
                # Compute MOCU for non-synchronized scenario (using PyCUDA)
                mocu_vals_nonsyn = np.zeros(self.it_idx)
                for l in range(self.it_idx):
                    mocu_vals_nonsyn[l] = MOCU_pycuda_impl(
                        self.K_max, w, self.N, self.deltaT,
                        self.MVirtual, self.TVirtual,
                        a_lower_nonsyn, a_upper_nonsyn, seed=0
                    )
                MOCU_nonsyn = np.mean(mocu_vals_nonsyn)
                
                # Expected MOCU
                R_matrix[i, j] = P_syn * MOCU_syn + P_nonsyn * MOCU_nonsyn
        
        return R_matrix
    
    def select_experiment(self, w, a_lower_bounds, a_upper_bounds, criticalK, isSynchronized, history):
        """
        Select next experiment using static ODE strategy.
        
        Computes R matrix only once, then greedily selects from it.
        """
        # Compute R matrix only on first call
        if not np.any(self.R_matrix):
            print("[ODE] Computing expected MOCU matrix (static, once only)...")
            print("[ODE] Warning: This may take a LONG time (exact sampling)...")
            self.R_matrix = self._compute_expected_mocu_matrix(w, a_lower_bounds, a_upper_bounds)
        
        # Mask out already selected experiments
        for (i, j), _ in history:
            self.R_matrix[i, j] = 0.0
            self.R_matrix[j, i] = 0.0
        
        # Find experiment with minimum expected MOCU
        valid_R_values = self.R_matrix[np.nonzero(self.R_matrix)]
        
        if valid_R_values.size == 0:
            print("[ODE] Warning: No valid experiments left!")
            return -1, -1
        
        min_val = np.min(valid_R_values)
        min_indices = np.where(self.R_matrix == min_val)
        
        if len(min_indices[0]) > 1:
            min_i = int(min_indices[0][0])
            min_j = int(min_indices[1][0])
        else:
            min_i = int(min_indices[0])
            min_j = int(min_indices[1])
        
        return min_i, min_j


class iODE_Method(OEDMethod):
    """
    Iterative ODE method using sampling (ground truth MOCU computation).
    
    Re-computes the expected MOCU at each step based on updated bounds.
    This is EXTREMELY slow but maximally accurate and adaptive.
    """
    
    def __init__(self, N, K_max, deltaT, MReal, TReal, it_idx,
                 MVirtual=None, TVirtual=None, gpu_id=0):
        """
        Args:
            N: Number of oscillators
            K_max: Number of Monte Carlo samples for MOCU
            deltaT: Time step
            MReal: Number of time steps for MOCU evaluation
            TReal: Time horizon for MOCU evaluation
            it_idx: Number of MOCU averaging iterations
            MVirtual: Number of time steps for prediction (defaults to MReal)
            TVirtual: Time horizon for prediction (defaults to TReal)
            gpu_id: GPU device ID (for explicit CUDA usage, avoids PyCUDA conflicts)
        """
        super().__init__(N, K_max, deltaT, MReal, TReal, it_idx)
        self.MVirtual = MVirtual if MVirtual is not None else MReal
        self.TVirtual = TVirtual if TVirtual is not None else TReal
        
        if not PYCUDA_AVAILABLE:
            raise RuntimeError("PyCUDA is REQUIRED for iODE methods but not available. Install with: pip install pycuda")
        
        print(f"[iODE] Initialized (iterative version, using PyCUDA)")
    
    def _compute_expected_mocu_matrix(self, w, a_lower_bounds, a_upper_bounds):
        """
        Compute R matrix using ground truth MOCU (sampling-based).
        
        Same as ODE, but this is called at EVERY step.
        """
        R_matrix = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            for j in range(i + 1, self.N):
                # Compute f_inv (critical coupling for this pair)
                w_i = w[i]
                w_j = w[j]
                f_inv = 0.5 * np.abs(w_i - w_j)
                
                # Scenario 1: Synchronized observation
                a_upper_syn = a_upper_bounds.copy()
                a_lower_syn = a_lower_bounds.copy()
                
                a_tilde = min(max(f_inv, a_lower_bounds[i, j]), a_upper_bounds[i, j])
                a_lower_syn[j, i] = a_tilde
                a_lower_syn[i, j] = a_tilde
                
                P_syn = (a_upper_bounds[i, j] - a_tilde) / (
                    a_upper_bounds[i, j] - a_lower_bounds[i, j] + 1e-10
                )
                
                # Compute MOCU for synchronized scenario (using PyCUDA)
                mocu_vals_syn = np.zeros(self.it_idx)
                for l in range(self.it_idx):
                    mocu_vals_syn[l] = MOCU_pycuda_impl(
                        self.K_max, w, self.N, self.deltaT,
                        self.MVirtual, self.TVirtual,
                        a_lower_syn, a_upper_syn, seed=0
                    )
                MOCU_syn = np.mean(mocu_vals_syn)
                
                # Scenario 2: Non-synchronized observation
                a_upper_nonsyn = a_upper_bounds.copy()
                a_lower_nonsyn = a_lower_bounds.copy()
                
                a_upper_nonsyn[i, j] = a_tilde
                a_upper_nonsyn[j, i] = a_tilde
                
                P_nonsyn = (a_tilde - a_lower_bounds[i, j]) / (
                    a_upper_bounds[i, j] - a_lower_bounds[i, j] + 1e-10
                )
                
                # Compute MOCU for non-synchronized scenario (using PyCUDA)
                mocu_vals_nonsyn = np.zeros(self.it_idx)
                for l in range(self.it_idx):
                    mocu_vals_nonsyn[l] = MOCU_pycuda_impl(
                        self.K_max, w, self.N, self.deltaT,
                        self.MVirtual, self.TVirtual,
                        a_lower_nonsyn, a_upper_nonsyn, seed=0
                    )
                MOCU_nonsyn = np.mean(mocu_vals_nonsyn)
                
                # Expected MOCU
                R_matrix[i, j] = P_syn * MOCU_syn + P_nonsyn * MOCU_nonsyn
        
        return R_matrix
    
    def select_experiment(self, w, a_lower_bounds, a_upper_bounds, criticalK, isSynchronized, history):
        """
        Select next experiment using iterative iODE strategy.
        
        Re-computes R matrix at every step based on current bounds.
        """
        # Re-compute R matrix at every step (iterative)
        print(f"[iODE] Computing expected MOCU matrix (step {len(history) + 1})...")
        print("[iODE] Warning: This may take a LONG time (exact sampling)...")
        R_matrix = self._compute_expected_mocu_matrix(w, a_lower_bounds, a_upper_bounds)
        
        # Mask out already selected experiments
        for (i, j), _ in history:
            R_matrix[i, j] = 0.0
            R_matrix[j, i] = 0.0
        
        # Find experiment with minimum expected MOCU
        valid_R_values = R_matrix[np.nonzero(R_matrix)]
        
        if valid_R_values.size == 0:
            print("[iODE] Warning: No valid experiments left!")
            return -1, -1
        
        min_val = np.min(valid_R_values)
        min_indices = np.where(R_matrix == min_val)
        
        if len(min_indices[0]) > 1:
            min_i = int(min_indices[0][0])
            min_j = int(min_indices[1][0])
        else:
            min_i = int(min_indices[0])
            min_j = int(min_indices[1])
        
        return min_i, min_j

