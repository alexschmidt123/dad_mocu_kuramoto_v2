"""
Base class for OED methods.

Provides common interface for all experimental design algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np
import time
from typing import Tuple, List, Dict, Any


class OEDMethod(ABC):
    """
    Abstract base class for Optimal Experimental Design methods.
    
    All OED methods must implement select_experiment() which chooses
    the next experiment given current state.
    """
    
    def __init__(self, N, K_max, deltaT, MReal, TReal, it_idx):
        """
        Args:
            N: Number of oscillators
            K_max: Number of Monte Carlo samples for MOCU
            deltaT: Time step
            MReal: Number of time steps for MOCU evaluation
            TReal: Time horizon for MOCU evaluation
            it_idx: Number of MOCU averaging iterations
        """
        self.N = N
        self.K_max = K_max
        self.deltaT = deltaT
        self.MReal = MReal
        self.TReal = TReal
        self.it_idx = it_idx
        self.name = self.__class__.__name__
    
    @abstractmethod
    def select_experiment(self, state: Dict[str, Any], available_pairs: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], Dict]:
        """
        Select next experiment to perform.
        
        Args:
            state: Current state dictionary containing:
                - 'w': Natural frequencies [N]
                - 'a_lower': Lower bounds [N, N]
                - 'a_upper': Upper bounds [N, N]
                - 'history': List of (i, j, observation) tuples
            available_pairs: List of (i, j) pairs not yet observed
        
        Returns:
            action: Selected (i, j) pair
            info: Dict with auxiliary information (computation time, values, etc.)
        """
        pass
    
    def run_sequential_design(self, 
                             initial_state: Dict[str, Any],
                             ground_truth: Dict[str, Any],
                             num_iterations: int,
                             mocu_computer) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        Run full sequential experimental design process.
        
        Args:
            initial_state: Initial state with uncertainty bounds
            ground_truth: Ground truth for simulating observations
                - 'a_true': True coupling strengths
                - 'is_synchronized': Synchronization matrix
                - 'critical_k': Critical coupling thresholds
            num_iterations: Number of experiments to perform
            mocu_computer: Function to compute ground truth MOCU
        
        Returns:
            mocu_curve: MOCU values at each iteration [num_iterations+1]
            sequence: List of selected (i, j) pairs
            times: Computation time per iteration [num_iterations]
        """
        N = len(initial_state['w'])
        
        # Initialize
        mocu_curve = np.zeros(num_iterations + 1)
        sequence = []
        times = np.zeros(num_iterations)
        
        # Compute initial MOCU
        mocu_curve[0] = mocu_computer(
            initial_state['w'],
            initial_state['a_lower'],
            initial_state['a_upper']
        )
        
        # Current state
        state = {
            'w': initial_state['w'].copy(),
            'a_lower': initial_state['a_lower'].copy(),
            'a_upper': initial_state['a_upper'].copy(),
            'history': []
        }
        
        observed_pairs = set()
        
        # Sequential experimental design
        for iteration in range(num_iterations):
            start_time = time.time()
            
            # Get available pairs
            available_pairs = [
                (i, j) for i in range(N) for j in range(i+1, N)
                if (i, j) not in observed_pairs
            ]
            
            # Select experiment
            (i_sel, j_sel), info = self.select_experiment(state, available_pairs)
            
            # Simulate observation
            observation = int(ground_truth['is_synchronized'][i_sel, j_sel])
            f_critical = ground_truth['critical_k'][i_sel, j_sel]
            
            # Update bounds
            if observation == 0:  # Not synchronized
                state['a_upper'][i_sel, j_sel] = min(state['a_upper'][i_sel, j_sel], f_critical)
                state['a_upper'][j_sel, i_sel] = state['a_upper'][i_sel, j_sel]
            else:  # Synchronized
                state['a_lower'][i_sel, j_sel] = max(state['a_lower'][i_sel, j_sel], f_critical)
                state['a_lower'][j_sel, i_sel] = state['a_lower'][i_sel, j_sel]
            
            # Update history
            state['history'].append((i_sel, j_sel, observation))
            observed_pairs.add((i_sel, j_sel))
            
            # Record
            sequence.append((i_sel, j_sel))
            times[iteration] = time.time() - start_time
            
            # Compute MOCU
            mocu_new = mocu_computer(state['w'], state['a_lower'], state['a_upper'])
            
            # Ensure monotonicity (MOCU should decrease or stay same)
            mocu_curve[iteration + 1] = min(mocu_new, mocu_curve[iteration])
        
        return mocu_curve, sequence, times
    
    def run_episode(self, w_init, a_lower_init, a_upper_init, criticalK_init, 
                    isSynchronized_init, update_cnt):
        """
        Run a complete OED episode.
        
        This is the main entry point for evaluation. It runs the sequential
        experimental design process and tracks:
        - MOCU curve over iterations
        - Selected experiment sequence  
        - Time complexity per iteration
        
        Args:
            w_init: Natural frequencies [N]
            a_lower_init: Initial lower bounds [N, N]
            a_upper_init: Initial upper bounds [N, N]
            criticalK_init: Ground truth critical couplings [N, N]
            isSynchronized_init: Ground truth synchronization status [N, N]
            update_cnt: Number of experiments to perform
        
        Returns:
            MOCUCurve: MOCU values at each step [update_cnt+1]
            experimentSequence: List of (i, j) tuples
            timeComplexity: Time per iteration [update_cnt]
        """
        from ..core.mocu_cuda import MOCU
        
        N = len(w_init)
        MOCUCurve = np.ones(update_cnt + 1) * 50.0
        experimentSequence = []
        timeComplexity = np.zeros(update_cnt)
        history = []
        
        # Compute initial MOCU using PyCUDA (original paper 2023 pattern)
        # This happens BEFORE any MPNN operations (separate usage pattern)
        it_temp_val = np.zeros(self.it_idx)
        for l in range(self.it_idx):
            it_temp_val[l] = MOCU(self.K_max, w_init, N, self.deltaT, 
                                 self.MReal, self.TReal, 
                                 a_lower_init, a_upper_init, 0)
        MOCUCurve[0] = np.mean(it_temp_val)
        
        # Ensure all PyCUDA operations are complete before MPNN usage
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for PyCUDA kernels to finish
        except:
            pass
        
        a_lower_current = a_lower_init.copy()
        a_upper_current = a_upper_init.copy()
        
        # Sequential experimental design
        # Follows original paper 2023 pattern: MPNN and PyCUDA used separately, not simultaneously
        for iteration in range(update_cnt):
            iterationStartTime = time.time()
            
            # Ensure all PyCUDA operations are complete before MPNN usage (separate usage pattern)
            # This prevents any concurrent GPU access between PyCUDA kernels and PyTorch/cuDNN
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for any pending CUDA operations
            except:
                pass
            
            # Select experiment using the method's specific logic
            # For iNN/NN: This uses MPNN predictor (separate from PyCUDA)
            selected_i, selected_j = self.select_experiment(
                w_init, a_lower_current, a_upper_current, 
                criticalK_init, isSynchronized_init, history
            )
            
            # Ensure MPNN operations (PyTorch/cuDNN) are complete before PyCUDA usage
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for MPNN forward passes to complete
            except:
                pass
            
            iterationTime = time.time() - iterationStartTime
            timeComplexity[iteration] = iterationTime
            
            experimentSequence.append((selected_i, selected_j))
            
            # Update bounds based on ground truth observation
            f_inv = criticalK_init[selected_i, selected_j]
            observation_sync = isSynchronized_init[selected_i, selected_j]
            
            if observation_sync == 0.0:  # Not synchronized
                a_upper_current[selected_i, selected_j] = min(
                    a_upper_current[selected_i, selected_j], f_inv
                )
                a_upper_current[selected_j, selected_i] = a_upper_current[selected_i, selected_j]
            else:  # Synchronized
                a_lower_current[selected_i, selected_j] = max(
                    a_lower_current[selected_i, selected_j], f_inv
                )
                a_lower_current[selected_j, selected_i] = a_lower_current[selected_i, selected_j]
            
            history.append(((selected_i, selected_j), observation_sync))
            
            # Re-compute MOCU for the updated bounds using PyCUDA
            # This happens AFTER MPNN operations are complete (separate usage pattern)
            it_temp_val = np.zeros(self.it_idx)
            for l in range(self.it_idx):
                it_temp_val[l] = MOCU(self.K_max, w_init, N, self.deltaT,
                                     self.MReal, self.TReal,
                                     a_lower_current, a_upper_current, 0)
            MOCUCurve[iteration + 1] = np.mean(it_temp_val)
            
            # Ensure MOCU is non-increasing (monotonicity)
            if MOCUCurve[iteration + 1] > MOCUCurve[iteration]:
                MOCUCurve[iteration + 1] = MOCUCurve[iteration]
        
        return MOCUCurve, experimentSequence, timeComplexity
    
    def get_name(self) -> str:
        """Return method name."""
        return self.name

