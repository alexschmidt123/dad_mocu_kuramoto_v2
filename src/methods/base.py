"""
Base class for OED methods.

Provides common interface for all experimental design algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np
import time
import os
from typing import Tuple, List, Dict, Any

# Import torch for device checks
# Lazy import torch to avoid initializing PyTorch CUDA unnecessarily
# Import will happen lazily when needed (for PyTorch-based methods)
torch = None


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
                    isSynchronized_init, update_cnt, initial_mocu=None):
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
        
        N = len(w_init)
        # Initialize MOCUCurve - will be filled by computation
        MOCUCurve = np.zeros(update_cnt + 1)
        experimentSequence = []
        timeComplexity = np.zeros(update_cnt)
        history = []
        
        # Compute initial MOCU
        # initial_mocu is passed from evaluate.py (computed with torchdiffeq)
        # Use it to avoid redundant computation
        it_temp_val = np.zeros(self.it_idx)
        
        # Check method type
        method_name = self.__class__.__name__
        is_mpnn_method = method_name in ['iNN_Method', 'NN_Method']
        is_dad_method = method_name == 'DAD_MOCU_Method'
        
        if is_mpnn_method:
            # For MPNN methods, use initial MOCU from evaluate.py
            # They compute their own MOCU values iteratively via predictor
            if initial_mocu is not None:
                it_temp_val.fill(initial_mocu)
            else:
                it_temp_val.fill(0.0)
        elif is_dad_method:
            # For DAD method, use initial MOCU from evaluate.py
            # It uses policy network, doesn't compute MOCU directly
            if initial_mocu is not None:
                it_temp_val.fill(initial_mocu)
            else:
                it_temp_val.fill(0.0)
        else:
            # For ODE/RANDOM/ENTROPY methods, use initial MOCU from evaluate.py
            # Avoid redundant recomputation - evaluate.py already computed initial MOCU
            if initial_mocu is not None:
                # Use the pre-computed initial MOCU from evaluate.py (avoids redundant computation)
                it_temp_val.fill(initial_mocu)
                self._last_valid_mocu = initial_mocu  # Store for iterative fallback
            else:
                # Fallback: Only recompute if initial_mocu was not provided (shouldn't happen in normal flow)
                # Use PyCUDA for steps 1-3, torchdiffeq for DAD (steps 4-5)
                use_pycuda = os.getenv('USE_PYCUDA_FOR_BASELINES', '0') == '1'
                
                if use_pycuda:
                    # Steps 1-3: Use PyCUDA (original paper workflow)
                    try:
                        from ..core.mocu_pycuda import MOCU_pycuda
                        for l in range(self.it_idx):
                            it_temp_val[l] = MOCU_pycuda(
                                self.K_max, w_init, N, self.deltaT,
                                self.MReal, self.TReal,
                                a_lower_init, a_upper_init, 0
                            )
                        self._last_valid_mocu = np.mean(it_temp_val)
                    except (ImportError, RuntimeError) as e:
                        # PyCUDA failed - use zero as last resort
                        if not hasattr(self, '_pycuda_warned'):
                            print(f"[WARNING] PyCUDA unavailable and no initial_mocu provided: {e}")
                            print(f"[WARNING] Using zero for initial MOCU (this may affect results)")
                            self._pycuda_warned = True
                        it_temp_val.fill(0.0)
                        self._last_valid_mocu = 0.0
                else:
                    # Steps 4-5: Use torchdiffeq (DAD-specific)
                    try:
                        # Lazy import torch if needed
                        global torch
                        if torch is None:
                            try:
                                import torch as _torch
                                torch = _torch
                            except ImportError:
                                torch = None
                        
                        # Determine device
                        device = 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'
                        
                        from ..core.mocu_torchdiffeq import MOCU_torchdiffeq
                        for l in range(self.it_idx):
                            it_temp_val[l] = MOCU_torchdiffeq(
                                self.K_max, w_init, N, self.deltaT,
                                self.MReal, self.TReal,
                                a_lower_init, a_upper_init, 0, device=device
                            )
                        self._last_valid_mocu = np.mean(it_temp_val)
                    except (ImportError, RuntimeError) as e:
                        # torchdiffeq failed - use zero as last resort
                        if not hasattr(self, '_torchdiffeq_warned'):
                            print(f"[WARNING] torchdiffeq unavailable and no initial_mocu provided: {e}")
                            print(f"[WARNING] Using zero for initial MOCU (this may affect results)")
                            self._torchdiffeq_warned = True
                        it_temp_val.fill(0.0)
                        self._last_valid_mocu = 0.0


        MOCUCurve[0] = np.mean(it_temp_val)
        
        # Sequential experimental design
        # Get method name early for debug prints
        method_name = self.__class__.__name__
        is_mpnn_method = method_name in ['iNN_Method', 'NN_Method']
        is_dad_method = method_name == 'DAD_MOCU_Method'
        
        # Import torch for PyTorch-based methods (iNN, NN, DAD)
        # All methods now use torchdiffeq, so torch is safe to import
        if is_mpnn_method or is_dad_method:
            # Ensure all CUDA operations are complete before MPNN usage
            # Lazy import torch only when needed
            try:
                if torch is None:
                    import torch as _torch
                    globals()['torch'] = _torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
            except:
                pass
        
        a_lower_current = a_lower_init.copy()
        a_upper_current = a_upper_init.copy()
        
        for iteration in range(update_cnt):
            iterationStartTime = time.time()
            
            # Synchronize CUDA for PyTorch-based methods
            # All methods now use torchdiffeq, so torch is safe to import
            if is_mpnn_method or is_dad_method:
                # Ensure all CUDA operations are complete before MPNN usage
                try:
                    if torch is None:
                        import torch as _torch
                        globals()['torch'] = _torch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()  # Wait for any pending CUDA operations
                except:
                    pass
            
            # Select experiment using the method's specific logic
            # For iNN/NN: This uses MPNN predictor
            selected_i, selected_j = self.select_experiment(
                w_init, a_lower_current, a_upper_current, 
                criticalK_init, isSynchronized_init, history
            )
            
            # Synchronize CUDA for PyTorch-based methods
            if is_mpnn_method or is_dad_method:
                # Ensure MPNN operations are complete before next iteration
                # Lazy import torch only when needed
                try:
                    if torch is None:
                        import torch as _torch
                        globals()['torch'] = _torch
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
            
            # Debug: Print bound update for DAD method
            if is_dad_method and iteration < 2:
                print(f"[DAD] After bounds update at iteration {iteration+1}: selected=({selected_i},{selected_j}), sync={observation_sync}")
                print(f"[DAD] Updated bounds[{selected_i},{selected_j}]=({a_lower_current[selected_i,selected_j]:.4f},{a_upper_current[selected_i,selected_j]:.4f})")
                print(f"[DAD] bounds[0,1]={a_lower_current[0,1]:.4f},{a_upper_current[0,1]:.4f} (should change if (0,1) was selected)")
            
            # Re-compute MOCU for the updated bounds
            # CRITICAL: Match original paper code - ALL methods use actual MOCU computation (not predictor)
            # Original code uses PyCUDA's MOCU() for actual MOCU tracking after each experiment
            # We use torchdiffeq as replacement (same logic, different backend)
            
            if is_mpnn_method:
                # For MPNN methods (iNN/NN): Match original paper code's findMPSequence()
                # Original uses PyCUDA MOCU() for actual MOCU computation (not MPNN predictor)
                # MPNN predictor is ONLY used for R-matrix computation (selection), not for tracking
                try:
                    # Use PyCUDA for steps 1-3 (original paper), torchdiffeq for DAD (steps 4-5)
                    use_pycuda = os.getenv('USE_PYCUDA_FOR_BASELINES', '0') == '1'
                    
                    if use_pycuda:
                        # Steps 1-3: Use PyCUDA (original paper workflow)
                        from ..core.mocu_pycuda import MOCU_pycuda
                        it_temp_val = np.zeros(self.it_idx)
                        for l in range(self.it_idx):
                            it_temp_val[l] = MOCU_pycuda(
                                self.K_max, w_init, self.N, self.deltaT,
                                self.MReal, self.TReal,  # Use MReal/TReal for actual MOCU (matching original)
                                a_lower_current, a_upper_current, 0
                            )
                        MOCUCurve[iteration + 1] = np.mean(it_temp_val)
                    else:
                        # Steps 4-5: Use torchdiffeq (DAD-specific, runs in separate process)
                        from ..core.mocu_torchdiffeq import MOCU_torchdiffeq
                        device = 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'
                        it_temp_val = np.zeros(self.it_idx)
                        for l in range(self.it_idx):
                            it_temp_val[l] = MOCU_torchdiffeq(
                                self.K_max, w_init, self.N, self.deltaT,
                                self.MReal, self.TReal,  # Use MReal/TReal for actual MOCU (matching original)
                                a_lower_current, a_upper_current, 0, device=device
                            )
                        MOCUCurve[iteration + 1] = np.mean(it_temp_val)
                    
                    # Apply monotonicity constraint (matching original)
                    if MOCUCurve[iteration + 1] > MOCUCurve[iteration]:
                        MOCUCurve[iteration + 1] = MOCUCurve[iteration]
                except Exception as e:
                    # If computation fails, keep previous value
                    if not hasattr(self, '_mpnn_mocu_warned'):
                        print(f"[{method_name}] Warning: Failed to compute MOCU after bounds update: {e}")
                        self._mpnn_mocu_warned = True
                    MOCUCurve[iteration + 1] = MOCUCurve[iteration]
            elif is_dad_method:
                # For DAD method: Use torchdiffeq for MOCU tracking (matching baselines pattern)
                # DAD uses policy network for selection, torchdiffeq for MOCU tracking (real MOCU)
                # This matches the pattern: iNN/NN use PyCUDA for tracking, DAD uses torchdiffeq
                try:
                    # Use torchdiffeq for real MOCU computation (not MPNN predictor)
                    # DAD should use real MOCU for fair comparison with baselines
                    use_pycuda = os.getenv('USE_PYCUDA_FOR_BASELINES', '0') == '1'
                    
                    if not use_pycuda:  # DAD uses torchdiffeq (not PyCUDA)
                        from ..core.mocu_torchdiffeq import MOCU_torchdiffeq
                        device = 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'
                        it_temp_val = np.zeros(self.it_idx)
                        for l in range(self.it_idx):
                            it_temp_val[l] = MOCU_torchdiffeq(
                                self.K_max, w_init, self.N, self.deltaT,
                                self.MReal, self.TReal,  # Use MReal/TReal for actual MOCU (matching original)
                                a_lower_current, a_upper_current, 0, device=device
                            )
                        MOCUCurve[iteration + 1] = np.mean(it_temp_val)
                        # Apply monotonicity constraint (matching original)
                        if MOCUCurve[iteration + 1] > MOCUCurve[iteration]:
                            MOCUCurve[iteration + 1] = MOCUCurve[iteration]
                    else:
                        # Fallback: Use PyCUDA if available (for consistency with baselines)
                        try:
                            from ..core.mocu_pycuda import MOCU_pycuda
                            it_temp_val = np.zeros(self.it_idx)
                            for l in range(self.it_idx):
                                it_temp_val[l] = MOCU_pycuda(
                                    self.K_max, w_init, self.N, self.deltaT,
                                    self.MReal, self.TReal,
                                    a_lower_current, a_upper_current, 0
                                )
                            MOCUCurve[iteration + 1] = np.mean(it_temp_val)
                            # Apply monotonicity constraint
                            if MOCUCurve[iteration + 1] > MOCUCurve[iteration]:
                                MOCUCurve[iteration + 1] = MOCUCurve[iteration]
                        except Exception as e:
                            # PyCUDA failed, fall back to keeping previous value
                            if not hasattr(self, '_dad_pycuda_warned'):
                                print(f"[DAD] Warning: PyCUDA failed, using previous MOCU: {e}")
                                self._dad_pycuda_warned = True
                            MOCUCurve[iteration + 1] = MOCUCurve[iteration]
                except Exception as e:
                    if not hasattr(self, '_dad_exception_warned'):
                        print(f"[DAD] Exception during MOCU computation: {e}")
                        self._dad_exception_warned = True
                    MOCUCurve[iteration + 1] = MOCUCurve[iteration]
            else:
                # For ODE/RANDOM/ENTROPY methods: Use PyCUDA for steps 1-3 (original paper), torchdiffeq for DAD
                it_temp_val = np.zeros(self.it_idx)
                
                # Use PyCUDA for steps 1-3 (original paper workflow)
                use_pycuda = os.getenv('USE_PYCUDA_FOR_BASELINES', '0') == '1'
                
                if use_pycuda:
                    # Steps 1-3: Use PyCUDA (original paper workflow)
                    try:
                        from ..core.mocu_pycuda import MOCU_pycuda
                        for l in range(self.it_idx):
                            it_temp_val[l] = MOCU_pycuda(
                                self.K_max, w_init, self.N, self.deltaT,
                                self.MReal, self.TReal,
                                a_lower_current, a_upper_current, 
                                seed=0
                            )
                        MOCUCurve[iteration + 1] = np.mean(it_temp_val)
                        self._last_valid_mocu = MOCUCurve[iteration + 1]
                    except Exception as e:
                        # PyCUDA failed
                        if not hasattr(self, '_pycuda_iter_warned'):
                            print(f"[{method_name}] ERROR: PyCUDA failed for iterative MOCU (iteration {iteration+1}): {type(e).__name__}: {e}")
                            print(f"[{method_name}] Using fallback MOCU computation")
                            self._pycuda_iter_warned = True
                        if hasattr(self, '_last_valid_mocu') and self._last_valid_mocu is not None:
                            MOCUCurve[iteration + 1] = self._last_valid_mocu
                        else:
                            MOCUCurve[iteration + 1] = MOCUCurve[iteration]
                else:
                    # Steps 4-5: Use torchdiffeq (DAD-specific, runs in separate process)
                    # Lazy import torch if needed
                    if torch is None:
                        try:
                            import torch as _torch
                            globals()['torch'] = _torch
                        except ImportError:
                            pass
                    
                    # Determine device
                    device = 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'
                    
                    try:
                        from ..core.mocu_torchdiffeq import MOCU_torchdiffeq
                        
                        # Compute MOCU using torchdiffeq
                        for l in range(self.it_idx):
                            it_temp_val[l] = MOCU_torchdiffeq(
                                self.K_max, w_init, self.N, self.deltaT,
                                self.MReal, self.TReal,
                                a_lower_current, a_upper_current, 
                                seed=0, device=device
                            )
                        MOCUCurve[iteration + 1] = np.mean(it_temp_val)
                        
                        # Update last valid MOCU for future fallbacks
                        self._last_valid_mocu = MOCUCurve[iteration + 1]
                            
                    except ImportError:
                        # torchdiffeq not available
                        if not hasattr(self, '_torchdiffeq_import_warned'):
                            print(f"[{method_name}] Warning: torchdiffeq not available (ImportError)")
                            print(f"[{method_name}] Install with: pip install torchdiffeq")
                            print(f"[{method_name}] Using fallback MOCU computation")
                            self._torchdiffeq_import_warned = True
                        # Use last valid MOCU or previous value
                        if hasattr(self, '_last_valid_mocu') and self._last_valid_mocu is not None:
                            MOCUCurve[iteration + 1] = self._last_valid_mocu
                        else:
                            MOCUCurve[iteration + 1] = MOCUCurve[iteration]
                            
                    except Exception as e:
                        # torchdiffeq computation failed
                        if not hasattr(self, '_torchdiffeq_iter_warned'):
                            print(f"[{method_name}] ERROR: torchdiffeq failed for iterative MOCU (iteration {iteration+1}): {type(e).__name__}: {e}")
                            print(f"[{method_name}] Using fallback MOCU computation")
                            self._torchdiffeq_iter_warned = True
                        # Use last valid MOCU or previous value
                        if hasattr(self, '_last_valid_mocu') and self._last_valid_mocu is not None:
                            MOCUCurve[iteration + 1] = self._last_valid_mocu
                        else:
                            MOCUCurve[iteration + 1] = MOCUCurve[iteration]
            
            # Ensure MOCU is non-increasing (monotonicity constraint)
            # Note: This is already applied in MPNN methods (line 359) and DAD (line 400)
            # Only apply for ODE/RANDOM/ENTROPY if not already applied
            if not (is_mpnn_method or is_dad_method):
                if MOCUCurve[iteration + 1] > MOCUCurve[iteration]:
                    MOCUCurve[iteration + 1] = MOCUCurve[iteration]
        
        return MOCUCurve, experimentSequence, timeComplexity
    
    def get_name(self) -> str:
        """Return method name."""
        return self.name

