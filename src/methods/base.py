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
# Lazy import torch to avoid initializing PyTorch CUDA before PyCUDA methods run
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
        # CRITICAL: initial_mocu is passed from evaluate.py (computed with PyCUDA before methods)
        # Use it as fallback if PyCUDA fails in method initialization
        it_temp_val = np.zeros(self.it_idx)
        
        # Check method type to determine if PyCUDA should be used
        method_name = self.__class__.__name__
        is_mpnn_method = method_name in ['iNN_Method', 'NN_Method']
        is_dad_method = method_name == 'DAD_MOCU_Method'
        
        # Track if we've warned about PyCUDA for this method (to avoid spam)
        self._pycuda_warned = getattr(self, '_pycuda_warned', False)
        
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
            # FIXED: Avoid redundant recomputation - evaluate.py already computed initial MOCU
            # This prevents PyCUDA/PyTorch conflicts and improves performance
            if initial_mocu is not None:
                # Use the pre-computed initial MOCU from evaluate.py (avoids redundant computation)
                it_temp_val.fill(initial_mocu)
                self._last_valid_mocu = initial_mocu  # Store for iterative fallback
            else:
                # Fallback: Only recompute if initial_mocu was not provided (shouldn't happen in normal flow)
                # This case handles edge cases where evaluate.py didn't compute initial MOCU
                try:
                    # Clear PyTorch CUDA if it exists before PyCUDA (to avoid conflicts)
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    from ..core.mocu_pycuda import MOCU_pycuda
                    for l in range(self.it_idx):
                        it_temp_val[l] = MOCU_pycuda(self.K_max, w_init, N, self.deltaT,
                                                     self.MReal, self.TReal,
                                                     a_lower_init, a_upper_init, 0)
                    self._last_valid_mocu = np.mean(it_temp_val)
                except (ImportError, RuntimeError) as e:
                    # PyCUDA failed - use zero as last resort
                    if not self._pycuda_warned:
                        print(f"[WARNING] PyCUDA unavailable and no initial_mocu provided: {e}")
                        print(f"[WARNING] Using zero for initial MOCU (this may affect results)")
                        self._pycuda_warned = True
                    it_temp_val.fill(0.0)
                    self._last_valid_mocu = 0.0


        MOCUCurve[0] = np.mean(it_temp_val)
        
        # Sequential experimental design
        # Get method name early for debug prints
        method_name = self.__class__.__name__
        is_mpnn_method = method_name in ['iNN_Method', 'NN_Method']
        is_dad_method = method_name == 'DAD_MOCU_Method'
        
        # CRITICAL: Only import torch for PyTorch-based methods (iNN, NN, DAD)
        # Do NOT import torch for PyCUDA methods (RANDOM, ODE, ENTROPY) - it will initialize PyTorch CUDA!
        if is_mpnn_method or is_dad_method:
            # Ensure all CUDA operations are complete before MPNN usage
            # Lazy import torch only when needed (for PyTorch methods)
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
            
            # CRITICAL: Only synchronize CUDA for PyTorch-based methods
            # Do NOT import torch for PyCUDA methods (RANDOM, ODE, ENTROPY)
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
            
            # CRITICAL: Only synchronize CUDA for PyTorch-based methods
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
            # CRITICAL: For MPNN methods (iNN/NN), they handle MOCU computation themselves
            # DAD method also doesn't need PyCUDA (uses policy network)
            
            if is_mpnn_method:
                # For MPNN methods (iNN/NN), compute MOCU using MPNN predictor for updated bounds
                # The method's select_experiment computes R-matrix for selection, not current MOCU
                # We need to compute actual MOCU using predictor after bounds are updated
                try:
                    # iNN/NN methods have model loaded, use it to predict MOCU
                    if hasattr(self, 'model') and hasattr(self, 'mean') and hasattr(self, 'std'):
                        # Use the same predictor function as DAD
                        from ..models.predictors.predictor_utils import predict_mocu
                        # Debug: Show bounds that will be passed to predictor
                        if iteration < 2 and not hasattr(self, '_inn_bounds_debugged'):
                            sample_i, sample_j = experimentSequence[-1] if experimentSequence else (0, 1)
                            print(f"[{method_name}] Before predictor call at iteration {iteration+1}:")
                            print(f"  Selected pair: ({selected_i},{selected_j})")
                            print(f"  bounds[{selected_i},{selected_j}]=({a_lower_current[selected_i,selected_j]:.4f},{a_upper_current[selected_i,selected_j]:.4f})")
                            print(f"  bounds[0,1]=({a_lower_current[0,1]:.4f},{a_upper_current[0,1]:.4f})")
                            print(f"  bounds[0,2]=({a_lower_current[0,2]:.4f},{a_upper_current[0,2]:.4f})")
                            if iteration == 1:
                                self._inn_bounds_debugged = True
                        mocu_pred = predict_mocu(
                            self.model, self.mean, self.std,
                            w_init, a_lower_current, a_upper_current,
                            device='cuda' if torch.cuda.is_available() else 'cpu'
                        )
                        MOCUCurve[iteration + 1] = mocu_pred
                        # Debug: Print first few predictions to verify predictor is working
                        if iteration < 2 and not hasattr(self, '_inn_mocu_printed'):
                            prev_val = MOCUCurve[iteration]
                            change = mocu_pred - prev_val
                            print(f"[{method_name}] MOCU prediction at iteration {iteration+1}: {mocu_pred:.6f} (prev: {prev_val:.6f}, change: {change:+.6f})")
                            if iteration == 1:
                                self._inn_mocu_printed = True
                    else:
                        # Fallback: keep previous value if model not available
                        MOCUCurve[iteration + 1] = MOCUCurve[iteration]
                except Exception as e:
                    # If prediction fails, keep previous value
                    if not hasattr(self, '_mpnn_mocu_warned'):
                        print(f"[{method_name}] Warning: Failed to compute MOCU after bounds update: {e}")
                        self._mpnn_mocu_warned = True
                    MOCUCurve[iteration + 1] = MOCUCurve[iteration]
            elif is_dad_method:
                # For DAD method, use MPNN predictor to compute MOCU (same as iNN)
                # Cannot use PyCUDA after PyTorch CUDA is initialized
                # Use MPNN predictor to estimate MOCU based on updated bounds
                try:
                    # Try to load MPNN predictor if not already loaded
                    if not hasattr(self, '_mocu_predictor_loaded'):
                        # Lazy load predictor only when needed
                        model_name = os.getenv('MOCU_MODEL_NAME', f'cons{N}')
                        try:
                            from ..models.predictors.predictor_utils import load_mpnn_predictor, predict_mocu
                            if not hasattr(self, '_mocu_predictor_warned'):
                                print(f"[DAD] Loading MPNN predictor for MOCU computation: {model_name}")
                            self._mocu_model, self._mocu_mean, self._mocu_std = load_mpnn_predictor(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')
                            self._mocu_predictor_loaded = True
                            self._predict_mocu_fn = predict_mocu
                            if not hasattr(self, '_mocu_predictor_warned'):
                                print(f"[DAD] MPNN predictor loaded successfully")
                                self._mocu_predictor_warned = True  # Use same flag to avoid duplicate prints
                        except Exception as e:
                            # Predictor loading failed - use fallback
                            if not hasattr(self, '_mocu_predictor_warned'):
                                print(f"[DAD] Warning: MPNN predictor not available for MOCU computation: {e}")
                                print(f"[DAD] Using previous MOCU value")
                                self._mocu_predictor_warned = True
                            self._mocu_predictor_loaded = False
                    
                    # Compute MOCU using MPNN predictor
                    if hasattr(self, '_mocu_predictor_loaded') and self._mocu_predictor_loaded:
                        try:
                            # Debug: Enable detailed prediction logging for first 2 iterations
                            if iteration < 2:
                                self._mocu_model._debug_prediction = True
                            else:
                                self._mocu_model._debug_prediction = False
                            
                            # Debug: Show bounds that will be passed to predictor
                            if iteration < 2 and not hasattr(self, '_dad_bounds_debugged'):
                                print(f"[DAD] Before predictor call at iteration {iteration+1}:")
                                print(f"  Selected pair: ({selected_i},{selected_j})")
                                print(f"  bounds[{selected_i},{selected_j}]=({a_lower_current[selected_i,selected_j]:.4f},{a_upper_current[selected_i,selected_j]:.4f})")
                                print(f"  bounds[0,1]=({a_lower_current[0,1]:.4f},{a_upper_current[0,1]:.4f})")
                                print(f"  bounds[0,2]=({a_lower_current[0,2]:.4f},{a_upper_current[0,2]:.4f})")
                                # Check if bounds matrix actually changed
                                bounds_changed = not np.array_equal(a_lower_current, a_lower_init) or not np.array_equal(a_upper_current, a_upper_init)
                                print(f"  Bounds changed from initial: {bounds_changed}")
                                if iteration == 1:
                                    self._dad_bounds_debugged = True
                            
                            mocu_pred = self._predict_mocu_fn(
                                self._mocu_model, self._mocu_mean, self._mocu_std,
                                w_init, a_lower_current, a_upper_current,
                                device='cuda' if torch.cuda.is_available() else 'cpu'
                            )
                            prev_mocu = MOCUCurve[iteration]
                            MOCUCurve[iteration + 1] = mocu_pred
                            # Debug: Print first few predictions with bounds info
                            if iteration < 2 and not hasattr(self, '_dad_pred_printed'):
                                # Show a sample bound change to verify bounds are updating
                                sample_i, sample_j = experimentSequence[-1] if experimentSequence else (0, 1)
                                prev_lower = MOCUCurve[iteration - 1] if iteration > 0 else prev_mocu
                                bound_change = f"bounds[{sample_i},{sample_j}]=({a_lower_current[sample_i,sample_j]:.4f},{a_upper_current[sample_i,sample_j]:.4f})"
                                print(f"[DAD] MOCU prediction at iteration {iteration+1}: {mocu_pred:.6f} (prev: {prev_mocu:.6f}, change: {mocu_pred-prev_mocu:+.6f}, {bound_change})")
                                if iteration == 1:
                                    self._dad_pred_printed = True
                        except Exception as pred_err:
                            if not hasattr(self, '_dad_pred_error_warned'):
                                print(f"[DAD] Error during MOCU prediction: {pred_err}")
                                self._dad_pred_error_warned = True
                            MOCUCurve[iteration + 1] = MOCUCurve[iteration]
                    else:
                        # Fallback: keep previous value
                        MOCUCurve[iteration + 1] = MOCUCurve[iteration]
                except Exception as e:
                    # Any error - keep previous value
                    if not hasattr(self, '_dad_exception_warned'):
                        print(f"[DAD] Exception during MOCU computation: {e}")
                        self._dad_exception_warned = True
                    MOCUCurve[iteration + 1] = MOCUCurve[iteration]
            else:
                # For ODE/RANDOM/ENTROPY methods, use PyCUDA (as in original paper 2023)
                # CRITICAL: These methods MUST run BEFORE PyTorch CUDA is initialized
                # Use same troubleshooting approach as DAD training
                it_temp_val = np.zeros(self.it_idx)
                
                # CRITICAL: Do NOT import torch here - it will initialize PyTorch CUDA
                # This is a PyCUDA method (RANDOM/ODE/ENTROPY) - must avoid PyTorch CUDA
                # Clear any potential PyTorch CUDA context (lazy check)
                try:
                    # Only check/clear if torch was already imported (shouldn't happen for PyCUDA methods)
                    if torch is not None:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            # Small delay to let CUDA operations complete
                            time.sleep(0.01)
                except:
                    pass
                
                # CRITICAL: Check if PyCUDA context can be used
                # Do NOT check torch.cuda.is_initialized() here - it might trigger PyTorch CUDA initialization!
                pycuda_available = True
                conflict_detected = False
                
                try:
                    import pycuda.driver as drv
                    # Check if PyCUDA context exists and can be used
                    try:
                        pycuda_ctx = drv.Context.get_current()
                        if pycuda_ctx is not None:
                            # PyCUDA context exists - should work
                            pass
                    except:
                        # No PyCUDA context - might work anyway (lazy initialization)
                        pass
                    
                    # Only check PyTorch CUDA if torch was already imported (avoid importing it here)
                    if torch is not None:
                        try:
                            if torch.cuda.is_initialized():
                                # PyTorch CUDA is active - PyCUDA will fail due to context conflict
                                conflict_detected = True
                                pycuda_available = False
                                if not hasattr(self, '_pycuda_conflict_warned'):
                                    print(f"[{method_name}] Warning: PyTorch CUDA is initialized - PyCUDA will fail (context conflict)")
                                    print(f"[{method_name}] Using fallback MOCU computation")
                                    self._pycuda_conflict_warned = True
                        except:
                            # torch.cuda.is_initialized() might not be available
                            pass
                except ImportError:
                    pycuda_available = False
                    if not hasattr(self, '_pycuda_import_warned'):
                        print(f"[{method_name}] Warning: PyCUDA not available (ImportError)")
                        self._pycuda_import_warned = True
                
                if pycuda_available and not conflict_detected:
                    try:
                        from ..core.mocu_pycuda import MOCU_pycuda
                        # Try PyCUDA computation
                        for l in range(self.it_idx):
                            it_temp_val[l] = MOCU_pycuda(self.K_max, w_init, N, self.deltaT,
                                                         self.MReal, self.TReal,
                                                         a_lower_current, a_upper_current, 0)
                        MOCUCurve[iteration + 1] = np.mean(it_temp_val)
                        # Update last valid MOCU for future fallbacks
                        self._last_valid_mocu = MOCUCurve[iteration + 1]
                        # Debug: Print first successful computation
                        if iteration == 0 and not hasattr(self, '_pycuda_success_printed'):
                            print(f"[{method_name}] PyCUDA iterative MOCU computed: {MOCUCurve[iteration + 1]:.6f}")
                            self._pycuda_success_printed = True
                    except Exception as e:
                        # PyCUDA failed - catch ALL exceptions
                        # CRITICAL: Do NOT use slow PyTorch MOCU or CPU computation
                        # Instead, use cached value and warn user
                        if not hasattr(self, '_pycuda_iter_warned'):
                            print(f"[{method_name}] ERROR: PyCUDA failed for iterative MOCU (iteration {iteration+1}): {type(e).__name__}: {e}")
                            print(f"[{method_name}] WARNING: Using cached MOCU value - MOCU will not update correctly!")
                            print(f"[{method_name}] SOLUTION: Run PyCUDA methods (RANDOM/ODE/ENTROPY) BEFORE PyTorch methods to avoid conflicts")
                            self._pycuda_iter_warned = True
                        # Use last valid MOCU or previous value (no slow fallback)
                        if hasattr(self, '_last_valid_mocu') and self._last_valid_mocu is not None:
                            MOCUCurve[iteration + 1] = self._last_valid_mocu
                        else:
                            MOCUCurve[iteration + 1] = MOCUCurve[iteration]
                else:
                    # PyCUDA not available or conflict detected
                    # CRITICAL: Do NOT use slow PyTorch MOCU or CPU computation
                    # Use cached value and warn user about method ordering
                    if not hasattr(self, '_pycuda_fallback_warned'):
                        if conflict_detected:
                            print(f"[{method_name}] ERROR: PyTorch CUDA conflict detected - PyCUDA cannot work!")
                            print(f"[{method_name}] WARNING: Using cached MOCU value - MOCU will not update correctly!")
                            print(f"[{method_name}] SOLUTION: Run PyCUDA methods (RANDOM/ODE/ENTROPY) BEFORE PyTorch methods (iNN/NN/DAD)")
                        elif not pycuda_available:
                            print(f"[{method_name}] ERROR: PyCUDA not available!")
                            print(f"[{method_name}] WARNING: Using cached MOCU value - MOCU will not update correctly!")
                        self._pycuda_fallback_warned = True
                    # Use last valid MOCU or previous value (no slow fallback)
                    if hasattr(self, '_last_valid_mocu') and self._last_valid_mocu is not None:
                        MOCUCurve[iteration + 1] = self._last_valid_mocu
                    else:
                        MOCUCurve[iteration + 1] = MOCUCurve[iteration]
            
            # Ensure MOCU is non-increasing (monotonicity)
            # NOTE: For MPNN predictors, small prediction errors might cause slight increases
            # Clamp to previous value if prediction is higher
            if MOCUCurve[iteration + 1] > MOCUCurve[iteration]:
                # Debug: Warn if prediction is significantly higher (indicates predictor issue)
                if is_mpnn_method or is_dad_method:
                    increase = MOCUCurve[iteration + 1] - MOCUCurve[iteration]
                    if increase > 0.01:  # Significant increase (>1%)
                        if not hasattr(self, '_monotonicity_warned'):
                            print(f"[{method_name}] Warning: MOCU prediction {MOCUCurve[iteration + 1]:.6f} > previous {MOCUCurve[iteration]:.6f} (increase: {increase:.6f})")
                            print(f"[{method_name}] Clamped to previous value due to monotonicity constraint")
                            self._monotonicity_warned = True
                MOCUCurve[iteration + 1] = MOCUCurve[iteration]
        
        return MOCUCurve, experimentSequence, timeComplexity
    
    def get_name(self) -> str:
        """Return method name."""
        return self.name

