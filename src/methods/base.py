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
try:
    import torch
except ImportError:
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
            # For ODE/RANDOM/ENTROPY methods, use PyCUDA (as in original paper 2023)
            # CRITICAL: These methods MUST run BEFORE PyTorch CUDA is initialized
            # The evaluation script ensures this by ordering methods correctly
            # Use same troubleshooting approach as DAD training
            
            # CRITICAL: Check for PyCUDA context conflict (same as DAD training)
            # Clear PyTorch CUDA if it exists before PyCUDA
            try:
                if torch is not None and torch.cuda.is_available():
                    # Clear any PyTorch CUDA operations before PyCUDA
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                # Check if PyCUDA context can be used
                try:
                    import pycuda.driver as drv
                    pycuda_ctx = drv.Context.get_current()
                    if pycuda_ctx is not None and torch is not None and torch.cuda.is_initialized():
                        # Both contexts active - this is OK for initial MOCU (evaluate.py handles this)
                        pass
                except:
                    pass
            except:
                pass
            
            try:
                from ..core.mocu_pycuda import MOCU_pycuda
                for l in range(self.it_idx):
                    it_temp_val[l] = MOCU_pycuda(self.K_max, w_init, N, self.deltaT, 
                                                 self.MReal, self.TReal, 
                                                 a_lower_init, a_upper_init, 0)
            except (ImportError, RuntimeError) as e:
                # PyCUDA failed - use fallback from evaluate.py
                if not self._pycuda_warned:
                    print(f"[WARNING] PyCUDA unavailable for {method_name} initial MOCU: {e}")
                    if initial_mocu is not None:
                        print(f"[WARNING] Using initial MOCU from evaluate.py: {initial_mocu:.6f}")
                    self._pycuda_warned = True
                # Use initial MOCU from evaluate.py (computed with PyCUDA before methods)
                if initial_mocu is not None:
                    it_temp_val.fill(initial_mocu)
                    self._last_valid_mocu = initial_mocu  # Store for iterative fallback
                else:
                    it_temp_val.fill(0.0)
        
        MOCUCurve[0] = np.mean(it_temp_val)
        
        # Ensure all CUDA operations are complete before MPNN usage
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for CUDA kernels to finish
        except:
            pass
        
        a_lower_current = a_lower_init.copy()
        a_upper_current = a_upper_init.copy()
        
        # Sequential experimental design
        for iteration in range(update_cnt):
            iterationStartTime = time.time()
            
            # Ensure all CUDA operations are complete before MPNN usage
            try:
                import torch
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
            
            # Ensure MPNN operations are complete before next iteration
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
            
            # Re-compute MOCU for the updated bounds
            # CRITICAL: For MPNN methods (iNN/NN), they handle MOCU computation themselves
            # DAD method also doesn't need PyCUDA (uses policy network)
            method_name = self.__class__.__name__
            is_mpnn_method = method_name in ['iNN_Method', 'NN_Method']
            is_dad_method = method_name == 'DAD_MOCU_Method'
            
            if is_mpnn_method:
                # For MPNN methods (iNN/NN), compute MOCU using MPNN predictor for updated bounds
                # The method's select_experiment computes R-matrix for selection, not current MOCU
                # We need to compute actual MOCU using predictor after bounds are updated
                try:
                    # iNN/NN methods have model loaded, use it to predict MOCU
                    if hasattr(self, 'model') and hasattr(self, 'mean') and hasattr(self, 'std'):
                        # Use the same predictor function as DAD
                        from ..models.predictors.predictor_utils import predict_mocu
                        mocu_pred = predict_mocu(
                            self.model, self.mean, self.std,
                            w_init, a_lower_current, a_upper_current,
                            device='cuda' if torch.cuda.is_available() else 'cpu'
                        )
                        MOCUCurve[iteration + 1] = mocu_pred
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
                            mocu_pred = self._predict_mocu_fn(
                                self._mocu_model, self._mocu_mean, self._mocu_std,
                                w_init, a_lower_current, a_upper_current,
                                device='cuda' if torch.cuda.is_available() else 'cpu'
                            )
                            MOCUCurve[iteration + 1] = mocu_pred
                            # Debug: Print first few predictions
                            if iteration < 2 and not hasattr(self, '_dad_pred_printed'):
                                print(f"[DAD] MOCU prediction at iteration {iteration+1}: {mocu_pred:.6f}")
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
                
                # CRITICAL: Clear PyTorch CUDA context before PyCUDA (same as DAD training)
                # This ensures PyCUDA can work even if PyTorch was imported
                try:
                    if torch is not None and torch.cuda.is_available():
                        # Clear any PyTorch CUDA operations before PyCUDA
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        # Small delay to let CUDA operations complete
                        time.sleep(0.01)
                except:
                    pass
                
                # CRITICAL: Check if PyCUDA context can be used (same as DAD training check)
                try:
                    import pycuda.driver as drv
                    # Try to get current context - if it fails, PyCUDA might still work
                    try:
                        pycuda_ctx = drv.Context.get_current()
                        # If PyTorch CUDA is also active, this might cause issues
                        if pycuda_ctx is not None and torch is not None and torch.cuda.is_initialized():
                            # Both contexts active - PyCUDA might fail
                            # Use fallback instead
                            raise RuntimeError("PyCUDA and PyTorch CUDA both active")
                    except:
                        # No PyCUDA context or error checking - try PyCUDA anyway
                        pass
                except ImportError:
                    pass
                
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
                except Exception as e:
                    # PyCUDA failed - catch ALL exceptions (not just ImportError/RuntimeError)
                    # PyCUDA can raise various exceptions: LogicError, Error, etc.
                    # Only warn once per method to avoid spam
                    if not hasattr(self, '_pycuda_iter_warned'):
                        print(f"[{method_name}] PyCUDA failed for iterative MOCU (iteration {iteration+1}): {type(e).__name__}: {e}")
                        if hasattr(self, '_last_valid_mocu'):
                            print(f"[{method_name}] Using last valid MOCU: {self._last_valid_mocu:.6f}")
                        else:
                            print(f"[{method_name}] Using previous MOCU value: {MOCUCurve[iteration]:.6f}")
                        self._pycuda_iter_warned = True
                    if hasattr(self, '_last_valid_mocu'):
                        MOCUCurve[iteration + 1] = self._last_valid_mocu
                    else:
                        MOCUCurve[iteration + 1] = MOCUCurve[iteration]
            
            # Ensure MOCU is non-increasing (monotonicity)
            if MOCUCurve[iteration + 1] > MOCUCurve[iteration]:
                MOCUCurve[iteration + 1] = MOCUCurve[iteration]
        
        return MOCUCurve, experimentSequence, timeComplexity
    
    def get_name(self) -> str:
        """Return method name."""
        return self.name

