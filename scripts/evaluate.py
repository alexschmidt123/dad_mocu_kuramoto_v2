"""
Evaluation Script for Baseline Methods (Steps 1-3 - Original Paper Workflow)

This script evaluates baseline OED methods (iNN, NN, ODE, ENTROPY, RANDOM) using PyCUDA
for MOCU computation, matching the original paper 2023 implementation exactly.

Steps 1-3 use PyCUDA (original paper workflow)
Steps 4-5 use torchdiffeq (DAD-specific, runs in separate process)

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --methods "ODE,iNN,NN"
"""

import sys
import time
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress verbose warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*DataLoader.*deprecated.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch_geometric')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import sync detection (CPU-based, always available)
from src.core.sync_detection import determineSyncN, determineSyncTwo


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate OED methods with smart ordering')
    parser.add_argument('--methods', type=str, default=None,
                        help='Comma-separated list of methods to evaluate (e.g., "ODE,iNN,NN")')
    parser.add_argument('--force-pytorch', action='store_true',
                        help='Force PyTorch CUDA for all methods (slower but simpler)')
    args = parser.parse_args()
    
    # ========== Configuration ==========
    def safe_getenv_int(key, default):
        """Get environment variable as int, handling empty strings."""
        val = os.getenv(key, default)
        return int(val) if val else int(default)
    
    it_idx = safe_getenv_int('EVAL_IT_IDX', '10')
    update_cnt = safe_getenv_int('EVAL_UPDATE_CNT', '10')
    N = safe_getenv_int('EVAL_N', '5')
    K_max = safe_getenv_int('EVAL_K_MAX', '20480')
    numberOfSimulationsPerMethod = safe_getenv_int('EVAL_NUM_SIMULATIONS', '10')
    
    result_folder = os.getenv('RESULT_FOLDER', str(PROJECT_ROOT / 'results' / 'default'))
    os.makedirs(result_folder, exist_ok=True)
    
    # Time parameters
    deltaT = 1.0 / 160.0
    TVirtual = 5
    MVirtual = int(TVirtual / deltaT)
    TReal = 5
    MReal = int(TReal / deltaT)
    
    # Natural frequencies
    w = np.array([-2.5000, -0.6667, 1.1667, 2.0000, 5.8333])
    
    # ========== Method Selection ==========
    # Steps 1-3: Use PyCUDA for baseline methods (original paper workflow)
    # Steps 4-5: Use torchdiffeq for DAD (runs in separate process via dad_eval.py)
    
    if args.methods:
        method_names = [m.strip() for m in args.methods.split(',')]
    else:
        method_names = ['iNN', 'NN', 'ODE', 'ENTROPY', 'RANDOM']
    
    # Print configuration (after method_names is defined)
    print(f"\n{'='*80}")
    print(f"Evaluation Configuration (matches original paper 2023)")
    print(f"{'='*80}")
    print(f"  N={N}, update_cnt={update_cnt}, it_idx={it_idx}, K_max={K_max}")
    print(f"  num_simulations={numberOfSimulationsPerMethod}")
    print(f"  methods={method_names}")
    print(f"  result_folder={result_folder}")
    print(f"{'='*80}\n")
    
    # Note: --force-pytorch flag is deprecated (baselines use PyCUDA, DAD uses torchdiffeq)
    if args.force_pytorch:
        print(f"\n🔧 Note: Baseline methods use PyCUDA (original paper)")
        print(f"   (--force-pytorch flag is deprecated but kept for compatibility)")
    
    # ========== Choose MOCU backend for initial computation ==========
    # Use PyCUDA for steps 1-3 (original paper workflow) - matches original paper exactly
    # PyCUDA runs in separate process via bash scripts, so no conflicts with PyTorch
    device = None  # Only set if using torchdiffeq
    try:
        from src.core.mocu_pycuda import MOCU_pycuda as MOCU_initial
        use_pycuda = True
    except (ImportError, RuntimeError) as e:
        print(f"⚠️  Warning: PyCUDA not available: {e}")
        print(f"   Falling back to torchdiffeq...")
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        from src.core.mocu_torchdiffeq import MOCU_torchdiffeq as MOCU_initial
        use_pycuda = False
    
    # Set environment variable so methods know to use PyCUDA
    if use_pycuda:
        os.environ['USE_PYCUDA_FOR_BASELINES'] = '1'
    else:
        # Ensure env var is not set if using torchdiffeq
        if 'USE_PYCUDA_FOR_BASELINES' in os.environ:
            del os.environ['USE_PYCUDA_FOR_BASELINES']
    
    numberOfVaildSimulations = 0
    numberOfSimulations = 0
    
    # ========== Initial bounds ==========
    aInitialUpper = np.zeros((N, N))
    aInitialLower = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i + 1, N):
            syncThreshold = np.abs(w[i] - w[j]) / 2.0
            aInitialUpper[i, j] = syncThreshold * 1.15
            aInitialLower[i, j] = syncThreshold * 0.85
            aInitialUpper[j, i] = aInitialUpper[i, j]
            aInitialLower[j, i] = aInitialLower[i, j]
    
    aInitialUpper[0, 2:5] = aInitialUpper[0, 2:5] * 0.3
    aInitialLower[0, 2:5] = aInitialLower[0, 2:5] * 0.3
    aInitialUpper[1, 3:5] = aInitialUpper[1, 3:5] * 0.45
    aInitialLower[1, 3:5] = aInitialLower[1, 3:5] * 0.45
    
    for i in range(N):
        for j in range(i + 1, N):
            aInitialUpper[j, i] = aInitialUpper[i, j]
            aInitialLower[j, i] = aInitialLower[i, j]
    
    # Save parameters
    np.savetxt(os.path.join(result_folder, 'paramNaturalFrequencies.txt'), w, fmt='%.64e')
    np.savetxt(os.path.join(result_folder, 'paramInitialUpper.txt'), aInitialUpper, fmt='%.64e')
    np.savetxt(os.path.join(result_folder, 'paramInitialLower.txt'), aInitialLower, fmt='%.64e')
    
    # ========== Results storage ==========
    save_MOCU_matrix = np.zeros([update_cnt + 1, len(method_names), numberOfSimulationsPerMethod])
    
    # ========== Main simulation loop ==========
    sim_pbar = tqdm(total=numberOfSimulationsPerMethod, desc="Simulations", unit="sim", ncols=100, mininterval=1.0)
    
    while numberOfVaildSimulations < numberOfSimulationsPerMethod:
        sim_pbar.set_description(f"Simulation {numberOfVaildSimulations + 1}/{numberOfSimulationsPerMethod}")
        
        # Generate random coupling strengths
        randomState = np.random.RandomState(int(numberOfSimulations))
        a = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                randomNumber = randomState.uniform()
                a[i, j] = aInitialLower[i, j] + randomNumber * (aInitialUpper[i, j] - aInitialLower[i, j])
                a[j, i] = a[i, j]
        
        numberOfSimulations += 1
        
        # Check if system is already synchronized
        # "Unstable" = not fully synchronized - this is the interesting case where OED can help
        #   The system has some oscillators that are not synchronized, so we can learn about
        #   coupling parameters through experiments. This is what we want to evaluate.
        # "Stable" = already synchronized - skip this simulation (no learning needed)
        #   All oscillators are synchronized from the start, so no experiments can help.
        init_sync_check = determineSyncN(w, deltaT, N, MReal, a)
        if init_sync_check == 1:
            sim_pbar.write(f'  ⚠️  System {numberOfSimulations}: Already synchronized (skipping - no learning needed)')
            continue
        else:
            sim_pbar.write(f'  ✓ System {numberOfSimulations}: Not synchronized (good for OED evaluation)')
        
        # Determine synchronization status and critical couplings
        isSynchronized = np.zeros((N, N))
        criticalK = np.zeros((N, N))
        
        for i in range(N):
            for j in range(i + 1, N):
                w_i = w[i]
                w_j = w[j]
                a_ij = a[i, j]
                syncThreshold = 0.5 * np.abs(w_i - w_j)
                criticalK[i, j] = syncThreshold
                criticalK[j, i] = syncThreshold
                isSynchronized[i, j] = determineSyncTwo(w_i, w_j, deltaT, 2, MReal, a_ij)
                isSynchronized[j, i] = isSynchronized[i, j]
        
        # Save coupling strengths
        coupling_file = os.path.join(result_folder, f'paramCouplingStrength{numberOfVaildSimulations}.txt')
        np.savetxt(coupling_file, a, fmt='%.64e')
        
        # ========== Compute initial MOCU ==========
        timeMOCU = time.time()
        it_temp_val = np.zeros(it_idx)
        
        # Initial MOCU computation (matches original paper)
        with tqdm(total=it_idx, desc="  Initial MOCU", leave=False, unit="iter", ncols=80, mininterval=0.5) as pbar:
            for l in range(it_idx):
                if use_pycuda:
                    it_temp_val[l] = MOCU_initial(K_max, w, N, deltaT, MReal, TReal,
                                                  aInitialLower.copy(), aInitialUpper.copy(), 0)
                else:
                    it_temp_val[l] = MOCU_initial(K_max, w, N, deltaT, MReal, TReal,
                                                  aInitialLower.copy(), aInitialUpper.copy(), 0, device=device)
                pbar.update(1)
        
        MOCUInitial = np.mean(it_temp_val)
        elapsed = time.time() - timeMOCU
        sim_pbar.write(f'  Initial MOCU: {MOCUInitial:.6f} ({elapsed:.1f}s)')
        
        # ========== Evaluate each method ==========
        method_pbar = tqdm(method_names, desc="  Methods", leave=False, unit="method", ncols=80, mininterval=1.0)
        
        # Monkey-patch print() to redirect to tqdm.write() during method execution
        # This prevents method prints from interfering with progress bars
        original_print = print
        def redirect_print(*args, **kwargs):
            """Redirect print to tqdm.write() to avoid interfering with progress bars."""
            # Filter out verbose method initialization messages
            msg = ' '.join(str(arg) for arg in args)
            if any(marker in msg for marker in ['[iNN]', '[NN]', '[ODE]', '[iODE]', '[ENTROPY]', '[RANDOM]']):
                # Only show important messages, suppress verbose ones
                if any(important in msg for important in ['Warning:', 'Error:', 'ERROR']):
                    method_pbar.write(f'  {msg}')
                # Suppress: "Loaded MPNN...", "Computing expected MOCU...", "Initialized..."
            else:
                # Non-method prints go through normally
                original_print(*args, **kwargs)
        
        for method_idx, method_name in enumerate(method_pbar):
            method_pbar.set_postfix({'method': method_name})
            
            method_start_time = time.time()
            
            try:
                # Temporarily redirect print for method initialization and execution
                import builtins
                builtins.print = redirect_print
                
                # Lazy import methods
                if method_name == 'iNN':
                    from src.methods.inn import iNN_Method
                    method = iNN_Method(N, K_max, deltaT, MReal, TReal, it_idx, 
                                       model_name=os.getenv('MOCU_MODEL_NAME', f'cons{N}'))
                
                elif method_name == 'NN':
                    from src.methods.nn import NN_Method
                    method = NN_Method(N, K_max, deltaT, MReal, TReal, it_idx,
                                      model_name=os.getenv('MOCU_MODEL_NAME', f'cons{N}'))
                
                elif method_name == 'ODE':
                    from src.methods.ode import ODE_Method
                    method = ODE_Method(N, K_max, deltaT, MReal, TReal, it_idx,
                                       MVirtual=MVirtual, TVirtual=TVirtual)
                
                elif method_name == 'iODE':
                    from src.methods.ode import iODE_Method
                    method = iODE_Method(N, K_max, deltaT, MReal, TReal, it_idx,
                                        MVirtual=MVirtual, TVirtual=TVirtual)
                
                elif method_name == 'ENTROPY':
                    from src.methods.entropy import ENTROPY_Method
                    method = ENTROPY_Method(N, K_max, deltaT, MReal, TReal, it_idx)
                
                elif method_name == 'RANDOM':
                    from src.methods.random import RANDOM_Method
                    method = RANDOM_Method(N, K_max, deltaT, MReal, TReal, it_idx,
                                          seed=numberOfVaildSimulations)
                
                elif method_name == 'DAD':
                    from src.methods.dad_mocu import DAD_MOCU_Method
                    policy_path = None
                    if 'DAD_POLICY_PATH' in os.environ:
                        policy_path = Path(os.environ['DAD_POLICY_PATH'])
                        if not policy_path.exists():
                            policy_path = None
                    method = DAD_MOCU_Method(N, K_max, deltaT, MReal, TReal, it_idx, 
                                            policy_model_path=policy_path)
                
                else:
                    print(f"Unknown method: {method_name}")
                    continue
                
                # Run the method (with print redirection active)
                MOCUCurve, experimentSequence, timeComplexity = method.run_episode(
                    w_init=w,
                    a_lower_init=aInitialLower.copy(),
                    a_upper_init=aInitialUpper.copy(),
                    criticalK_init=criticalK,
                    isSynchronized_init=isSynchronized,
                    update_cnt=update_cnt,
                    initial_mocu=MOCUInitial
                )
                
                # Restore original print
                import builtins
                builtins.print = original_print
                
                total_time = time.time() - method_start_time
                method_pbar.set_postfix({
                    'Time': f'{total_time:.1f}s',
                    'Final MOCU': f'{MOCUCurve[-1]:.6f}'
                })
                
                # Save results
                outMOCUFile = open(os.path.join(result_folder, f'{method_name}_MOCU.txt'), 'a')
                outTimeFile = open(os.path.join(result_folder, f'{method_name}_timeComplexity.txt'), 'a')
                outSequenceFile = open(os.path.join(result_folder, f'{method_name}_sequence.txt'), 'a')
                
                np.savetxt(outMOCUFile, MOCUCurve.reshape(1, MOCUCurve.shape[0]), delimiter="\t")
                np.savetxt(outTimeFile, timeComplexity.reshape(1, timeComplexity.shape[0]), delimiter="\t")
                np.savetxt(outSequenceFile, experimentSequence, delimiter="\t")
                
                outMOCUFile.close()
                outTimeFile.close()
                outSequenceFile.close()
                
                save_MOCU_matrix[:, method_idx, numberOfVaildSimulations] = MOCUCurve
            
            except Exception as e:
                # Restore original print before error handling
                import builtins
                builtins.print = original_print
                method_pbar.write(f'  ✗ Error running {method_name}: {e}')
                import traceback
                traceback.print_exc()
                continue
        
        numberOfVaildSimulations += 1
        sim_pbar.update(1)
        sim_pbar.set_postfix({'Completed': f'{numberOfVaildSimulations}/{numberOfSimulationsPerMethod}'})
    
    sim_pbar.close()
    
    # ========== Final summary ==========
    print(f"\n{'='*80}")
    print("All simulations completed!")
    print(f"{'='*80}")
    
    mean_MOCU_matrix = np.mean(save_MOCU_matrix, axis=2)
    print("\nMean MOCU values across all simulations:")
    print(mean_MOCU_matrix)
    
    outMOCUFile = open(os.path.join(result_folder, 'mean_MOCU.txt'), 'w')
    np.savetxt(outMOCUFile, mean_MOCU_matrix, delimiter="\t")
    outMOCUFile.close()
    
    print(f"\n✓ Results saved to: {result_folder}")