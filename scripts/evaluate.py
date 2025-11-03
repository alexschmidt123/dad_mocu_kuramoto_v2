"""
Enhanced Evaluation Script with Smart Method Ordering

This script automatically orders methods to run PyCUDA-based methods (ODE, RANDOM, ENTROPY)
before PyTorch-based methods (iNN, NN, DAD) to avoid context conflicts while maintaining
maximum performance.

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
    
    print(f"Evaluation Configuration:")
    print(f"  N={N}, update_cnt={update_cnt}, it_idx={it_idx}, K_max={K_max}")
    print(f"  num_simulations={numberOfSimulationsPerMethod}")
    print(f"  result_folder={result_folder}")
    
    # Time parameters
    deltaT = 1.0 / 160.0
    TVirtual = 5
    MVirtual = int(TVirtual / deltaT)
    TReal = 5
    MReal = int(TReal / deltaT)
    
    # Natural frequencies
    w = np.array([-2.5000, -0.6667, 1.1667, 2.0000, 5.8333])
    
    # ========== Smart Method Ordering ==========
    # Categorize methods by CUDA backend
    PYCUDA_METHODS = ['RANDOM', 'ENTROPY', 'ODE', 'iODE']
    PYTORCH_METHODS = ['iNN', 'NN', 'DAD']
    
    if args.methods:
        method_names = [m.strip() for m in args.methods.split(',')]
    else:
        method_names = ['iNN', 'NN', 'ODE', 'ENTROPY', 'RANDOM']
    
    # Sort methods: PyCUDA first, PyTorch second (unless --force-pytorch)
    if args.force_pytorch:
        print(f"\n🔧 Force PyTorch mode enabled - all methods will use PyTorch CUDA")
        print(f"   (ODE will be slower but no context conflicts)")
        use_pycuda_ordering = False
    else:
        # Check if we have both PyCUDA and PyTorch methods
        has_pycuda = any(m in PYCUDA_METHODS for m in method_names)
        has_pytorch = any(m in PYTORCH_METHODS for m in method_names)
        use_pycuda_ordering = has_pycuda and has_pytorch
        
        if use_pycuda_ordering:
            pycuda_list = [m for m in method_names if m in PYCUDA_METHODS]
            pytorch_list = [m for m in method_names if m in PYTORCH_METHODS]
            method_names = pycuda_list + pytorch_list
            
            print(f"\n🔧 Smart method ordering enabled:")
            print(f"   PyCUDA methods (first):  {pycuda_list}")
            print(f"   PyTorch methods (after): {pytorch_list}")
            print(f"   → This maximizes ODE performance while avoiding conflicts")
        elif has_pycuda:
            print(f"\n🔧 Only PyCUDA methods detected - using PyCUDA (fast)")
        elif has_pytorch:
            print(f"\n🔧 Only PyTorch methods detected - using PyTorch CUDA")
    
    print(f"\n📋 Method execution order: {method_names}")
    
    # ========== Choose MOCU backend for initial computation ==========
    # Use PyCUDA for initial MOCU (as in original paper 2023)
    # This ensures compatibility with PyCUDA-based methods (ODE, RANDOM, ENTROPY)
    # PyTorch methods (iNN, NN, DAD) don't need initial MOCU computation
    from src.core.mocu_pycuda import MOCU_pycuda as MOCU_initial
    mocu_backend = "PyCUDA (original paper 2023)"
    
    print(f"   Initial MOCU computation: {mocu_backend}")
    print()
    
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
    sim_pbar = tqdm(total=numberOfSimulationsPerMethod, desc="Simulations", unit="sim", ncols=100)
    
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
        init_sync_check = determineSyncN(w, deltaT, N, MReal, a)
        if init_sync_check == 1:
            print('             The system has been already stable. Skipping...')
            continue
        else:
            print('             Unstable system has been found ✓')
        
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
        
        with tqdm(total=it_idx, desc=f"  Initial MOCU ({mocu_backend})", leave=False, unit="iter", ncols=100) as pbar:
            for l in range(it_idx):
                it_temp_val[l] = MOCU_initial(K_max, w, N, deltaT, MReal, TReal, 
                                              aInitialLower.copy(), aInitialUpper.copy(), 0)
                pbar.update(1)
        
        MOCUInitial = np.mean(it_temp_val)
        print(f"Initial MOCU: {MOCUInitial:.6f} (computed in {time.time() - timeMOCU:.2f}s)")
        
        # ========== Evaluate each method ==========
        method_pbar = tqdm(method_names, desc=f"  Round {numberOfVaildSimulations + 1}/{numberOfSimulationsPerMethod}", 
                          leave=False, unit="method", ncols=100)
        
        for method_idx, method_name in enumerate(method_pbar):
            method_pbar.set_description(f"  Round {numberOfVaildSimulations + 1}/{numberOfSimulationsPerMethod} - {method_name}")
            
            method_start_time = time.time()
            
            try:
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
                
                # Run the method
                MOCUCurve, experimentSequence, timeComplexity = method.run_episode(
                    w_init=w,
                    a_lower_init=aInitialLower.copy(),
                    a_upper_init=aInitialUpper.copy(),
                    criticalK_init=criticalK,
                    isSynchronized_init=isSynchronized,
                    update_cnt=update_cnt,
                    initial_mocu=MOCUInitial
                )
                
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
                print(f"✗ Error running {method_name}: {e}")
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