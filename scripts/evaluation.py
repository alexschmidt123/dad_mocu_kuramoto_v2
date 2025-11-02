"""
Unified Evaluation Script using New OED Methods Structure

This script evaluates all OED methods using the new unified interface.
All methods inherit from OEDMethod base class and share a common API.

Usage:
    python scripts/evaluation.py
    python scripts/evaluation.py --methods "iNN,NN,RANDOM"
"""

import sys
import time
import os
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.methods import (
    iNN_Method,
    NN_Method,
    ODE_Method,
    iODE_Method,
    ENTROPY_Method,
    RANDOM_Method,
    DAD_MOCU_Method,
)
from src.core.sync_detection import determineSyncN, determineSyncTwo
from src.core.mocu_cuda import MOCU  # This imports PyCUDA (via pycuda.autoinit in mocu_cuda.py)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate OED methods')
    parser.add_argument('--methods', type=str, default=None,
                        help='Comma-separated list of methods to evaluate (e.g., "iNN,NN,RANDOM")')
    args = parser.parse_args()
    # ========== Configuration ==========
    it_idx = 10  # Number of MOCU averaging iterations
    update_cnt = 10  # Number of sequential experiments
    N = 5  # Number of oscillators
    K_max = 20480  # Monte Carlo samples for MOCU
    
    # Get result folder from environment variable
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
    
    # ========== Methods to evaluate ==========
    # Can be set via command line or use defaults
    if args.methods:
        # Methods from command line (from config file via run.sh)
        method_names = [m.strip() for m in args.methods.split(',')]
        print(f"Using methods from command line: {method_names}")
    else:
        # Default methods if not specified
        method_names = [
            'iNN',      # Iterative MPNN (needs: train_mocu_predictor.py)
            'NN',       # Static MPNN (needs: train_mocu_predictor.py)
            'ODE',      # Static sampling-based (no training needed, but VERY slow)
            # 'iODE',   # Iterative sampling-based (no training, EXTREMELY slow)
            'ENTROPY',  # Greedy uncertainty (no training needed)
            'RANDOM',   # Random baseline (no training needed)
            # 'DAD',    # Deep Adaptive Design (needs: train_dad_policy.py) ⭐
        ]
        print(f"Using default methods: {method_names}")
    
    numberOfSimulationsPerMethod = 10
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
    
    # Apply specific reductions
    aInitialUpper[0, 2:5] = aInitialUpper[0, 2:5] * 0.3
    aInitialLower[0, 2:5] = aInitialLower[0, 2:5] * 0.3
    aInitialUpper[1, 3:5] = aInitialUpper[1, 3:5] * 0.45
    aInitialLower[1, 3:5] = aInitialLower[1, 3:5] * 0.45
    
    # Ensure symmetry
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
    while numberOfVaildSimulations < numberOfSimulationsPerMethod:
        print(f"\n{'='*80}")
        print(f"Starting simulation {numberOfSimulations + 1}")
        print(f"{'='*80}")
        
        # Generate random coupling strengths
        randomState = np.random.RandomState(int(numberOfSimulations))
        a = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                randomNumber = randomState.uniform()
                a[i, j] = aInitialLower[i, j] + randomNumber * (aInitialUpper[i, j] - aInitialLower[i, j])
                a[j, i] = a[i, j]
        
        numberOfSimulations += 1
        
        # Check if system is already synchronized (skip if so)
        init_sync_check = determineSyncN(w, deltaT, N, MReal, a)
        if init_sync_check == 1:
            print('             The system has been already stable. Skipping...')
            continue
        else:
            print('             Unstable system has been found ✓')
        
        # Determine synchronization status and critical couplings for each pair
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
        
        # Save coupling strengths for this simulation
        coupling_file = os.path.join(result_folder, f'paramCouplingStrength{numberOfVaildSimulations}.txt')
        np.savetxt(coupling_file, a, fmt='%.64e')
        
        # ========== Evaluate each method ==========
        for method_idx, method_name in enumerate(method_names):
            print(f"\n{'-'*80}")
            print(f"Method: {method_name} (Round {numberOfVaildSimulations + 1}/{numberOfSimulationsPerMethod})")
            print(f"{'-'*80}")
            
            # Compute initial MOCU
            timeMOCU = time.time()
            it_temp_val = np.zeros(it_idx)
            for l in range(it_idx):
                it_temp_val[l] = MOCU(K_max, w, N, deltaT, MReal, TReal, 
                                     aInitialLower.copy(), aInitialUpper.copy(), 0)
            MOCUInitial = np.mean(it_temp_val)
            
            print(f"Initial MOCU: {MOCUInitial:.6f} (computed in {time.time() - timeMOCU:.2f}s)")
            
            # Initialize method
            method_start_time = time.time()
            
            try:
                if method_name == 'iNN':
                    method = iNN_Method(N, K_max, deltaT, MReal, TReal, it_idx, 
                                       model_name=os.getenv('MOCU_MODEL_NAME', f'cons{N}'))
                
                elif method_name == 'NN':
                    method = NN_Method(N, K_max, deltaT, MReal, TReal, it_idx,
                                      model_name=os.getenv('MOCU_MODEL_NAME', f'cons{N}'))
                
                elif method_name == 'ODE':
                    method = ODE_Method(N, K_max, deltaT, MReal, TReal, it_idx,
                                       MVirtual=MVirtual, TVirtual=TVirtual)
                
                elif method_name == 'iODE':
                    method = iODE_Method(N, K_max, deltaT, MReal, TReal, it_idx,
                                        MVirtual=MVirtual, TVirtual=TVirtual)
                
                elif method_name == 'ENTROPY':
                    method = ENTROPY_Method(N, K_max, deltaT, MReal, TReal, it_idx)
                
                elif method_name == 'RANDOM':
                    method = RANDOM_Method(N, K_max, deltaT, MReal, TReal, it_idx,
                                          seed=numberOfVaildSimulations)
                
                elif method_name == 'DAD':
                    # Try to find DAD policy in models/{config_name}/{timestamp}/dad_policy_N{N}.pth
                    # Or use environment variable if set
                    policy_path = None
                    if 'DAD_POLICY_PATH' in os.environ:
                        policy_path = Path(os.environ['DAD_POLICY_PATH'])
                        if not policy_path.exists():
                            print(f"[DAD] Warning: DAD_POLICY_PATH set but file not found: {policy_path}")
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
                    update_cnt=update_cnt
                )
                
                total_time = time.time() - method_start_time
                print(f"✓ Completed in {total_time:.2f}s")
                print(f"  Final MOCU: {MOCUCurve[-1]:.6f}")
                print(f"  Experiment sequence: {experimentSequence}")
                
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

