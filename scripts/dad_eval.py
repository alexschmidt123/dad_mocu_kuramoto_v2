"""
DAD Method Evaluation Script

This script evaluates the DAD method using the same initial MOCU values
as the baseline methods. It loads the initial MOCU from baseline results
to ensure fair comparison.
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
    parser = argparse.ArgumentParser(description='Evaluate DAD method with baseline initial MOCU')
    parser.add_argument('--baseline_results', type=str, required=True,
                        help='Path to baseline results folder (contains initial MOCU info)')
    parser.add_argument('--result_folder', type=str, default=None,
                        help='Result folder for DAD results (default: baseline_results)')
    parser.add_argument('--method_name', type=str, default='DAD',
                        help='Method name for output files (default: DAD, options: DAD_MOCU, IDAD_MOCU)')
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
    
    # Use baseline results folder if result_folder not specified
    baseline_results = Path(args.baseline_results)
    if args.result_folder:
        result_folder = Path(args.result_folder)
    else:
        result_folder = baseline_results
    
    result_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"DAD Evaluation Configuration:")
    print(f"  Method: {args.method_name}")
    print(f"  N={N}, update_cnt={update_cnt}, it_idx={it_idx}, K_max={K_max}")
    print(f"  num_simulations={numberOfSimulationsPerMethod}")
    print(f"  result_folder={result_folder}")
    print(f"  baseline_results={baseline_results}")
    
    # Time parameters
    deltaT = 1.0 / 160.0
    TVirtual = 5
    MVirtual = int(TVirtual / deltaT)
    TReal = 5
    MReal = int(TReal / deltaT)
    
    # ========== Load system parameters from baseline results ==========
    # Load natural frequencies (should be same as baselines)
    w_file = baseline_results / 'paramNaturalFrequencies.txt'
    if w_file.exists():
        w = np.loadtxt(w_file)
        print(f"Loaded natural frequencies from baseline results: {w}")
    else:
        # Fallback to hardcoded values (should not happen if baselines were run first)
        print("⚠️  Warning: paramNaturalFrequencies.txt not found, using hardcoded values")
        w = np.array([-2.5000, -0.6667, 1.1667, 2.0000, 5.8333])
    
    # Load initial bounds (should be same for all methods)
    aInitialUpper_file = baseline_results / 'paramInitialUpper.txt'
    aInitialLower_file = baseline_results / 'paramInitialLower.txt'
    
    if not aInitialUpper_file.exists() or not aInitialLower_file.exists():
        raise FileNotFoundError(
            f"Baseline results not found: {aInitialUpper_file} or {aInitialLower_file}\n"
            f"Please run baseline evaluation first."
        )
    
    aInitialUpper = np.loadtxt(aInitialUpper_file)
    aInitialLower = np.loadtxt(aInitialLower_file)
    
    # ========== Choose MOCU backend for initial computation ==========
    # Use same MOCU computation method as baseline evaluation (prefer PyCUDA, fallback to torchdiffeq)
    # This ensures fair comparison - each simulation computes its own initial MOCU
    device = None  # Only set if using torchdiffeq
    try:
        from src.core.mocu_pycuda import MOCU_pycuda as MOCU_initial
        use_pycuda = True
        mocu_backend = "PyCUDA (matches baseline evaluation)"
    except (ImportError, RuntimeError) as e:
        print(f"⚠️  Warning: PyCUDA not available: {e}")
        print(f"   Falling back to torchdiffeq...")
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        from src.core.mocu_torchdiffeq import MOCU_torchdiffeq as MOCU_initial
        use_pycuda = False
        mocu_backend = f"torchdiffeq (device: {device})"
    
    print(f"Using MOCU backend: {mocu_backend}")
    print(f"Note: Initial MOCU will be computed per simulation (matches baseline evaluation)")
    
    # ========== Initialize DAD method ==========
    from src.methods.dad_mocu import DAD_MOCU_Method
    
    policy_path = None
    if 'DAD_POLICY_PATH' in os.environ:
        policy_path = Path(os.environ['DAD_POLICY_PATH'])
        if not policy_path.exists():
            policy_path = None
    
    if policy_path is None:
        raise RuntimeError(
            "DAD_POLICY_PATH environment variable not set or path does not exist.\n"
            "Please set DAD_POLICY_PATH to the trained DAD policy model."
        )
    
    print(f"Loading DAD policy from: {policy_path}")
    method = DAD_MOCU_Method(N, K_max, deltaT, MReal, TReal, it_idx, 
                            policy_model_path=policy_path)
    
    # ========== Results storage ==========
    save_MOCU_matrix = np.zeros([update_cnt + 1, 1, numberOfSimulationsPerMethod])
    
    # ========== Main simulation loop ==========
    sim_pbar = tqdm(total=numberOfSimulationsPerMethod, desc="Simulations", unit="sim", ncols=100)
    
    numberOfVaildSimulations = 0
    numberOfSimulations = 0
    
    while numberOfVaildSimulations < numberOfSimulationsPerMethod:
        sim_pbar.set_description(f"Simulation {numberOfVaildSimulations + 1}/{numberOfSimulationsPerMethod}")
        
        # CRITICAL: Load coupling strengths from baseline results to ensure SAME simulations
        # This ensures DAD/iDAD use EXACTLY the same initial conditions as baselines
        # → Same initial MOCU values for fair comparison
        coupling_file = baseline_results / f'paramCouplingStrength{numberOfVaildSimulations}.txt'
        
        if coupling_file.exists():
            # Load coupling strengths from baseline results (ensures same simulation)
            a = np.loadtxt(coupling_file)
            print(f'  ✓ Loaded coupling strengths from baseline: {coupling_file.name}')
        else:
            # Fallback: Generate new coupling strengths (should not happen if baselines run first)
            print(f'  ⚠️  Warning: Baseline coupling file not found: {coupling_file.name}')
            print(f'     Generating new coupling strengths (may cause different initial MOCU)')
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
        
        # Determine synchronization status and critical couplings (same as baseline)
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
        
        # Save coupling strengths
        coupling_file = os.path.join(result_folder, f'paramCouplingStrength{numberOfVaildSimulations}.txt')
        np.savetxt(coupling_file, a, fmt='%.64e')
        
        # ========== Load initial MOCU from baseline results ==========
        # CRITICAL: Use EXACT same initial MOCU as baselines for fair comparison
        # Initial MOCU is a property of the problem, not the method
        # Using same value ensures identical starting conditions
        initial_mocu_file = baseline_results / f'initial_MOCU_{numberOfVaildSimulations}.txt'
        
        if initial_mocu_file.exists():
            # Load initial MOCU from baseline results (ensures exact match)
            MOCUInitial = float(np.loadtxt(initial_mocu_file))
            sim_pbar.write(f'  ✓ Loaded initial MOCU from baseline: {MOCUInitial:.6f}')
        else:
            # Fallback: Compute initial MOCU (should not happen if baselines run first)
            sim_pbar.write(f'  ⚠️  Warning: Baseline initial MOCU not found, computing new...')
            timeMOCU = time.time()
            it_temp_val = np.zeros(it_idx)
            
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
        
        # ========== Run DAD method ==========
        method_start_time = time.time()
        
        MOCUCurve, experimentSequence, timeComplexity = method.run_episode(
            w_init=w,
            a_lower_init=aInitialLower.copy(),
            a_upper_init=aInitialUpper.copy(),
            criticalK_init=criticalK,
            isSynchronized_init=isSynchronized,
            update_cnt=update_cnt,
            initial_mocu=MOCUInitial  # Use per-simulation initial MOCU (matches baselines)
        )
        
        total_time = time.time() - method_start_time
        print(f"{args.method_name}: Time={total_time:.1f}s, Final MOCU={MOCUCurve[-1]:.6f}")
        
        # Save results with method name
        method_prefix = args.method_name
        outMOCUFile = open(os.path.join(result_folder, f'{method_prefix}_MOCU.txt'), 'a')
        outTimeFile = open(os.path.join(result_folder, f'{method_prefix}_timeComplexity.txt'), 'a')
        outSequenceFile = open(os.path.join(result_folder, f'{method_prefix}_sequence.txt'), 'a')
        
        np.savetxt(outMOCUFile, MOCUCurve.reshape(1, MOCUCurve.shape[0]), delimiter="\t")
        np.savetxt(outTimeFile, timeComplexity.reshape(1, timeComplexity.shape[0]), delimiter="\t")
        np.savetxt(outSequenceFile, experimentSequence, delimiter="\t")
        
        outMOCUFile.close()
        outTimeFile.close()
        outSequenceFile.close()
        
        save_MOCU_matrix[:, 0, numberOfVaildSimulations] = MOCUCurve
        numberOfVaildSimulations += 1
        sim_pbar.update(1)
    
    sim_pbar.close()
    
    # Compute mean MOCU across simulations
    mean_MOCU_matrix = np.mean(save_MOCU_matrix, axis=2)
    outMOCUFile = open(os.path.join(result_folder, 'mean_MOCU.txt'), 'a')
    np.savetxt(outMOCUFile, mean_MOCU_matrix, delimiter="\t")
    outMOCUFile.close()
    
    print(f"\n✓ DAD evaluation complete: {result_folder}")
    print(f"  Final mean MOCU: {mean_MOCU_matrix[-1, 0]:.6f}")
    print(f"  Note: Initial MOCU computed per simulation (matches baseline evaluation)")

