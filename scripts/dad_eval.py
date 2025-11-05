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
    
    # Natural frequencies
    w = np.array([-2.5000, -0.6667, 1.1667, 2.0000, 5.8333])
    
    # ========== Load initial bounds and MOCU from baseline results ==========
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
    
    # Load initial MOCU from baseline results
    # Try to find it from any baseline method's MOCU file (first iteration)
    baseline_methods = ['iNN', 'NN', 'ODE', 'ENTROPY', 'RANDOM']
    initial_mocu = None
    
    for method in baseline_methods:
        mocu_file = baseline_results / f'{method}_MOCU.txt'
        if mocu_file.exists():
            mocu_data = np.loadtxt(mocu_file)
            if mocu_data.ndim == 1:
                initial_mocu = mocu_data[0]  # First value is initial MOCU
            elif mocu_data.ndim == 2:
                initial_mocu = mocu_data[0, 0]  # First row, first column
            break
    
    if initial_mocu is None:
        print(f"Warning: Could not find initial MOCU from baseline results.")
        print(f"Computing initial MOCU using torchdiffeq...")
        # Fallback: compute initial MOCU
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        from src.core.mocu_torchdiffeq import MOCU_torchdiffeq as MOCU_initial
        mocu_backend = f"torchdiffeq (device: {device})"
        
        timeMOCU = time.time()
        it_temp_val = np.zeros(it_idx)
        with tqdm(total=it_idx, desc=f"  Initial MOCU ({mocu_backend})", leave=False, unit="iter", ncols=100) as pbar:
            for l in range(it_idx):
                it_temp_val[l] = MOCU_initial(K_max, w, N, deltaT, MReal, TReal, 
                                              aInitialLower.copy(), aInitialUpper.copy(), 0, device=device)
                pbar.update(1)
        initial_mocu = np.mean(it_temp_val)
        elapsed = time.time() - timeMOCU
        print(f"Initial MOCU: {initial_mocu:.6f} (computed in {elapsed:.2f}s / {elapsed/60:.1f}min)")
    else:
        print(f"Loaded initial MOCU from baseline results: {initial_mocu:.6f}")
    
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
        
        # Generate random coupling strengths (same as baseline evaluation)
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
        
        # ========== Run DAD method ==========
        method_start_time = time.time()
        
        MOCUCurve, experimentSequence, timeComplexity = method.run_episode(
            w_init=w,
            a_lower_init=aInitialLower.copy(),
            a_upper_init=aInitialUpper.copy(),
            criticalK_init=criticalK,
            isSynchronized_init=isSynchronized,
            update_cnt=update_cnt,
            initial_mocu=initial_mocu  # Use same initial MOCU as baselines
        )
        
        total_time = time.time() - method_start_time
        print(f"DAD: Time={total_time:.1f}s, Final MOCU={MOCUCurve[-1]:.6f}")
        
        # Save results
        outMOCUFile = open(os.path.join(result_folder, 'DAD_MOCU.txt'), 'a')
        outTimeFile = open(os.path.join(result_folder, 'DAD_timeComplexity.txt'), 'a')
        outSequenceFile = open(os.path.join(result_folder, 'DAD_sequence.txt'), 'a')
        
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
    print(f"  Initial MOCU (from baselines): {initial_mocu:.6f}")
    print(f"  Final mean MOCU: {mean_MOCU_matrix[-1, 0]:.6f}")

