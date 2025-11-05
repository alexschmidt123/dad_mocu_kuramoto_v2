"""
Combined data generation and dataset preparation script.
Generates training data with two different coupling distributions and converts to PyTorch Geometric format.
"""

import sys
from pathlib import Path
import time
import json
import argparse
import os
from tqdm import tqdm
import multiprocessing as mp

# Get absolute path to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# PyCUDA MOCU is used for all MOCU computation (mocu.py removed)
import numpy as np
import random
import torch  # Need torch for device detection at module level
import os

# PyCUDA is REQUIRED for data generation to avoid segfaults with PyTorch
# Data generation should run in isolation, separate from PyTorch workflows
try:
    from src.core.mocu_pycuda import MOCU_pycuda
    PYCUDA_MOCU_AVAILABLE = True
    print("[INFO] PyCUDA MOCU enabled (REQUIRED for data generation)")
except (ImportError, RuntimeError) as e:
    PYCUDA_MOCU_AVAILABLE = False
    print(f"[ERROR] PyCUDA is REQUIRED for data generation but not available: {e}")
    print(f"[ERROR] Please install PyCUDA: pip install pycuda")
    print(f"[ERROR] Data generation cannot proceed without PyCUDA.")
    sys.exit(1)

def generate_coupling_type1(w, N):
    """
    Generate coupling bounds with per-edge random multiplier.
    This creates more diverse coupling patterns across edges.
    """
    a_upper_bound = np.zeros((N, N))
    a_lower_bound = np.zeros((N, N))
    
    uncertainty = 0.3 * random.random()
    
    for i in range(N):
        for j in range(i + 1, N):
            # Per-edge random multiplier (TYPE 1)
            if random.random() < 0.5:
                mul = 0.6 * random.random()
            else:
                mul = 1.1 * random.random()
            
            f_inv = np.abs(w[i] - w[j]) / 2.0
            a_upper_bound[i, j] = f_inv * (1 + uncertainty) * mul
            a_lower_bound[i, j] = f_inv * (1 - uncertainty) * mul
            a_upper_bound[j, i] = a_upper_bound[i, j]
            a_lower_bound[j, i] = a_lower_bound[i, j]
    
    return a_lower_bound, a_upper_bound


def generate_coupling_type2(w, N):
    """
    Generate coupling bounds with per-oscillator random multiplier.
    This creates more correlated coupling patterns (all edges from one oscillator share multiplier).
    """
    a_upper_bound = np.zeros((N, N))
    a_lower_bound = np.zeros((N, N))
    
    uncertainty = 0.3 * random.random()
    
    for i in range(N):
        # Per-oscillator random multiplier (TYPE 2)
        if random.random() < 0.5:
            mul_ = 0.6
        else:
            mul_ = 1.1
        
        for j in range(i + 1, N):
            mul = mul_ * random.random()
            f_inv = np.abs(w[i] - w[j]) / 2.0
            a_upper_bound[i, j] = f_inv * (1 + uncertainty) * mul
            a_lower_bound[i, j] = f_inv * (1 - uncertainty) * mul
            a_upper_bound[j, i] = a_upper_bound[i, j]
            a_lower_bound[j, i] = a_lower_bound[i, j]
    
    return a_lower_bound, a_upper_bound


def generate_single_sample(args_tuple):
    """Generate a single training sample with MOCU computation.
    
    Wrapper function for multiprocessing that unpacks arguments.
    Always computes MOCU twice and averages for stability.
    """
    N, K_max, h, M, T, coupling_type, device, worker_id = args_tuple
    
    # Set worker-specific random seed for reproducibility
    if worker_id is not None:
        random.seed(worker_id * 12345 + int(time.time()) % 10000)
        np.random.seed(worker_id * 12345 + int(time.time()) % 10000)
    
    # Generate random natural frequencies
    w = np.zeros(N)
    for i in range(N):
        w[i] = 12 * (0.5 - random.random())
    
    # Generate coupling bounds based on type
    if coupling_type == 'type1':
        a_lower_bound, a_upper_bound = generate_coupling_type1(w, N)
    else:
        a_lower_bound, a_upper_bound = generate_coupling_type2(w, N)
    
    # Check if system is already synchronized using GPU version (faster)
    a = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            a[i, j] = a_lower_bound[i, j] + 0.5 * (a_upper_bound[i, j] - a_lower_bound[i, j])
            a[j, i] = a[i, j]
    
    # Use CPU sync check (fast enough, avoids CUDA context issues)
    from src.core.sync_detection import mocu_comp
    init_sync_check = mocu_comp(w, h, N, M, a)
    
    if init_sync_check == 1:
        return None  # Skip synchronized systems
    
    # Always compute MOCU twice and average for stability (reduces Monte Carlo variance)
    # NOTE: MOCU computation is sequential per sample (binary search dependencies)
    # GPU accelerates RK4 integration within each binary search iteration
    # For K_max=20480: ~20480 samples * ~25 binary searches * 640 RK4 steps = significant computation
    
    # PyCUDA is REQUIRED for data generation
    if not PYCUDA_MOCU_AVAILABLE:
        raise RuntimeError("PyCUDA is required for data generation but not available")
    
    # Debug: Check GPU usage
    debug_gpu = os.getenv('DEBUG_GPU', 'false').lower() == 'true'
    if debug_gpu and worker_id == 0:  # Only print from first worker
        print(f"[DEBUG] Starting MOCU computation with PyCUDA (required for data generation)")
        start_time = time.time()
    
    # Always use PyCUDA for data generation (required, no PyTorch option)
    MOCU_val1 = MOCU_pycuda(K_max, w, N, h, M, T, a_lower_bound, a_upper_bound, 0)
    MOCU_val2 = MOCU_pycuda(K_max, w, N, h, M, T, a_lower_bound, a_upper_bound, 0)
    
    if debug_gpu and worker_id == 0:
        elapsed = time.time() - start_time
        print(f"[DEBUG] Both MOCU computations completed in {elapsed:.2f}s")
    
    mean_MOCU = (MOCU_val1 + MOCU_val2) / 2
    
    data_dic = {
        'w': w.tolist(),
        'a_lower': a_lower_bound.tolist(),
        'a_upper': a_upper_bound.tolist(),
        'MOCU1': float(MOCU_val1),
        'MOCU2': float(MOCU_val2),
        'mean_MOCU': float(mean_MOCU)
    }
    
    return data_dic


def getEdgeAtt(attr1, attr2, n):
    """Convert matrix attributes to edge attributes for PyTorch Geometric."""
    edge_attr = torch.zeros([2, n * (n - 1)])
    k = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                edge_attr[0, k] = attr1[i, j]
                edge_attr[1, k] = attr2[i, j]
                k = k + 1
    return edge_attr


def convert_to_pytorch_geometric(data_list):
    """Convert JSON data to PyTorch Geometric Data objects."""
    from torch_geometric.data import Data
    
    pyg_data_list = []
    
    for data_item in data_list:
        # Node features: natural frequencies
        x = np.asarray(data_item['w'])
        x = torch.from_numpy(x.astype(np.float32))
        n = x.size()[0]
        x = x.unsqueeze(dim=1)
        
        # Edge indices (fully connected graph)
        edge_index = getEdgeAtt(
            np.tile(np.asarray([i for i in range(n)]), (n, 1)),
            np.tile(np.asarray([[i] for i in range(n)]), (1, n)),
            n
        ).long()
        
        # Edge features: [a_lower, a_upper]
        edge_attr = getEdgeAtt(
            torch.from_numpy(np.asarray(data_item['a_lower']).astype(np.float32)),
            torch.from_numpy(np.asarray(data_item['a_upper']).astype(np.float32)),
            n
        )
        
        # Target: mean MOCU
        y = torch.from_numpy(np.asarray(data_item['mean_MOCU']).astype(np.float32))
        y = y.unsqueeze(dim=0).unsqueeze(dim=0)
        
        pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.t(), y=y)
        pyg_data_list.append(pyg_data)
    
    return pyg_data_list


def main():
    # Import torch at function level to ensure it's available
    # (module-level import should work, but this avoids any scoping issues)
    import torch
    
    parser = argparse.ArgumentParser(description='Generate training dataset for MOCU prediction')
    parser.add_argument('--N', type=int, default=5, help='Number of oscillators')
    parser.add_argument('--samples_per_type', type=int, default=37500, 
                        help='Number of samples per coupling type (total = 2 * samples_per_type)')
    parser.add_argument('--K_max', type=int, default=20480, help='Monte Carlo samples for MOCU')
    parser.add_argument('--train_size', type=int, default=70000, 
                        help='Expected training set size (for reference only, actual split done by training script)')
    parser.add_argument('--output_dir', type=str, default='../data/', help='Output directory (with trailing slash)')
    parser.add_argument('--save_json', action='store_true', 
                        help='Save intermediate JSON files with MOCU1 and MOCU2 values')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: number of CPU cores - 1, 0 to disable multiprocessing)')
    parser.add_argument('--chunk_size', type=int, default=10,
                        help='Chunk size for multiprocessing (samples per worker batch)')
    args = parser.parse_args()
    
    # Configuration
    N = args.N
    K_max = args.K_max
    T = 4.0
    h = 1.0 / 160.0
    M = int(T / h)
    samples_per_type = args.samples_per_type
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Determine number of workers
    if args.num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # Leave one core free
    elif args.num_workers == 0:
        num_workers = 1  # Disable multiprocessing
    else:
        num_workers = args.num_workers
    
    # Set multiprocessing start method for CUDA compatibility
    # Note: Multiprocessing with CUDA has limited benefits since GPU already parallelizes RK4
    # Each process will create its own CUDA context, which can be memory-intensive
    if device == 'cuda' and torch.cuda.is_available() and num_workers > 1:
        # For CUDA, we need 'spawn' method to properly initialize CUDA contexts in each process
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, that's fine
            pass
        # Warning: Multiple CUDA contexts may cause OOM errors
        if num_workers > 4:
            print(f"\n  ⚠ Warning: Using {num_workers} workers with CUDA may cause GPU memory issues.")
            print(f"    Consider reducing --num_workers if you encounter GPU memory errors.")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("MOCU Dataset Generation")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Number of oscillators: {N}")
    print(f"  - Samples per type: {samples_per_type}")
    print(f"  - Expected total: {samples_per_type * 2} (some may be filtered)")
    print(f"  - Monte Carlo samples (K_max): {K_max}")
    print(f"  - Backend: PyCUDA (REQUIRED for data generation)")
    print(f"  - Note: PyCUDA is isolated from PyTorch to prevent segfaults")
    
    # GPU detection and verification
    if device == 'cuda' and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
        print(f"  - GPU: {gpu_name}")
        print(f"  - GPU Memory: {gpu_memory:.1f} GB total, {gpu_memory_allocated:.2f} MB allocated")
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - PyTorch CUDA Available: ✓ YES (GPU acceleration enabled)")
        
        # Test GPU computation
        print(f"\n  [GPU Test] Running quick GPU computation test...")
        test_tensor = torch.randn(100, 100, device='cuda')
        result = torch.matmul(test_tensor, test_tensor)
        torch.cuda.synchronize()
        if result.is_cuda:
            print(f"  [GPU Test] ✓ PASSED - GPU computation working correctly")
        else:
            print(f"  [GPU Test] ✗ FAILED - GPU computation not working!")
    else:
        print(f"  - CUDA Available: ✗ NO (using CPU - much slower)")
        if device == 'cuda':
            print(f"  - Warning: Device set to 'cuda' but CUDA not available, will use CPU")
    
    print(f"  - Parallel workers: {num_workers}")
    print(f"  - Compute MOCU twice: Always enabled (for stability - reduces Monte Carlo variance)")
    print(f"  - Chunk size: {args.chunk_size}")
    print(f"  - Note: Training script will split at 96%%/4%% automatically")
    
    # Debug mode info
    debug_gpu = os.getenv('DEBUG_GPU', 'false').lower() == 'true'
    if debug_gpu:
        print(f"\n  [DEBUG MODE] GPU debugging enabled (set DEBUG_GPU=false to disable)")
    print("=" * 80)
    
    # Helper function to generate samples with multiprocessing
    def generate_samples_parallel(coupling_type, num_samples, desc):
        """Generate samples using multiprocessing if enabled."""
        if num_workers <= 1:
            # Sequential processing (no multiprocessing)
            data = []
            with tqdm(total=num_samples, desc=desc, unit="sample", ncols=100) as pbar:
                for i in range(num_samples):
                    sample = generate_single_sample((N, K_max, h, M, T, coupling_type, device, None))
                    if sample is not None:
                        data.append(sample)
                    pbar.update(1)
        else:
            # Multiprocessing with progress tracking
            # Note: Each worker needs its own CUDA context, so we'll use CPU for sync checks
            # and let each process handle its own CUDA operations
            data = []
            
            # Prepare arguments for all samples
            # For multiprocessing with CUDA, we assign different GPU IDs or use CPU
            # Since CUDA contexts can't be easily shared, we'll process in smaller batches
            # and let each worker handle its own CUDA operations
            
            # Create argument tuples
            args_list = [(N, K_max, h, M, T, coupling_type, device, i % num_workers) 
                        for i in range(num_samples)]
            
            # Process in chunks to show progress
            with tqdm(total=num_samples, desc=desc, unit="sample", ncols=100) as pbar:
                # Use multiprocessing pool
                with mp.Pool(processes=num_workers) as pool:
                    # Process in chunks for better progress updates
                    chunk_size = args.chunk_size
                    for chunk_start in range(0, num_samples, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, num_samples)
                        chunk_args = args_list[chunk_start:chunk_end]
                        
                        # Process chunk
                        chunk_results = pool.map(generate_single_sample, chunk_args)
                        
                        # Collect valid samples
                        for sample in chunk_results:
                            if sample is not None:
                                data.append(sample)
                        
                        # Update progress
                        pbar.update(len(chunk_results))
        
        return data
    
    # Generate Type 1 data
    print("\n[1/4] Generating Type 1 data (per-edge coupling distribution)...")
    start_time = time.time()
    data_type1 = generate_samples_parallel('type1', samples_per_type, "  Type 1")
    elapsed = time.time() - start_time
    print(f"  ✓ Completed Type 1: {len(data_type1)} valid samples ({samples_per_type - len(data_type1)} skipped) in {elapsed:.1f}s")
    if len(data_type1) > 0:
        print(f"    Average: {elapsed/len(data_type1):.2f}s/sample, {len(data_type1)/elapsed:.2f} samples/s")
    
    # Generate Type 2 data
    print("\n[2/4] Generating Type 2 data (per-oscillator coupling distribution)...")
    start_time = time.time()
    data_type2 = generate_samples_parallel('type2', samples_per_type, "  Type 2")
    elapsed = time.time() - start_time
    print(f"  ✓ Completed Type 2: {len(data_type2)} valid samples ({samples_per_type - len(data_type2)} skipped) in {elapsed:.1f}s")
    if len(data_type2) > 0:
        print(f"    Average: {elapsed/len(data_type2):.2f}s/sample, {len(data_type2)/elapsed:.2f} samples/s")
    
    # Save JSON files if requested
    if args.save_json:
        print("\n[3/4] Saving intermediate JSON files...")
        with open(output_dir / f'{N}o_type1.json', 'w') as f:
            json.dump(data_type1, f)
        with open(output_dir / f'{N}o_type2.json', 'w') as f:
            json.dump(data_type2, f)
        print(f"  Saved: {N}o_type1.json, {N}o_type2.json")
    else:
        print("\n[3/4] Skipping JSON save (use --save_json to enable)")
    
    # Combine and shuffle
    print("\n[4/4] Converting to PyTorch Geometric format...")
    all_data = data_type1 + data_type2
    random.shuffle(all_data)
    
    print(f"  Total valid samples: {len(all_data)}")
    print(f"  Converting to PyTorch Geometric Data objects...")
    
    with tqdm(total=len(all_data), desc="  Converting", unit="sample", ncols=100) as pbar:
        pyg_data_list = []
        for data_item in all_data:
            # Convert one at a time with progress
            pyg_data = convert_to_pytorch_geometric([data_item])[0]
            pyg_data_list.append(pyg_data)
            pbar.update(1)
    
    total_samples = len(pyg_data_list)
    
    # Match original paper implementation: Split into train/test and save separately
    # Smart split that handles both small and large datasets (matching original code)
    if total_samples >= 2000:
        # Large dataset: use original logic (reserve at least 1000 for test)
        train_size = min(args.train_size, total_samples - 1000)
        test_size = total_samples - train_size
    else:
        # Small dataset: use percentage-based split
        min_test_samples = max(int(total_samples * 0.2), 10)  # At least 20% or 10 samples
        
        if total_samples < min_test_samples:
            print(f"\n⚠️  Warning: Only {total_samples} samples generated!")
            print(f"   This is too small for proper train/test split.")
            print(f"   Consider increasing samples_per_type in config.")
            train_size = total_samples
            test_size = 0
        else:
            max_train = total_samples - min_test_samples
            train_size = min(args.train_size, max_train)
            test_size = total_samples - train_size
    
    train_data = pyg_data_list[:train_size]
    test_data = pyg_data_list[train_size:] if test_size > 0 else []
    
    # Save PyTorch files separately (matching original paper naming convention)
    train_file = output_dir / f'{train_size}_{N}o_train.pth'
    torch.save(train_data, train_file)
    
    print("\n" + "=" * 80)
    print("Dataset Generation Complete!")
    print("=" * 80)
    print(f"Training set: {train_file} ({train_size} samples)")
    
    if test_size > 0:
        test_file = output_dir / f'{test_size}_{N}o_test.pth'
        torch.save(test_data, test_file)
        print(f"Test set:     {test_file} ({test_size} samples)")
    else:
        print(f"Test set:     None (dataset too small)")
    
    print("=" * 80)
    
    # Print statistics for training set (matching original code)
    if train_size > 0:
        train_mocu = [d.y.item() for d in train_data]
        print(f"\nMOCU Statistics (Training Set):")
        print(f"  Mean: {np.mean(train_mocu):.6f}")
        print(f"  Std:  {np.std(train_mocu):.6f}")
        print(f"  Min:  {np.min(train_mocu):.6f}")
        print(f"  Max:  {np.max(train_mocu):.6f}")


if __name__ == '__main__':
    main()

