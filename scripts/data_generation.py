"""
Combined data generation and dataset preparation script.
Generates training data with two different coupling distributions and converts to PyTorch Geometric format.
"""

import sys
from pathlib import Path
import time
import json
import argparse

# Get absolute path to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.mocu_cuda import MOCU
from src.core.sync_detection import mocu_comp
import numpy as np
import random
import torch
from torch_geometric.data import Data

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


def generate_single_sample(N, K_max, h, M, T, coupling_type='type1'):
    """Generate a single training sample with MOCU computation."""
    
    # Generate random natural frequencies
    w = np.zeros(N)
    for i in range(N):
        w[i] = 12 * (0.5 - random.random())
    
    # Generate coupling bounds based on type
    if coupling_type == 'type1':
        a_lower_bound, a_upper_bound = generate_coupling_type1(w, N)
    else:
        a_lower_bound, a_upper_bound = generate_coupling_type2(w, N)
    
    # Check if system is already synchronized
    a = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            a[i, j] = a_lower_bound[i, j] + 0.5 * (a_upper_bound[i, j] - a_lower_bound[i, j])
            a[j, i] = a[i, j]
    
    init_sync_check = mocu_comp(w, h, N, M, a)
    if init_sync_check == 1:
        return None  # Skip synchronized systems
    
    # Compute MOCU values (compute twice and average for stability)
    MOCU_val1 = MOCU(K_max, w, N, h, M, T, a_lower_bound, a_upper_bound, 0)
    MOCU_val2 = MOCU(K_max, w, N, h, M, T, a_lower_bound, a_upper_bound, 0)
    mean_MOCU = (MOCU_val1 + MOCU_val2) / 2
    
    # Create data dictionary
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
    parser = argparse.ArgumentParser(description='Generate training dataset for MOCU prediction')
    parser.add_argument('--N', type=int, default=5, help='Number of oscillators')
    parser.add_argument('--samples_per_type', type=int, default=37500, 
                        help='Number of samples per coupling type (total = 2 * samples_per_type)')
    parser.add_argument('--K_max', type=int, default=20480, help='Monte Carlo samples for MOCU')
    parser.add_argument('--train_size', type=int, default=70000, help='Training set size')
    parser.add_argument('--output_dir', type=str, default='../data', help='Output directory')
    parser.add_argument('--save_json', action='store_true', help='Save intermediate JSON files')
    args = parser.parse_args()
    
    # Configuration
    N = args.N
    K_max = args.K_max
    T = 4.0
    h = 1.0 / 160.0
    M = int(T / h)
    samples_per_type = args.samples_per_type
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("MOCU Dataset Generation")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Number of oscillators: {N}")
    print(f"  - Samples per type: {samples_per_type}")
    print(f"  - Total samples: {samples_per_type * 2}")
    print(f"  - Monte Carlo samples (K_max): {K_max}")
    print(f"  - Training set size: {args.train_size}")
    print(f"  - Test set size: {samples_per_type * 2 - args.train_size}")
    print("=" * 80)
    
    # Generate Type 1 data
    print("\n[1/4] Generating Type 1 data (per-edge coupling distribution)...")
    start_time = time.time()
    data_type1 = []
    
    for i in range(samples_per_type):
        sample = generate_single_sample(N, K_max, h, M, T, coupling_type='type1')
        if sample is not None:
            data_type1.append(sample)
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            eta = avg_time * (samples_per_type - i - 1)
            print(f"  Progress: {i+1}/{samples_per_type} samples | "
                  f"Avg: {avg_time:.2f}s/sample | ETA: {eta/60:.1f} min")
    
    print(f"  Completed Type 1: {len(data_type1)} valid samples")
    
    # Generate Type 2 data
    print("\n[2/4] Generating Type 2 data (per-oscillator coupling distribution)...")
    start_time = time.time()
    data_type2 = []
    
    for i in range(samples_per_type):
        sample = generate_single_sample(N, K_max, h, M, T, coupling_type='type2')
        if sample is not None:
            data_type2.append(sample)
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            eta = avg_time * (samples_per_type - i - 1)
            print(f"  Progress: {i+1}/{samples_per_type} samples | "
                  f"Avg: {avg_time:.2f}s/sample | ETA: {eta/60:.1f} min")
    
    print(f"  Completed Type 2: {len(data_type2)} valid samples")
    
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
    
    pyg_data_list = convert_to_pytorch_geometric(all_data)
    
    total_samples = len(pyg_data_list)
    
    # Smart split that handles both small and large datasets
    # For large datasets: keep old behavior (reserve 1000 for test)
    # For small datasets: use 20% split
    
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
    
    # Save PyTorch files
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
    
    # Print statistics (only if we have training data)
    if train_size > 0:
        train_mocu = [d.y.item() for d in train_data]
        print(f"\nMOCU Statistics (Training Set):")
        print(f"  Mean: {np.mean(train_mocu):.6f}")
        print(f"  Std:  {np.std(train_mocu):.6f}")
        print(f"  Min:  {np.min(train_mocu):.6f}")
        print(f"  Max:  {np.max(train_mocu):.6f}")


if __name__ == '__main__':
    main()

