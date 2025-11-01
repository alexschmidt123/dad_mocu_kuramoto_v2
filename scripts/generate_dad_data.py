"""
Generate trajectory data for training DAD (Deep Adaptive Design) policy network.

This script generates sequential decision trajectories for REINFORCE training.
Uses random expert (fast) - REINFORCE doesn't need expert labels, only a_true.
"""

import sys
from pathlib import Path
import time
import argparse
import random

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.sync_detection import mocu_comp
import numpy as np
import torch
from tqdm import tqdm


def generate_random_system(N):
    """Generate a random oscillator system."""
    # Generate random natural frequencies
    w = np.zeros(N)
    for i in range(N):
        w[i] = 12 * (0.5 - random.random())
    
    # Generate initial bounds
    a_upper_bound = np.zeros((N, N))
    a_lower_bound = np.zeros((N, N))
    
    uncertainty = 0.3 * random.random()
    
    for i in range(N):
        for j in range(i + 1, N):
            # Random multiplier
            if random.random() < 0.5:
                mul = 0.6 * random.random()
            else:
                mul = 1.1 * random.random()
            
            f_inv = np.abs(w[i] - w[j]) / 2.0
            a_upper_bound[i, j] = f_inv * (1 + uncertainty) * mul
            a_lower_bound[i, j] = f_inv * (1 - uncertainty) * mul
            a_upper_bound[j, i] = a_upper_bound[i, j]
            a_lower_bound[j, i] = a_lower_bound[i, j]
    
    # Generate true coupling strengths (ground truth)
    a_true = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            a_true[i, j] = a_lower_bound[i, j] + random.random() * (a_upper_bound[i, j] - a_lower_bound[i, j])
            a_true[j, i] = a_true[i, j]
    
    # Check if initially synchronized
    h = 1.0 / 160.0
    M = int(5.0 / h)
    init_sync = mocu_comp(w, h, N, M, a_true)
    
    return w, a_lower_bound, a_upper_bound, a_true, init_sync


def perform_experiment(a_true, i, j, w, h, M):
    """
    Perform experiment on pair (i, j) and observe if synchronized.
    
    Returns:
        observation: 1 if synchronized, 0 if not
    """
    # Determine if this pair is synchronized
    w_i = w[i]
    w_j = w[j]
    a_ij = a_true[i, j]
    
    # Create 2-oscillator system
    w_pair = np.array([w_i, w_j])
    a_pair = np.array([[0, a_ij], [a_ij, 0]])
    
    sync_result = mocu_comp(w_pair, h, 2, M, a_pair)
    
    return sync_result


def update_bounds(a_lower, a_upper, i, j, observation, w):
    """Update bounds based on observation."""
    a_lower_new = a_lower.copy()
    a_upper_new = a_upper.copy()
    
    f_inv = 0.5 * np.abs(w[i] - w[j])
    
    if observation == 1:  # Synchronized
        a_lower_new[i, j] = max(a_lower_new[i, j], f_inv)
        a_lower_new[j, i] = max(a_lower_new[j, i], f_inv)
    else:  # Not synchronized
        a_upper_new[i, j] = min(a_upper_new[i, j], f_inv)
        a_upper_new[j, i] = min(a_upper_new[j, i], f_inv)
    
    return a_lower_new, a_upper_new


def generate_trajectory(N, K, verbose=False):
    """
    Generate a single trajectory using random expert policy.
    
    For REINFORCE training: Random expert is sufficient (REINFORCE doesn't use expert actions).
    
    Args:
        N: Number of oscillators
        K: Number of sequential experiments
        verbose: Print progress
    
    Returns:
        trajectory: Dictionary containing the full trajectory
    """
    # Generate random system
    max_attempts = 100
    for attempt in range(max_attempts):
        w, a_lower_0, a_upper_0, a_true, init_sync = generate_random_system(N)
        if init_sync == 0:  # Not initially synchronized
            break
    else:
        return None  # Failed to find valid system
    
    # Simulation parameters
    h = 1.0 / 160.0
    T = 5.0
    M = int(T / h)
    
    # Initialize trajectory
    # NOTE: 'a_true' is REQUIRED for REINFORCE - needed to simulate experiments during policy rollouts.
    trajectory = {
        'w': w,                                    # Natural frequencies [N]
        'a_true': a_true,                          # Ground truth coupling [N, N] - REQUIRED
        'states': [(a_lower_0.copy(), a_upper_0.copy())],  # Initial bounds in states[0]
        'actions': []                              # Expert actions (only used to determine trajectory length K)
    }
    
    # Track which pairs have been observed
    observed_pairs = set()
    
    a_lower = a_lower_0.copy()
    a_upper = a_upper_0.copy()
    
    # Run K steps
    for step in range(K):
        # Random expert policy (sufficient for REINFORCE - expert actions not used during training)
        available_pairs = [(i, j) for i in range(N) for j in range(i+1, N) if (i, j) not in observed_pairs]
        i_selected, j_selected = random.choice(available_pairs)
        
        # Perform experiment
        observation = perform_experiment(a_true, i_selected, j_selected, w, h, M)
        
        # Update bounds
        a_lower, a_upper = update_bounds(a_lower, a_upper, i_selected, j_selected, observation, w)
        
        # Record trajectory data
        trajectory['actions'].append((i_selected, j_selected))
        trajectory['states'].append((a_lower.copy(), a_upper.copy()))
        
        observed_pairs.add((i_selected, j_selected))
        
        if verbose:
            print(f"Step {step+1}: Selected ({i_selected},{j_selected}), Obs={observation}")
    
    return trajectory


def main():
    parser = argparse.ArgumentParser(description='Generate DAD policy training data')
    parser.add_argument('--N', type=int, default=5, help='Number of oscillators')
    parser.add_argument('--K', type=int, default=4, help='Number of sequential experiments')
    parser.add_argument('--num-episodes', type=int, default=1000, help='Number of trajectories to generate')
    # Note: Removed expert-type and K-max - always use random expert for REINFORCE
    parser.add_argument('--output-dir', type=str, default='../data/', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("DAD Policy Training Data Generation")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Number of oscillators (N): {args.N}")
    print(f"  - Sequential experiments (K): {args.K}")
    print(f"  - Number of episodes: {args.num_episodes}")
    print(f"  - Expert policy: random (for REINFORCE training)")
    print("=" * 80)
    
    trajectories = []
    
    start_time = time.time()
    
    for episode in tqdm(range(args.num_episodes), desc="Generating trajectories"):
        trajectory = generate_trajectory(
            N=args.N,
            K=args.K,
            verbose=False
        )
        
        if trajectory is not None:
            trajectories.append(trajectory)
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (episode + 1)
            eta = avg_time * (args.num_episodes - episode - 1)
            print(f"\n  Progress: {episode+1}/{args.num_episodes} | "
                  f"Valid: {len(trajectories)} | "
                  f"Avg: {avg_time:.2f}s/episode | ETA: {eta/60:.1f} min")
    
    print(f"\n✓ Generated {len(trajectories)} valid trajectories")
    
    # Save data
    output_file = output_dir / f'dad_trajectories_N{args.N}_K{args.K}_random.pth'
    torch.save({
        'trajectories': trajectories,
        'config': {
            'N': args.N,
            'K': args.K,
            'expert_type': 'random',
            'num_episodes': len(trajectories)
        }
    }, output_file)
    
    print(f"✓ Saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("Data generation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

