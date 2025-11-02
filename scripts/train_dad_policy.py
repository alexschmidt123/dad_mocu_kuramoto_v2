"""
Train DAD (Deep Adaptive Design) policy network using imitation learning or RL.

This script trains a policy network to minimize terminal MOCU through sequential
experimental design decisions.
"""

import sys
from pathlib import Path
import argparse
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.models.policy_networks import DADPolicyNetwork, create_state_data


class DADTrajectoryDataset(Dataset):
    """Dataset for DAD policy training."""
    
    def __init__(self, trajectories):
        """
        Args:
            trajectories: List of trajectory dictionaries
        """
        self.trajectories = trajectories
        
        # Flatten into (state, history, action, available_mask) tuples
        self.samples = []
        
        for traj_idx, traj in enumerate(trajectories):
            w = traj['w']
            N = len(w)
            
            for step in range(len(traj['actions'])):
                # State at this step
                a_lower, a_upper = traj['states'][step]
                
                # History up to this step
                if step == 0:
                    history = []
                else:
                    history = [(traj['actions'][k][0], 
                               traj['actions'][k][1], 
                               traj['observations'][k]) 
                              for k in range(step)]
                
                # Expert action
                action_i, action_j = traj['actions'][step]
                
                # Available actions mask
                available_mask = traj['available_masks'][step]
                
                self.samples.append({
                    'w': w,
                    'a_lower': a_lower,
                    'a_upper': a_upper,
                    'history': history,
                    'action_i': action_i,
                    'action_j': action_j,
                    'available_mask': available_mask,
                    'terminal_MOCU': traj['terminal_MOCU']
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Custom collate function for batching trajectories."""
    # Extract data
    w_list = [sample['w'] for sample in batch]
    a_lower_list = [sample['a_lower'] for sample in batch]
    a_upper_list = [sample['a_upper'] for sample in batch]
    history_list = [sample['history'] for sample in batch]
    action_i_list = [sample['action_i'] for sample in batch]
    action_j_list = [sample['action_j'] for sample in batch]
    available_mask_list = [sample['available_mask'] for sample in batch]
    terminal_MOCU_list = [sample['terminal_MOCU'] for sample in batch]
    
    return {
        'w': w_list,
        'a_lower': a_lower_list,
        'a_upper': a_upper_list,
        'history': history_list,
        'action_i': action_i_list,
        'action_j': action_j_list,
        'available_mask': available_mask_list,
        'terminal_MOCU': terminal_MOCU_list
    }


def train_imitation(model, dataloader, optimizer, device, N):
    """
    Train one epoch using behavior cloning (imitation learning).
    
    Loss: Cross-entropy between policy and expert actions
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        batch_size = len(batch['w'])
        
        # Create state data for batch
        from torch_geometric.data import Batch as GeoBatch
        state_data_list = []
        for i in range(batch_size):
            state_data = create_state_data(
                batch['w'][i],
                batch['a_lower'][i],
                batch['a_upper'][i],
                device=device
            )
            state_data_list.append(state_data)
        
        state_batch = GeoBatch.from_data_list(state_data_list)
        
        # Pad histories to same length
        max_history_len = max(len(h) for h in batch['history'])
        if max_history_len == 0:
            history_batch = None
        else:
            history_batch = []
            for h in batch['history']:
                if len(h) == 0:
                    # Pad with dummy values
                    padded = [[0, 0, 0]] * max_history_len
                else:
                    padded = h + [[0, 0, 0]] * (max_history_len - len(h))
                history_batch.append(padded)
            history_batch = torch.tensor(history_batch, dtype=torch.long, device=device)
        
        # Available actions mask
        available_mask = torch.tensor(
            np.array(batch['available_mask']),
            dtype=torch.float32,
            device=device
        )
        
        # Forward pass
        action_logits, action_probs = model(state_batch, history_batch, available_mask)
        
        # Expert actions (convert to action indices)
        expert_actions = []
        for i in range(batch_size):
            action_idx = model.pair_to_idx(batch['action_i'][i], batch['action_j'][i])
            expert_actions.append(action_idx)
        expert_actions = torch.tensor(expert_actions, dtype=torch.long, device=device)
        
        # Behavior cloning loss (cross-entropy)
        loss = F.cross_entropy(action_logits, expert_actions)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * batch_size
        predicted = torch.argmax(action_logits, dim=-1)
        total_correct += (predicted == expert_actions).sum().item()
        total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy


def train_reinforce(model, trajectories, optimizer, device, N, gamma=0.99, K_max=20480, 
                    use_baseline=True, mocu_model=None, mocu_mean=None, mocu_std=None, use_predicted_mocu=False):
    """
    Train using REINFORCE policy gradient with MOCU DIRECTLY as the loss.
    
    This performs true on-policy learning: samples actions from the current policy,
    rolls out full trajectories, and uses terminal MOCU DIRECTLY as the reward signal.
    
    IMPORTANT: The loss is directly tied to terminal MOCU. Minimizing this loss
    directly minimizes the expected terminal MOCU, which is the true objective.
    
    Args:
        mocu_model: Loaded MPNNPlusPredictor model (from load_mpnn_predictor)
        mocu_mean: Normalization mean for MPNN predictor
        mocu_std: Normalization std for MPNN predictor
        use_predicted_mocu: If True, use MPNN predictor for fast MOCU estimation (recommended)
                           If False, use direct CUDA MOCU computation (slow but exact)
    
    Required data in trajectories:
        - 'w': Natural frequencies [N]
        - 'a_true': Ground truth coupling matrix [N, N] (REQUIRED for RL)
        - 'states': List of (a_lower, a_upper) tuples, with states[0] being initial bounds
        - 'actions': List of expert actions (used to determine trajectory length K)
    
    Reward: Negative terminal MOCU (want to maximize = minimize MOCU)
    
    Args:
        model: DADPolicyNetwork to train
        trajectories: List of trajectory dicts (must contain 'a_true')
        optimizer: PyTorch optimizer
        device: torch device
        N: Number of oscillators
        gamma: Discount factor (default 0.99, but we use undiscounted terminal reward)
        K_max: Monte Carlo samples for MOCU computation
        use_baseline: If True, subtract running average as baseline (reduces variance)
    """
    # Lazy import: Only import PyCUDA when actually needed (when use_predicted_mocu=False)
    # This avoids initializing PyCUDA context when using MPNN predictor, preventing stream conflicts
    # This matches the original paper's pattern: separate usage of MPNN and PyCUDA
    from scripts.generate_dad_data import perform_experiment, update_bounds
    
    model.train()
    total_loss = 0
    total_reward = 0
    
    # Running baseline for variance reduction
    baseline = 0.0
    baseline_alpha = 0.9  # Exponential moving average
    
    h = 1.0 / 160.0
    T = 5.0
    M = int(T / h)
    
    for traj in trajectories:
        optimizer.zero_grad()
        
        w = traj['w']
        a_true = traj.get('a_true', None)
        
        # REQUIRED: a_true is necessary for REINFORCE to perform experiments
        if a_true is None:
            raise ValueError(
                "REINFORCE training requires 'a_true' in trajectories to perform experiments. "
                "Please regenerate data with generate_dad_data.py (it includes a_true by default)."
            )
        
        # On-policy rollout: sample actions from current policy
        log_probs = []
        observed_pairs = []
        observations_list = []  # Track observations for history
        
        # Start with initial bounds (from states[0])
        a_lower = traj['states'][0][0].copy()
        a_upper = traj['states'][0][1].copy()
        
        K = len(traj['actions'])  # Number of steps (determined by expert trajectory length)
        
        for step in range(K):
            # Create state from current bounds
            state_data = create_state_data(w, a_lower, a_upper, device=device)
            
            # History from observed pairs
            if step == 0:
                history_tensor = None
            else:
                history_list = [(observed_pairs[k][0], 
                               observed_pairs[k][1], 
                               observations_list[k]) 
                              for k in range(len(observed_pairs))]
                # Convert to tensor format expected by model
                history_tensor = torch.tensor([history_list], dtype=torch.long, device=device)
            
            # Available mask: exclude already observed pairs
            num_actions = N * (N - 1) // 2
            available_mask = np.ones(num_actions, dtype=np.float32)
            for (i_obs, j_obs) in observed_pairs:
                action_idx = model.pair_to_idx(i_obs, j_obs)
                available_mask[action_idx] = 0.0
            
            # Convert numpy array to tensor properly (fixes warning and ensures correct device)
            available_mask_array = np.array([available_mask], dtype=np.float32)
            available_mask_tensor = torch.from_numpy(available_mask_array).to(device)
            
            # Sample action from current policy (NOT expert action!)
            action_logits, action_probs = model(state_data, history_tensor, available_mask_tensor)
            
            # Sample from policy distribution
            dist = torch.distributions.Categorical(probs=action_probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            
            # Convert to (i, j) pair
            action_i, action_j = model.idx_to_pair(action_idx.item())
            
            # Perform experiment using ground truth (get observation)
            observation = perform_experiment(a_true, action_i, action_j, w, h, M)
            
            # Update bounds based on observation
            a_lower, a_upper = update_bounds(a_lower, a_upper, action_i, action_j, observation, w)
            
            observed_pairs.append((action_i, action_j))
            observations_list.append(observation)
            log_probs.append(log_prob)
        
        # Compute terminal MOCU for this policy rollout
        # This is the DIRECT objective we want to minimize
        # Use fast MPNN prediction if available, otherwise use slow CUDA computation
        if use_predicted_mocu and mocu_model is not None:
            # Fast: Use MPNN predictor (reuses same logic as iNN/NN from paper 2023)
            # IMPORTANT: When using MPNN, we don't import PyCUDA at all
            # This keeps them separate, just like the original paper
            from src.models.predictors.mocu_predictor_utils import predict_mocu
            terminal_MOCU = predict_mocu(mocu_model, mocu_mean, mocu_std, w, a_lower, a_upper, device=str(device))
        else:
            # Slow but exact: Use direct CUDA MOCU computation
            # Only import PyCUDA here when actually needed (lazy import)
            # This ensures PyCUDA context is NOT initialized when using MPNN
            from src.core.mocu_cuda import MOCU
            terminal_MOCU = MOCU(K_max, w, N, h, M, T, a_lower, a_upper, 0)
        
        # Convert MOCU to reward signal for policy gradient
        # Reward = -MOCU (so maximizing reward = minimizing MOCU)
        # This ensures the loss directly optimizes terminal MOCU
        reward = -terminal_MOCU
        
        # Update baseline (exponential moving average) for variance reduction
        if use_baseline:
            baseline = baseline_alpha * baseline + (1 - baseline_alpha) * reward
        
        # Baseline-subtracted advantage (reduces variance without changing optimal policy)
        advantage = reward - baseline if use_baseline else reward
        returns = [advantage] * len(log_probs)
        
        # Policy gradient loss: DIRECTLY optimizes terminal MOCU
        # Loss = -sum(log_prob * advantage) where advantage = -MOCU - baseline
        # This means: minimizing loss = maximizing sum(log_prob * (-MOCU))
        #           = minimizing expected terminal MOCU
        policy_loss = []
        for log_prob, advantage_val in zip(log_probs, returns):
            policy_loss.append(-log_prob * advantage_val)
        
        # Total loss for this trajectory: directly tied to terminal MOCU
        loss = torch.stack(policy_loss).sum()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_reward += reward
    
    avg_loss = total_loss / len(trajectories)
    avg_reward = total_reward / len(trajectories)
    
    return avg_loss, avg_reward


def main():
    parser = argparse.ArgumentParser(description='Train DAD policy network')
    parser.add_argument('--data-path', type=str, required=True, help='Path to trajectory data')
    parser.add_argument('--method', type=str, default='reinforce', 
                       choices=['imitation', 'reinforce'], 
                       help='Training method: "reinforce" (RL, uses MOCU directly) or "imitation" (behavior cloning)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--encoding-dim', type=int, default=32, help='Encoding dimension')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='../models/', help='Output directory')
    parser.add_argument('--name', type=str, default='dad_policy', help='Model name')
    parser.add_argument('--use-predicted-mocu', action='store_true',
                       help='Use MPNN predictor for fast MOCU estimation (recommended). Requires trained MPNN model.')
    args = parser.parse_args()
    
    # Load data
    print("Loading trajectory data...")
    data = torch.load(args.data_path, weights_only=False)
    trajectories = data['trajectories']
    config = data['config']
    N = config['N']
    K = config['K']
    
    print(f"Loaded {len(trajectories)} trajectories")
    print(f"System: N={N}, K={K}")
    
    # Create dataset and dataloader
    if args.method == 'imitation':
        dataset = DADTrajectoryDataset(trajectories)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        print(f"Dataset: {len(dataset)} samples")
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = DADPolicyNetwork(
        N=N,
        hidden_dim=args.hidden_dim,
        encoding_dim=args.encoding_dim
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("\n" + "=" * 80)
    print(f"Training using {args.method}")
    print("=" * 80)
    
    train_losses = []
    train_accs = []
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        if args.method == 'imitation':
            loss, acc = train_imitation(model, dataloader, optimizer, device, N)
            train_losses.append(loss)
            train_accs.append(acc)
            
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss:.6f} | Acc: {acc:.4f} | "
                  f"Time: {time.time()-start_time:.2f}s")
        
        elif args.method == 'reinforce':
            # Get K_max from config or use default
            K_max = config.get('K_max', 20480)
            
            # Optionally use MPNN predictor for fast MOCU prediction
            # IMPORTANT: Ensure any previous PyCUDA context is cleared before loading MPNN
            # This ensures clean separation between PyCUDA and PyTorch/cuDNN usage
            mocu_model = None
            mocu_mean = None
            mocu_std = None
            use_predicted_mocu = args.use_predicted_mocu
            if use_predicted_mocu:
                try:
                    # Ensure CUDA is in a clean state before loading MPNN predictor
                    # This prevents conflicts between PyCUDA context (if any) and PyTorch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    
                    from src.models.predictors.mocu_predictor_utils import load_mpnn_predictor
                    # Get model name from environment variable (set by run.sh) or config, or use default
                    import os
                    model_name = os.getenv('MOCU_MODEL_NAME') or config.get('mocu_model_name') or f'cons{N}'
                    mocu_model, mocu_mean, mocu_std = load_mpnn_predictor(model_name=model_name, device=str(device))
                    
                    # Ensure model is properly moved to device and in eval mode
                    if mocu_model is not None:
                        mocu_model.eval()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    
                    print(f"[REINFORCE] Using MPNN predictor '{model_name}' for fast MOCU estimation")
                except FileNotFoundError as e:
                    print(f"[REINFORCE] Warning: {e}")
                    print(f"[REINFORCE] Falling back to direct CUDA MOCU computation (slow)")
                    use_predicted_mocu = False
                except Exception as e:
                    print(f"[REINFORCE] Error loading MPNN predictor: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"[REINFORCE] Falling back to direct CUDA MOCU computation (slow)")
                    use_predicted_mocu = False
                    mocu_model = None
            
            loss, reward = train_reinforce(
                model, trajectories, optimizer, device, N, 
                K_max=K_max,
                mocu_model=mocu_model,
                mocu_mean=mocu_mean,
                mocu_std=mocu_std,
                use_predicted_mocu=use_predicted_mocu
            )
            train_losses.append(loss)
            
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss:.6f} | Reward: {reward:.6f} | "
                  f"Time: {time.time()-start_time:.2f}s")
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f'{args.name}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'N': N,
            'hidden_dim': args.hidden_dim,
            'encoding_dim': args.encoding_dim
        },
        'train_config': {
            'method': args.method,
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size
        }
    }, model_path)
    
    print(f"\n✓ Model saved to: {model_path}")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2 if args.method == 'imitation' else 1, figsize=(12, 4))
    
    if args.method == 'imitation':
        axes[0].plot(train_losses)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True)
        
        axes[1].plot(train_accs)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].grid(True)
    else:
        axes.plot(train_losses)
        axes.set_xlabel('Epoch')
        axes.set_ylabel('Loss')
        axes.set_title('Training Loss (REINFORCE)')
        axes.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{args.name}_training_curve.png', dpi=300)
    print(f"✓ Training curve saved to: {output_dir / f'{args.name}_training_curve.png'}")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

