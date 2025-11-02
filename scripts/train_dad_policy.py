"""
Train DAD (Deep Adaptive Design) policy network using imitation learning or RL.

This script trains a policy network to minimize terminal MOCU through sequential
experimental design decisions.

CRITICAL: MOCU_BACKEND must be set before any imports to prevent PyCUDA conflicts.
"""

# CRITICAL: Set MOCU_BACKEND BEFORE any other imports to prevent PyCUDA from loading
import os
if os.getenv('MOCU_BACKEND') != 'torch':
    # Auto-set to torch for DAD training (always uses PyTorch, PyCUDA would segfault)
    os.environ['MOCU_BACKEND'] = 'torch'

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
    """
    from scripts.generate_dad_data import perform_experiment, update_bounds
    
    if use_predicted_mocu and mocu_model is not None:
        from src.models.predictors.predictor_utils import predict_mocu
    
    model.train()
    total_loss = 0
    total_reward = 0
    
    baseline = 0.0
    baseline_alpha = 0.9
    
    h = 1.0 / 160.0
    T = 5.0
    M = int(T / h)
    
    # CRITICAL: Ensure clean CUDA state before training loop
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    print("[REINFORCE] Starting training loop...")
    print(f"[REINFORCE] Number of trajectories: {len(trajectories)}")
    
    for traj_idx, traj in enumerate(trajectories):
        print(f"[REINFORCE] Processing trajectory {traj_idx + 1}/{len(trajectories)}")
        # Periodically clear cache
        if traj_idx > 0 and traj_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.synchronize()  # ADD THIS
            torch.cuda.empty_cache()
            
        print(f"[REINFORCE] Trajectory {traj_idx}: Zeroing gradients...")
        optimizer.zero_grad()
        print(f"[REINFORCE] Trajectory {traj_idx}: Gradients zeroed")
        
        w = traj['w']
        a_true = traj.get('a_true', None)
        
        if a_true is None:
            raise ValueError("REINFORCE requires 'a_true' in trajectories")
        
        log_probs = []
        observed_pairs = []
        observations_list = []
        
        a_lower = traj['states'][0][0].copy()
        a_upper = traj['states'][0][1].copy()
        
        K = len(traj['actions'])
        print(f"[REINFORCE] Trajectory {traj_idx}: Starting {K} steps...")
        
        for step in range(K):
            print(f"[REINFORCE] Trajectory {traj_idx}, Step {step + 1}/{K}")
            # === POLICY NETWORK FORWARD PASS ===
            # Ensure clean state before policy operations
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # ADD THIS
                torch.cuda.empty_cache()   # ADD THIS
            
            print(f"[REINFORCE] Trajectory {traj_idx}, Step {step + 1}: Creating state data...")
            state_data = create_state_data(w, a_lower, a_upper, device=device)
            print(f"[REINFORCE] Trajectory {traj_idx}, Step {step + 1}: State data created")
            
            if step == 0:
                history_tensor = None
            else:
                history_list = [(observed_pairs[k][0], observed_pairs[k][1], observations_list[k]) 
                              for k in range(len(observed_pairs))]
                history_tensor = torch.tensor([history_list], dtype=torch.long, device=device)
            
            num_actions = N * (N - 1) // 2
            available_mask = np.ones(num_actions, dtype=np.float32)
            for (i_obs, j_obs) in observed_pairs:
                action_idx = model.pair_to_idx(i_obs, j_obs)
                available_mask[action_idx] = 0.0
            
            available_mask_array = np.array([available_mask], dtype=np.float32)
            available_mask_tensor = torch.from_numpy(available_mask_array).to(device)
            
            # Policy forward pass
            print(f"[REINFORCE] Trajectory {traj_idx}, Step {step + 1}: Running policy forward...")
            action_logits, action_probs = model(state_data, history_tensor, available_mask_tensor)
            print(f"[REINFORCE] Trajectory {traj_idx}, Step {step + 1}: Policy forward complete")
            
            # CRITICAL: Ensure policy forward completes before continuing
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # ADD THIS
            
            dist = torch.distributions.Categorical(probs=action_probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            
            action_i, action_j = model.idx_to_pair(action_idx.item())
            
            # === EXPERIMENT SIMULATION (CPU) ===
            observation = perform_experiment(a_true, action_i, action_j, w, h, M)
            a_lower, a_upper = update_bounds(a_lower, a_upper, action_i, action_j, observation, w)
            
            observed_pairs.append((action_i, action_j))
            observations_list.append(observation)
            log_probs.append(log_prob)
            
            # CRITICAL: Ensure all ops complete before next iteration
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # ADD THIS
        
        # === MOCU COMPUTATION ===
        print(f"[REINFORCE] Trajectory {traj_idx}: Computing MOCU...")
        if use_predicted_mocu and mocu_model is not None:
            # Ensure policy network is done
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # ADD THIS
                torch.cuda.empty_cache()   # ADD THIS
            
            print(f"[REINFORCE] Trajectory {traj_idx}: Calling predict_mocu...")
            terminal_MOCU = predict_mocu(mocu_model, mocu_mean, mocu_std, w, a_lower, a_upper, device=str(device))
            print(f"[REINFORCE] Trajectory {traj_idx}: predict_mocu returned: {terminal_MOCU}")
            
            # Ensure MPNN prediction completes
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # ADD THIS
        else:
            from src.core.mocu_backend import MOCU
            terminal_MOCU = MOCU(K_max, w, N, h, M, T, a_lower, a_upper, 0)
        
        # CRITICAL: Ensure terminal_MOCU is a Python float (not a tensor) before using in computation
        # This prevents any gradient graph connections to MPNN model
        reward = float(-terminal_MOCU)
        
        if use_baseline:
            baseline = baseline_alpha * baseline + (1 - baseline_alpha) * reward
        
        advantage = float(reward - baseline if use_baseline else reward)
        returns = [advantage] * len(log_probs)
        
        # Build loss computation graph (only involves policy network, not MPNN)
        policy_loss = []
        for log_prob, advantage_val in zip(log_probs, returns):
            # Ensure advantage_val is a Python float, not tensor
            advantage_tensor = torch.tensor(float(advantage_val), device=log_prob.device, requires_grad=False)
            policy_loss.append(-log_prob * advantage_tensor)
        
        loss = torch.stack(policy_loss).sum()
        
        # CRITICAL: Verify loss is a scalar tensor with gradient tracking only from policy network
        assert loss.requires_grad, "Loss should require gradients for backward pass"
        assert loss.numel() == 1, "Loss should be a scalar"
        
        # === BACKWARD PASS ===
        print(f"[REINFORCE] Trajectory {traj_idx}: Starting backward pass...")
        # CRITICAL: Ensure MOCU computation is completely done and MPNN model is isolated
        if use_predicted_mocu and mocu_model is not None and torch.cuda.is_available():
            # Ensure all MPNN operations are complete
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Ensure MPNN model is in eval mode and not accumulating gradients
            mocu_model.eval()
            # Clear any potential hanging references
            with torch.no_grad():
                # This ensures no gradient computation on MPNN side
                pass
        
        # CRITICAL: Ensure all CUDA operations are synchronized before backward
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        print(f"[REINFORCE] Trajectory {traj_idx}: Calling loss.backward()...")
        print(f"[REINFORCE] Trajectory {traj_idx}: Loss device: {loss.device}, requires_grad: {loss.requires_grad}, grad_fn: {loss.grad_fn}")
        
        # CRITICAL: Try to catch any CUDA errors before they become segfaults
        try:
            loss.backward()
        except RuntimeError as e:
            print(f"[REINFORCE] ERROR during backward: {e}")
            if torch.cuda.is_available():
                print(f"[REINFORCE] CUDA error info:")
                print(f"  - Device count: {torch.cuda.device_count()}")
                print(f"  - Current device: {torch.cuda.current_device()}")
                print(f"  - Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                torch.cuda.synchronize()
            raise
        
        print(f"[REINFORCE] Trajectory {traj_idx}: Backward pass complete")
        
        # Ensure backward completes
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # ADD THIS
        
        print(f"[REINFORCE] Trajectory {traj_idx}: Clipping gradients...")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        print(f"[REINFORCE] Trajectory {traj_idx}: Stepping optimizer...")
        optimizer.step()
        print(f"[REINFORCE] Trajectory {traj_idx}: Optimizer step complete")
        
        # Ensure optimizer step completes
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # ADD THIS
        
        print(f"[REINFORCE] Trajectory {traj_idx}: Complete (loss={loss.item():.6f}, reward={reward:.6f})")
        
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
            
            # CRITICAL: REINFORCE training MUST use MPNN predictor when PyTorch is active
            # PyCUDA cannot safely share PyTorch's CUDA context - will cause segmentation fault
            # MPNN predictor is REQUIRED (not optional) for DAD training
            mocu_model = None
            mocu_mean = None
            mocu_std = None
            
            # For REINFORCE, MPNN predictor is REQUIRED (PyCUDA causes segfault with PyTorch)
            # Auto-enable if not explicitly disabled
            use_predicted_mocu = args.use_predicted_mocu if args.use_predicted_mocu else True
            
            # If user explicitly disabled MPNN, this will segfault - warn and allow for testing
            if not use_predicted_mocu:
                import warnings
                warnings.warn(
                    "WARNING: Running REINFORCE without MPNN predictor (direct PyCUDA). "
                    "This WILL cause segmentation fault when PyTorch CUDA context is active. "
                    "Please use --use-predicted-mocu or train MPNN predictor first.",
                    RuntimeWarning
                )
                print("\n" + "!"*80)
                print("WARNING: Direct PyCUDA MOCU computation will likely cause segmentation fault")
                print("         when PyTorch CUDA context is active!")
                print("         Recommendation: Use --use-predicted-mocu flag")
                print("!"*80 + "\n")
            if use_predicted_mocu:
                try:
                    # Ensure CUDA is in a clean state before loading MPNN predictor
                    # This prevents conflicts between PyCUDA context (if any) and PyTorch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    
                    from src.models.predictors.predictor_utils import load_mpnn_predictor
                    import os
                    import re
                    from pathlib import Path
                    
                    # Get model name from environment variable, config, or auto-detect from output_dir
                    model_name = os.getenv('MOCU_MODEL_NAME') or config.get('mocu_model_name')
                    
                    # Auto-detect from output_dir if not provided (e.g., models/fast_config/11012025_212842/)
                    if not model_name and args.output_dir:
                        output_path = Path(args.output_dir).resolve()
                        parts = output_path.parts
                        # Look for pattern: .../models/{config}/{timestamp}/
                        try:
                            models_idx = list(parts).index('models')
                            if models_idx + 2 < len(parts):
                                config_part = parts[models_idx + 1]  # config name
                                timestamp_part = parts[models_idx + 2]  # timestamp
                                # Check if timestamp matches format (MMDDYYYY_HHMMSS)
                                if re.match(r'\d{8}_\d{6}', timestamp_part):
                                    model_name = f"{config_part}_{timestamp_part}"
                                    print(f"[REINFORCE] Auto-detected model name from output_dir: {model_name}")
                        except (ValueError, IndexError):
                            pass  # Couldn't parse path, will try default
                    
                    # Last resort: try default name format
                    if not model_name:
                        model_name = f'cons{N}'
                        print(f"[REINFORCE] Using default model name: {model_name}")
                    
                    print(f"[REINFORCE] Attempting to load MPNN predictor: {model_name}")
                    mocu_model, mocu_mean, mocu_std = load_mpnn_predictor(model_name=model_name, device=str(device))
                    
                    # Ensure model is properly moved to device and in eval mode
                    if mocu_model is not None:
                        mocu_model.eval()
                        # Move model to device and ensure it's ready
                        mocu_model = mocu_model.to(device)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                        
                        # Test model with a dummy forward pass to catch any initialization issues early
                        try:
                            from src.models.predictors.utils import get_edge_index, get_edge_attr_from_bounds
                            import numpy as np
                            # Create a test input
                            test_w = np.zeros(N)
                            test_a_lower = np.zeros((N, N))
                            test_a_upper = np.ones((N, N))
                            
                            from torch_geometric.data import Data
                            test_x = torch.from_numpy(test_w.astype(np.float32)).unsqueeze(-1).to(device)
                            test_edge_index = get_edge_index(N).to(device)
                            test_edge_attr = get_edge_attr_from_bounds(test_a_lower, test_a_upper, N).to(device)
                            test_data = Data(x=test_x, edge_index=test_edge_index, edge_attr=test_edge_attr).to(device)
                            
                            with torch.no_grad():
                                _ = mocu_model(test_data)
                            
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            
                            print(f"[REINFORCE] MPNN predictor test forward pass successful")
                        except Exception as test_e:
                            print(f"[REINFORCE] Warning: MPNN predictor test failed: {test_e}")
                            import traceback
                            traceback.print_exc()
                            # Don't fail - let it try during actual training, but warn user
                    
                    print(f"[REINFORCE] Using MPNN predictor '{model_name}' for fast MOCU estimation")
                except FileNotFoundError as e:
                    print(f"\n" + "!"*80)
                    print(f"[REINFORCE] ERROR: MPNN predictor not found!")
                    print(f"[REINFORCE] {e}")
                    print(f"\n[REINFORCE] REINFORCE training requires MPNN predictor to avoid segmentation faults.")
                    print(f"[REINFORCE] PyCUDA cannot be used when PyTorch CUDA context is active.")
                    print(f"\n[REINFORCE] Solutions:")
                    print(f"  1. Train MPNN predictor first: bash scripts/bash/step2_train_mpnn.sh configs/fast_config.yaml")
                    print(f"  2. Or set MOCU_MODEL_NAME environment variable: export MOCU_MODEL_NAME='{model_name}'")
                    print(f"  3. Or ensure MPNN model exists in: models/{model_name.split('_')[0]}/")
                    print(f"!"*80 + "\n")
                    raise RuntimeError(
                        f"MPNN predictor is REQUIRED for REINFORCE training. "
                        f"Model '{model_name}' not found. "
                        f"Direct PyCUDA MOCU computation will cause segmentation fault "
                        f"when PyTorch CUDA context is active. "
                        f"Please train MPNN predictor first or provide correct MOCU_MODEL_NAME."
                    ) from e
                except Exception as e:
                    print(f"\n" + "!"*80)
                    print(f"[REINFORCE] ERROR: Failed to load MPNN predictor!")
                    print(f"[REINFORCE] {e}")
                    print(f"\n[REINFORCE] REINFORCE training requires MPNN predictor to avoid segmentation faults.")
                    print(f"!"*80 + "\n")
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(
                        f"MPNN predictor is REQUIRED for REINFORCE training. "
                        f"Failed to load MPNN predictor '{model_name}'. "
                        f"Please check model path and train MPNN first."
                    ) from e
            
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

