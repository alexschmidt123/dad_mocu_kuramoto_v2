"""
Train DAD (Deep Adaptive Design) policy network using imitation learning or RL.

This script trains a policy network to minimize terminal MOCU through sequential
experimental design decisions.

DAD policy training script using REINFORCE with MPNN predictor.
"""

# Standard library imports
import sys
import os
import time
import argparse
from pathlib import Path

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.models.policy_networks import DADPolicyNetwork, create_state_data

# Optional imports (for visualization only)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ========== CUDA Configuration ==========
# CUDA_LAUNCH_BLOCKING: Set to '1' for debugging (slower), '0' for production (faster)
CUDA_LAUNCH_BLOCKING = os.getenv('CUDA_LAUNCH_BLOCKING', '0')
os.environ['CUDA_LAUNCH_BLOCKING'] = CUDA_LAUNCH_BLOCKING

# cuDNN: PyTorch enables it by default for optimal performance
# Only disable if explicitly requested via DISABLE_CUDNN='1' (for debugging)
if os.getenv('DISABLE_CUDNN', '0') == '1':
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    print("[TRAIN] cuDNN disabled (via DISABLE_CUDNN='1')")
else:
    # PyTorch enables cuDNN by default, but enable benchmark mode for optimal performance
    torch.backends.cudnn.benchmark = True

# Initialize CUDA context explicitly
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()


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
                    use_baseline=True, mocu_model=None, mocu_mean=None, mocu_std=None, use_predicted_mocu=False,
                    epoch_num=None):
    """
    Train using REINFORCE policy gradient with MOCU DIRECTLY as the loss.
    """
    from scripts.generate_dad_data import update_bounds
    
    if use_predicted_mocu and mocu_model is not None:
        from src.models.predictors.predictor_utils import predict_mocu
    
    model.train()
    total_loss = 0
    total_reward = 0
    
    # Initialize baseline as None - compute it from first batch of rewards
    # This prevents baseline from starting at 0 and causing large initial advantages
    baseline = None
    baseline_alpha = 0.99  # Slower baseline update to prevent premature convergence
    reward_list = []  # Track rewards for better baseline initialization
    all_rewards = []  # Track all rewards for batch baseline computation
    
    h = 1.0 / 160.0
    T = 5.0
    M = int(T / h)
    
    # CRITICAL: Ensure clean CUDA state before training loop
    if torch.cuda.is_available():
        # Force PyTorch to create its own CUDA context
        # Try to reset/clear any existing CUDA state
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force a dummy operation to ensure PyTorch CUDA context is active
            dummy = torch.zeros(1, device=device)
            torch.cuda.synchronize()
            del dummy
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[WARNING] CUDA context reset encountered issue: {e}")
            print("[WARNING] Continuing anyway - may have CUDA issues")
    
    # Use tqdm for progress bar (update less frequently to reduce verbosity)
    epoch_desc = f"Epoch {epoch_num+1}" if epoch_num is not None else "Epoch"
    traj_pbar = tqdm(enumerate(trajectories), total=len(trajectories), desc=epoch_desc, unit="traj", ncols=100, leave=False, mininterval=1.0)
    
    # Update progress bar only every N trajectories to reduce verbosity
    update_frequency = max(1, len(trajectories) // 20)  # Update ~20 times per epoch
    
    for traj_idx, traj in traj_pbar:
        # Periodically clear cache and add error recovery
        if traj_idx > 0 and traj_idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Add periodic CUDA error check to catch issues early
            try:
                # Quick CUDA health check (non-blocking)
                _ = torch.cuda.current_device()
            except Exception as e:
                print(f"[REINFORCE] WARNING: CUDA health check failed at trajectory {traj_idx}: {e}")
                print("[REINFORCE] Attempting CUDA context recovery...")
                try:
                    torch.cuda.empty_cache()
                    # REMOVED: torch.cuda.synchronize() - can cause hangs if CUDA is corrupted
                    # Just clear cache, don't synchronize
                except:
                    print("[REINFORCE] CUDA recovery failed - continuing anyway")
            
        optimizer.zero_grad()
        
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
        
        for step in range(K):
            # === POLICY NETWORK FORWARD PASS ===
            state_data = create_state_data(w, a_lower, a_upper, device=device)
            
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
            action_logits, action_probs = model(state_data, history_tensor, available_mask_tensor)
            
            
            dist = torch.distributions.Categorical(probs=action_probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            
            action_i, action_j = model.idx_to_pair(action_idx.item())
            
            # === EXPERIMENT SIMULATION ===
            # Use CPU-based sync detection (no torchdiffeq during training to avoid CUDA issues)
            # This is fast enough for 2-oscillator systems and avoids CUDA context conflicts
            from src.core.sync_detection import determineSyncTwo
            w_i = w[action_i]
            w_j = w[action_j]
            a_ij = a_true[action_i, action_j]
            observation = determineSyncTwo(w_i, w_j, h, 2, M, a_ij)
            a_lower, a_upper = update_bounds(a_lower, a_upper, action_i, action_j, observation, w)
            
            observed_pairs.append((action_i, action_j))
            observations_list.append(observation)
            log_probs.append(log_prob)
            
            # Only synchronize if CUDA_LAUNCH_BLOCKING is enabled (debugging mode)
            # For production, let CUDA operations run asynchronously
            if torch.cuda.is_available() and CUDA_LAUNCH_BLOCKING == '1':
                torch.cuda.synchronize()
        
        # === MOCU COMPUTATION ===
        # PREFERRED: Use pre-computed MOCU if available (avoids MPNN predictor during training)
        if 'terminal_MOCU' in traj:
            terminal_MOCU = traj['terminal_MOCU']
        elif use_predicted_mocu and mocu_model is not None:
            # FALLBACK: Compute MOCU using MPNN predictor during training
            # Use no_grad context for MPNN prediction to prevent gradient issues
            with torch.no_grad():
                terminal_MOCU = predict_mocu(mocu_model, mocu_mean, mocu_std, w, a_lower, a_upper, device=str(device))
            
            # Only synchronize if CUDA_LAUNCH_BLOCKING is enabled (debugging mode)
            if torch.cuda.is_available() and CUDA_LAUNCH_BLOCKING == '1':
                torch.cuda.synchronize()
        else:
            # ERROR: No MOCU available - this should not happen with proper data generation
            raise RuntimeError(
                "No terminal MOCU available in trajectory and no MPNN predictor provided. "
                "Please regenerate data with --use-mpnn-predictor or provide mocu_model. "
                "torchdiffeq is NOT used during training (only for evaluation)."
            )
        
        # CRITICAL: Ensure terminal_MOCU is a Python float (not a tensor) before using in computation
        # This prevents any gradient graph connections to MPNN model
        # Convert to float and ensure it's detached from any computation graph
        if isinstance(terminal_MOCU, torch.Tensor):
            terminal_MOCU = terminal_MOCU.detach().cpu().item()
        reward = float(-terminal_MOCU)
        reward_list.append(reward)
        all_rewards.append(reward)
        
        # Compute advantage using reward normalization (z-score) to prevent baseline from canceling signal
        # This is more robust than adaptive baseline when reward variance is low
        if len(all_rewards) >= 50:
            # Use running statistics for normalization
            reward_mean = np.mean(all_rewards)
            reward_std = np.std(all_rewards)
            
            if reward_std > 1e-6:  # Avoid division by zero
                # Z-score normalization: (reward - mean) / std
                # This ensures advantages have consistent scale regardless of reward distribution
                advantage = float((reward - reward_mean) / reward_std)
                
                # Clip to prevent extreme values that could cause gradient explosion
                advantage = np.clip(advantage, -5.0, 5.0)
            else:
                # Extremely low variance: all rewards are nearly identical
                # In this case, use raw reward (no normalization) to maintain some signal
                advantage = float(reward - reward_mean) if use_baseline else float(reward)
        else:
            # Not enough samples yet: use simple baseline or raw reward
            if use_baseline:
                if baseline is None:
                    # Initialize baseline after collecting some rewards
                    if len(all_rewards) >= 10:
                        baseline = np.mean(all_rewards)
                else:
                    # Update baseline slowly
                    baseline = baseline_alpha * baseline + (1 - baseline_alpha) * reward
                
                advantage = float(reward - baseline) if baseline is not None else float(reward)
            else:
                advantage = float(reward)
        
        returns = [advantage] * len(log_probs)
        
        # CRITICAL: Validate all log_probs tensors before building loss computation graph
        # This prevents invalid memory access during loss construction
        valid_log_probs = []
        for idx, log_prob in enumerate(log_probs):
            try:
                # Validate tensor is valid and on correct device
                assert isinstance(log_prob, torch.Tensor), f"log_prob {idx} is not a tensor"
                assert log_prob.requires_grad, f"log_prob {idx} must require gradients"
                assert log_prob.numel() == 1, f"log_prob {idx} must be scalar (got shape {log_prob.shape})"
                assert log_prob.device.type == device.type, f"log_prob {idx} device mismatch: {log_prob.device} vs {device}"
                # Verify tensor data is valid (not NaN or Inf)
                assert not (torch.isnan(log_prob) or torch.isinf(log_prob)), f"log_prob {idx} is NaN/Inf"
                valid_log_probs.append(log_prob)
            except (AssertionError, RuntimeError) as e:
                print(f"[REINFORCE] ERROR: Invalid log_prob at step {idx}: {e}")
                raise RuntimeError(f"Invalid log_prob tensor at step {idx}: {e}") from e
        
        if len(valid_log_probs) != len(log_probs):
            raise RuntimeError(f"Only {len(valid_log_probs)}/{len(log_probs)} log_probs are valid")
        
        # Build loss computation graph (only involves policy network, not MPNN)
        # Only synchronize if CUDA_LAUNCH_BLOCKING is enabled (debugging mode)
        # Use validated log_probs to prevent memory access errors
        policy_loss = []
        for idx, (log_prob, advantage_val) in enumerate(zip(valid_log_probs, returns)):
            try:
                # Validate advantage_val is a valid Python float
                assert isinstance(advantage_val, (int, float)), f"advantage_val {idx} must be numeric"
                assert not (isinstance(advantage_val, float) and (np.isnan(advantage_val) or np.isinf(advantage_val))), \
                    f"advantage_val {idx} is NaN/Inf"
                
                # Ensure advantage_tensor is created on same device as log_prob
                advantage_tensor = torch.tensor(float(advantage_val), device=log_prob.device, requires_grad=False, dtype=log_prob.dtype)
                
                # Build loss component with explicit error checking
                loss_component = -log_prob * advantage_tensor
                
                # Validate loss component immediately
                assert isinstance(loss_component, torch.Tensor), f"loss_component {idx} is not a tensor"
                assert loss_component.requires_grad == log_prob.requires_grad, f"loss_component {idx} gradient flag mismatch"
                assert not (torch.isnan(loss_component) or torch.isinf(loss_component)), \
                    f"loss_component {idx} is NaN/Inf"
                
                policy_loss.append(loss_component)
            except (AssertionError, RuntimeError) as e:
                print(f"[REINFORCE] ERROR: Failed to build loss component at step {idx}: {e}")
                raise RuntimeError(f"Failed to build loss component at step {idx}: {e}") from e
        
        # CRITICAL: Validate before combining loss components
        # Use direct accumulation instead of stack+sum to avoid potential memory issues
        # REMOVED: torch.cuda.synchronize() here - not needed and can cause hangs
        # PyTorch handles CUDA synchronization automatically
        
        try:
            # ALTERNATIVE APPROACH: Accumulate loss directly instead of stacking
            # This avoids potential issues with torch.stack() and creates a simpler graph
            loss = policy_loss[0]
            for component in policy_loss[1:]:
                loss = loss + component
            
            # Validate after accumulation
            assert isinstance(loss, torch.Tensor), "Loss is not a tensor after accumulation"
            assert loss.requires_grad, "Loss should require gradients for backward pass"
            assert loss.numel() == 1, f"Loss should be a scalar (got shape {loss.shape})"
            assert not (torch.isnan(loss) or torch.isinf(loss)), "Loss is NaN/Inf - computation graph corrupted"
            
        except (RuntimeError, AssertionError) as e:
            print(f"[REINFORCE] ERROR during loss accumulation: {e}")
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except:
                    pass
            raise
        
        # CRITICAL: Ensure loss tensor is a pure PyTorch tensor, not mixed with numpy
        # Validate loss tensor is a valid PyTorch tensor
        if not isinstance(loss, torch.Tensor):
            raise RuntimeError(f"Loss is not a PyTorch tensor: {type(loss)}")
        
        # CRITICAL: Ensure loss is on the correct device and not accidentally a numpy array
        # If somehow loss became a numpy array, convert it back (shouldn't happen, but safety check)
        if isinstance(loss, np.ndarray):
            print(f"[REINFORCE] WARNING: Loss became numpy array - converting to tensor (this shouldn't happen)")
            loss = torch.from_numpy(loss).to(device).requires_grad_(True)
        
        # CRITICAL: Verify device consistency - all tensors in computation graph must be on same device
        if loss.device.type == 'cuda':
            # Check that loss.data_ptr() points to valid CUDA memory (PyTorch managed)
            try:
                # This will raise if tensor is corrupted or not a valid PyTorch CUDA tensor
                loss_ptr = loss.data_ptr()
                assert loss_ptr > 0, "Loss tensor has invalid data pointer"
            except Exception as e:
                print(f"[REINFORCE] ERROR: Loss tensor has invalid CUDA memory pointer: {e}")
                raise RuntimeError(f"Loss tensor CUDA memory corruption: {e}") from e
        
        # === BACKWARD PASS ===
        # CRITICAL: Ensure MOCU computation is completely done and MPNN model is isolated
        if use_predicted_mocu and mocu_model is not None:
            # Ensure MPNN model is in eval mode and not accumulating gradients
            mocu_model.eval()
        
        # REMOVED: torch.cuda.synchronize() before backward
        # This was causing hangs - PyTorch handles synchronization automatically
        # Only synchronize if explicitly debugging (CUDA_LAUNCH_BLOCKING='1')
        
        # CRITICAL: Segfault diagnosis
        # The segfault occurs in PyTorch's C++ autograd engine during backward()
        # All Python-level validations pass, indicating the issue is deep in PyTorch C++
        # This suggests: PyTorch/CUDA version bug, memory corruption, or driver issue
        
        # CRITICAL: Try standard backward() approach
        # Note: If this segfaults, it's in PyTorch C++ code and we can't catch it with try-except
        try:
            # Ensure we're on the correct device before backward
            if loss.device.type == 'cuda' and torch.cuda.is_available():
                # REMOVED: Multiple torch.cuda.synchronize() calls that can cause hangs
                # Only verify CUDA context is still valid (no sync)
                try:
                    test_op = torch.zeros(1, device=loss.device)
                    del test_op
                    # Don't synchronize - let PyTorch handle it
                except Exception:
                    pass  # Silent check
            
            # Perform backward pass
            # REMOVED: Extra try-except nesting - simplified error handling
            loss.backward(retain_graph=False)
        except RuntimeError as backward_error:
            # PyTorch's backward may raise RuntimeError for CUDA errors
            print(f"\n{'='*80}")
            print(f"[BACKWARD ERROR] RuntimeError during backward pass:")
            print(f"[BACKWARD ERROR] {backward_error}")
            print(f"[BACKWARD ERROR] This indicates a CUDA error in autograd engine")
            print(f"{'='*80}\n")
            raise
        except RuntimeError as e:
            print(f"[REINFORCE] ERROR during backward: {e}")
            if torch.cuda.is_available():
                print(f"[REINFORCE] CUDA error info:")
                print(f"  - Device count: {torch.cuda.device_count()}")
                print(f"  - Current device: {torch.cuda.current_device()}")
                print(f"  - Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                try:
                    # REMOVED: torch.cuda.synchronize() - can cause hangs if CUDA is corrupted
                    # Just report the error without synchronizing
                    pass
                except:
                    print("[REINFORCE] CUDA state query failed - context may be corrupted")
            raise
        except Exception as e:
            # Catch any other exception (including potential segfault precursors)
            print(f"[REINFORCE] UNEXPECTED ERROR during backward: {type(e).__name__}: {e}")
            if torch.cuda.is_available():
                try:
                    print(f"[REINFORCE] CUDA state before error:")
                    print(f"  - Device count: {torch.cuda.device_count()}")
                    print(f"  - Current device: {torch.cuda.current_device()}")
                    print(f"  - Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                except:
                    print("[REINFORCE] Could not query CUDA state")
            raise
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # REMOVED: torch.cuda.synchronize() after optimizer step
        # PyTorch handles CUDA synchronization automatically
        # This sync was causing hangs when CUDA context was corrupted
        
        # Update progress bar only periodically to reduce verbosity
        current_loss = loss.item()
        if (traj_idx + 1) % update_frequency == 0 or (traj_idx + 1) == len(trajectories):
            # Show diagnostic info including normalized statistics
            reward_mean = np.mean(all_rewards) if len(all_rewards) > 0 else 0.0
            reward_std = np.std(all_rewards) if len(all_rewards) > 1 else 0.0
            # Show advantage magnitude to diagnose if it's too small
            adv_magnitude = abs(advantage)
            traj_pbar.set_postfix({
                'loss': f'{current_loss:.4f}', 
                'reward': f'{reward:.4f}',
                'adv': f'{advantage:.4f}',
                '|adv|': f'{adv_magnitude:.4f}',
                'r_std': f'{reward_std:.4f}'
            })
            
            # Warn if advantage is too small (indicating baseline is canceling signal)
            if len(all_rewards) > 50 and adv_magnitude < 0.01:
                if not hasattr(train_reinforce, '_small_adv_warned'):
                    traj_pbar.write(f"[WARNING] Advantage magnitude is very small ({adv_magnitude:.6f}) - baseline may be canceling learning signal")
                    train_reinforce._small_adv_warned = True
        
        total_loss += current_loss
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
    
    epoch_pbar = tqdm(range(args.epochs), desc="Training", unit="epoch", ncols=100, mininterval=1.0)
    
    for epoch in epoch_pbar:
        start_time = time.time()
        
        if args.method == 'imitation':
            loss, acc = train_imitation(model, dataloader, optimizer, device, N)
            train_losses.append(loss)
            train_accs.append(acc)
            
            epoch_pbar.set_postfix({'loss': f'{loss:.4f}', 'acc': f'{acc:.4f}', 
                                   'time': f'{time.time()-start_time:.1f}s'})
        
        elif args.method == 'reinforce':
            # Get K_max from config or use default
            K_max = config.get('K_max', 20480)
            
            # REINFORCE training: Check if data has pre-computed MOCU
            mocu_model = None
            mocu_mean = None
            mocu_std = None
            
            # Check if trajectories have pre-computed MOCU
            has_precomputed_mocu = any('terminal_MOCU' in traj for traj in trajectories)
            
            if has_precomputed_mocu:
                print("[REINFORCE] Using pre-computed MOCU values")
                use_predicted_mocu = False  # Don't use predictor if pre-computed values exist
            else:
                # No pre-computed MOCU - need to use MPNN predictor during training
                print("[REINFORCE] No pre-computed MOCU - will use MPNN predictor during training")
            
            # Auto-enable MPNN predictor if not explicitly disabled
            use_predicted_mocu = args.use_predicted_mocu if args.use_predicted_mocu else True
            
            if not use_predicted_mocu and not has_precomputed_mocu:
                print("\n" + "!"*80)
                print("WARNING: Running REINFORCE without MPNN predictor.")
                print("         Direct MOCU computation is slow - use --use-predicted-mocu for faster training")
                print("!"*80 + "\n")
            if use_predicted_mocu:
                if has_precomputed_mocu:
                    mocu_model = None
                else:
                    try:
                        # Ensure CUDA is in a clean state before loading MPNN predictor
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                        
                        from src.models.predictors.predictor_utils import load_mpnn_predictor
                        import os
                        import re
                        # Path is already imported at module level
                    
                        # Get model name from environment variable, config, or auto-detect from output_dir
                        model_name = os.getenv('MOCU_MODEL_NAME') or config.get('mocu_model_name')
                        
                        # Auto-detect from output_dir if not provided (e.g., models/fast_config/)
                        if not model_name and args.output_dir:
                            output_path = Path(args.output_dir).resolve()
                            parts = output_path.parts
                            # Look for pattern: .../models/{config}/
                            try:
                                models_idx = list(parts).index('models')
                                if models_idx + 1 < len(parts):
                                    config_part = parts[models_idx + 1]  # config name
                                    model_name = config_part
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
                        
                            if mocu_model is not None:
                                print(f"[REINFORCE] Using MPNN predictor: {model_name}")
                            else:
                                print(f"[REINFORCE] Using pre-computed MOCU values")
                    except FileNotFoundError as e:
                        print(f"\n" + "!"*80)
                        print(f"[REINFORCE] ERROR: MPNN predictor not found!")
                        print(f"[REINFORCE] {e}")
                        print(f"\n[REINFORCE] REINFORCE training requires MPNN predictor to avoid segmentation faults.")
                        print(f"[REINFORCE] MPNN predictor is required for efficient training.")
                        print(f"\n[REINFORCE] Solutions:")
                        print(f"  1. Train MPNN predictor first: bash scripts/bash/step2_train_mpnn.sh configs/fast_config.yaml")
                        print(f"  2. Or set MOCU_MODEL_NAME environment variable: export MOCU_MODEL_NAME='{model_name}'")
                        print(f"  3. Or ensure MPNN model exists in: models/{model_name.split('_')[0]}/")
                        print(f"!"*80 + "\n")
                        raise RuntimeError(
                            f"MPNN predictor is REQUIRED for REINFORCE training. "
                            f"Model '{model_name}' not found. "
                            f"Direct MOCU computation is slow. "
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
                use_predicted_mocu=use_predicted_mocu,
                epoch_num=epoch
            )
            train_losses.append(loss)
            
            epoch_pbar.set_postfix({'loss': f'{loss:.4f}', 'reward': f'{reward:.4f}', 
                                   'time': f'{time.time()-start_time:.1f}s'})
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f'{args.name}.pth'
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': {
            'N': N,
            'K': K,  # Save K value for verification
            'hidden_dim': args.hidden_dim,
            'encoding_dim': args.encoding_dim
        },
        'train_config': {
            'method': args.method,
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size
        },
        'train_losses': train_losses  # Save loss history for analysis
    }
    
    if args.method == 'imitation' and train_accs:
        save_dict['train_accs'] = train_accs
    
    torch.save(save_dict, model_path)
    
    print(f"\n✓ Model saved to: {model_path}")
    
    # Print training summary
    if train_losses:
        print(f"\n=== Training Summary ===")
        print(f"Initial loss: {train_losses[0]:.6f}")
        print(f"Final loss: {train_losses[-1]:.6f}")
        print(f"Loss change: {train_losses[-1] - train_losses[0]:.6f}")
        print(f"Best loss: {min(train_losses):.6f} (epoch {train_losses.index(min(train_losses))+1})")
        
        if abs(train_losses[-1] - train_losses[0]) < 0.001:
            print("\n⚠️  WARNING: Loss barely changed - training may not be effective!")
            print("   Consider:")
            print("   - Increasing learning rate (try --lr 0.01)")
            print("   - Using more trajectories (1000+ instead of 100)")
            print("   - Checking if rewards are diverse enough")
    
    # Plot training curves (if matplotlib is available)
    if HAS_MATPLOTLIB:
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
    else:
        print("[INFO] Skipping training curve plot (matplotlib not available)")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

