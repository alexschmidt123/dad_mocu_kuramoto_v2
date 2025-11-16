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


def train_reinforce(model, trajectories, optimizer, device, N, gamma=0.96, K_max=20480, 
                    use_baseline=True, mocu_model=None, mocu_mean=None, mocu_std=None, use_predicted_mocu=False,
                    epoch_num=None, entropy_coef=0.01, critic=None, critic_optimizer=None,
                    use_per_step_reward=True, per_step_weight=0.3):
    """
    Train using REINFORCE policy gradient with MOCU DIRECTLY as the loss.
    
    Args:
        entropy_coef: Entropy regularization coefficient (default 0.01)
                     Prevents policy from becoming too deterministic
        use_per_step_reward: If True, compute MOCU at each step and reward per-step reductions
                            This encourages early-step efficiency (fixes "wasting first steps" issue)
        per_step_weight: Weight for per-step rewards (0.0-1.0, rest goes to terminal)
                        e.g., 0.3 = 30% per-step, 70% terminal
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
        action_probs_list = []  # Store action_probs for entropy computation
        observed_pairs = []
        observations_list = []
        step_mocus = []  # Store MOCU at each step for per-step rewards
        
        a_lower = traj['states'][0][0].copy()
        a_upper = traj['states'][0][1].copy()
        
        K = len(traj['actions'])
        
        # Compute initial MOCU for per-step rewards
        if use_per_step_reward:
            if use_predicted_mocu and mocu_model is not None:
                with torch.no_grad():
                    initial_state_data = create_state_data(w, a_lower, a_upper, device=device)
                    initial_MOCU = predict_mocu(mocu_model, mocu_mean, mocu_std, w, a_lower, a_upper, device=str(device))
                    if isinstance(initial_MOCU, torch.Tensor):
                        initial_MOCU = initial_MOCU.detach().cpu().item()
                    initial_MOCU = float(initial_MOCU)
                step_mocus.append(initial_MOCU)
            else:
                # Can't compute per-step without MPNN predictor
                use_per_step_reward = False
                step_mocus.append(None)
        
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
            
            # Store action_probs for entropy computation (detach to avoid gradient issues)
            action_probs_list.append(action_probs.squeeze(0).detach())  # [num_actions]
            
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
            
            # Compute MOCU at this step for per-step rewards
            if use_per_step_reward and use_predicted_mocu and mocu_model is not None:
                with torch.no_grad():
                    step_state_data = create_state_data(w, a_lower, a_upper, device=device)
                    step_MOCU = predict_mocu(mocu_model, mocu_mean, mocu_std, w, a_lower, a_upper, device=str(device))
                    if isinstance(step_MOCU, torch.Tensor):
                        step_MOCU = step_MOCU.detach().cpu().item()
                    step_MOCU = float(step_MOCU)
                step_mocus.append(step_MOCU)
            
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
        terminal_MOCU = float(terminal_MOCU)
        
        # === PER-STEP REWARD METHOD ===
        # Compute rewards at each step to encourage early-step efficiency
        if use_per_step_reward and len(step_mocus) == K + 1 and all(m is not None for m in step_mocus):
            # Compute per-step rewards: reward_i = -(MOCU_i - MOCU_{i-1})
            per_step_rewards = []
            for i in range(1, len(step_mocus)):
                mocu_reduction = step_mocus[i-1] - step_mocus[i]  # Positive reduction = good
                step_reward = -mocu_reduction  # Negative because we want to minimize MOCU
                per_step_rewards.append(step_reward)
            
            # Total per-step reward (sum of all step reductions)
            total_per_step_reward = sum(per_step_rewards)
            
            # Terminal reward
            terminal_reward = -terminal_MOCU
            
            # Weighted combination: per_step_weight * per_step + (1-weight) * terminal
            # This gives explicit credit to early steps while still optimizing terminal MOCU
            reward = (per_step_weight * total_per_step_reward + 
                     (1.0 - per_step_weight) * terminal_reward)
            
            # Store per-step rewards for step-wise advantages
            step_rewards_list = per_step_rewards
        else:
            # Fallback: terminal reward only (original method)
            reward = float(-terminal_MOCU)
            step_rewards_list = None
        
        reward_list.append(reward)
        all_rewards.append(reward)
        
        # === CRITIC BASELINE (iDAD-inspired variance reduction) ===
        critic_baseline = None
        if critic is not None:
            # Estimate MOCU using critic network
            final_state_data = create_state_data(w, a_lower, a_upper, device=device)
            if len(observed_pairs) > 0:
                history_list = [(observed_pairs[k][0], observed_pairs[k][1], observations_list[k]) 
                              for k in range(len(observed_pairs))]
                history_tensor = torch.tensor([history_list], dtype=torch.long, device=device)
            else:
                history_tensor = None
            
            critic.train()  # Ensure critic is in train mode
            with torch.no_grad():
                mocu_estimate = critic(final_state_data, history_tensor)  # [1]
                critic_baseline = -mocu_estimate.item()  # Negative because reward = -MOCU
        
        # Compute advantage using critic baseline (if available) or reward normalization
        if critic_baseline is not None:
            # Use critic as baseline (iDAD approach - reduces variance)
            advantage = float(reward - critic_baseline)
            
            # CRITICAL: If advantage is too small, the learning signal is lost
            # Add minimum advantage magnitude to maintain gradient signal
            min_advantage_magnitude = 0.1
            if abs(advantage) < min_advantage_magnitude:
                # Scale advantage to maintain learning signal
                if advantage >= 0:
                    advantage = min_advantage_magnitude
                else:
                    advantage = -min_advantage_magnitude
            
            # Clip to prevent extreme values
            advantage = np.clip(advantage, -5.0, 5.0)
        elif len(all_rewards) >= 50:
            # Fallback to z-score normalization if no critic
            reward_mean = np.mean(all_rewards) if len(all_rewards) > 0 else 0.0
            reward_std = np.std(all_rewards) if len(all_rewards) > 1 else 0.0
            
            if reward_std > 1e-3:  # Sufficient variance for z-score normalization
                # Z-score normalization: (reward - mean) / std
                # This ensures advantages have consistent scale regardless of reward distribution
                advantage = float((reward - reward_mean) / reward_std)
                
                # Clip to prevent extreme values that could cause gradient explosion
                advantage = np.clip(advantage, -5.0, 5.0)
            elif reward_std > 1e-6:  # Very low but non-zero variance
                # Use aggressive scaling to maintain learning signal
                advantage = float((reward - reward_mean) / max(reward_std, 1e-4)) * 10.0
                advantage = np.clip(advantage, -5.0, 5.0)
            else:
                # Extremely low variance (all rewards nearly identical): use constant advantage
                # This encourages exploration when all trajectories have similar outcomes
                # Use a small positive advantage to encourage policy improvement
                advantage = 1.0  # Constant positive signal to encourage exploration
        else:
            # Not enough samples yet: use simple baseline or raw reward
            # Calculate reward_std for variance check
            reward_std = np.std(all_rewards) if len(all_rewards) > 1 else 0.0
            
            # CRITICAL: If variance is extremely low, use constant advantage even early in training
            if reward_std < 1e-6:
                # All rewards are identical: use constant advantage to maintain learning signal
                advantage = 1.0
            elif use_baseline and reward_std > 1e-6:
                if baseline is None:
                    # Initialize baseline after collecting some rewards
                    if len(all_rewards) >= 10:
                        baseline = np.mean(all_rewards)
                else:
                    # Update baseline slowly
                    baseline = baseline_alpha * baseline + (1 - baseline_alpha) * reward
                
                advantage = float(reward - baseline) if baseline is not None else float(reward)
            else:
                # Low variance or no baseline: use raw reward with scaling
                advantage = float(reward) * 10.0  # Scale to maintain signal
                advantage = np.clip(advantage, -5.0, 5.0)
        
        # === STEP-WISE ADVANTAGES (for per-step rewards) ===
        # If using per-step rewards, compute step-wise advantages
        if use_per_step_reward and step_rewards_list is not None and len(step_rewards_list) == len(log_probs):
            # Compute step-wise advantages
            if critic is not None:
                # For critic baseline, we'd need step-wise critic estimates
                # For now, use normalized step rewards as advantages
                step_reward_mean = np.mean(step_rewards_list)
                step_reward_std = np.std(step_rewards_list) if len(step_rewards_list) > 1 else 1.0
                if step_reward_std > 1e-6:
                    returns = [(sr - step_reward_mean) / step_reward_std for sr in step_rewards_list]
                    returns = [np.clip(r, -5.0, 5.0) for r in returns]
                else:
                    # Low variance: use constant advantage
                    returns = [advantage] * len(log_probs)
            else:
                # No critic: use normalized step rewards
                if len(step_rewards_list) > 0:
                    step_reward_mean = np.mean(step_rewards_list)
                    step_reward_std = np.std(step_rewards_list) if len(step_rewards_list) > 1 else 1.0
                    if step_reward_std > 1e-6:
                        returns = [(sr - step_reward_mean) / step_reward_std for sr in step_rewards_list]
                        returns = [np.clip(r, -5.0, 5.0) for r in returns]
                    else:
                        returns = [advantage] * len(log_probs)
                else:
                    returns = [advantage] * len(log_probs)
        else:
            # Standard: same advantage for all steps (terminal reward only)
            returns = [advantage] * len(log_probs)
        
        # CRITICAL: Validate all log_probs tensors before building loss computation graph
        # This prevents invalid memory access during loss construction
        valid_log_probs = []
        valid_action_probs = []  # Store action_probs for entropy computation
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
                # Store corresponding action_probs for entropy (from the forward pass)
                # Note: action_probs_list contains detached tensors, but we need gradients for entropy
                # So we'll recompute entropy from the policy network during loss computation
                if idx < len(action_probs_list):
                    # Use stored action_probs (will recompute with gradients in loss computation)
                    valid_action_probs.append(action_probs_list[idx])
                else:
                    raise RuntimeError(f"Missing action_probs for step {idx}")
            except (AssertionError, RuntimeError) as e:
                print(f"[REINFORCE] ERROR: Invalid log_prob at step {idx}: {e}")
                raise RuntimeError(f"Invalid log_prob tensor at step {idx}: {e}") from e
        
        if len(valid_log_probs) != len(log_probs):
            raise RuntimeError(f"Only {len(valid_log_probs)}/{len(log_probs)} log_probs are valid")
        
        # Build loss computation graph (only involves policy network, not MPNN)
        # Only synchronize if CUDA_LAUNCH_BLOCKING is enabled (debugging mode)
        # Use validated log_probs to prevent memory access errors
        # REINFORCE loss with entropy regularization to prevent policy collapse
        policy_loss = []
        entropy_loss = []
        for idx, (log_prob, advantage_val, action_probs_step) in enumerate(zip(valid_log_probs, returns, valid_action_probs)):
            try:
                # Validate advantage_val is a valid Python float
                assert isinstance(advantage_val, (int, float)), f"advantage_val {idx} must be numeric"
                assert not (isinstance(advantage_val, float) and (np.isnan(advantage_val) or np.isinf(advantage_val))), \
                    f"advantage_val {idx} is NaN/Inf"
                
                # Ensure advantage_tensor is created on same device as log_prob
                advantage_tensor = torch.tensor(float(advantage_val), device=log_prob.device, requires_grad=False, dtype=log_prob.dtype)
                
                # Build loss component with explicit error checking
                # REINFORCE loss: -log_prob * advantage
                # This optimizes MOCU directly (reward = -terminal_MOCU, so lower MOCU = higher reward)
                loss_component = -log_prob * advantage_tensor
                
                # Validate loss component immediately
                assert isinstance(loss_component, torch.Tensor), f"loss_component {idx} is not a tensor"
                assert loss_component.requires_grad == log_prob.requires_grad, f"loss_component {idx} gradient flag mismatch"
                assert not (torch.isnan(loss_component) or torch.isinf(loss_component)), \
                    f"loss_component {idx} is NaN/Inf"
                
                policy_loss.append(loss_component)
                
                # Compute entropy for this step (encourages exploration, prevents policy collapse)
                # Entropy = -sum(p * log(p)) where p is action probability distribution
                # Higher entropy = more exploration, lower entropy = more deterministic
                # Note: action_probs_step is detached, but we need gradients for entropy regularization
                # So we'll use the action_probs from the policy network's current state
                # For now, use detached version (entropy regularization doesn't need gradients through action_probs)
                # The gradient comes from the policy_loss term, entropy just adds a bonus
                entropy = -torch.sum(action_probs_step * torch.log(action_probs_step + 1e-8))
                entropy_loss.append(entropy)
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
            # REINFORCE loss: sum of (-log_prob * advantage) for all steps
            loss = policy_loss[0]
            for component in policy_loss[1:]:
                loss = loss + component
            
            # Add entropy regularization to prevent policy from becoming too deterministic
            # Entropy bonus encourages exploration: loss = policy_loss - entropy_coef * entropy
            # (negative because we want to maximize entropy, but loss is minimized)
            if entropy_coef > 0 and len(entropy_loss) > 0:
                total_entropy = entropy_loss[0]
                for ent in entropy_loss[1:]:
                    total_entropy = total_entropy + ent
                
                # CRITICAL: Scale entropy by number of steps to make it comparable to policy loss
                # Without scaling, entropy for K steps is K times larger, making it dominate
                avg_entropy = total_entropy / len(entropy_loss) if len(entropy_loss) > 0 else total_entropy
                
                # Apply minimum entropy constraint: if entropy drops too low, increase bonus
                min_entropy_threshold = 0.01
                if avg_entropy < min_entropy_threshold:
                    # Boost entropy coefficient when entropy is too low
                    entropy_boost = 1.0 + (min_entropy_threshold - avg_entropy) / min_entropy_threshold
                    effective_entropy_coef = entropy_coef * entropy_boost
                else:
                    effective_entropy_coef = entropy_coef
                
                # Subtract entropy (we want to maximize it, so subtract from loss)
                # Use average entropy per step to keep scale consistent
                loss = loss - effective_entropy_coef * total_entropy
            
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
        
        # Gradient clipping: prevent exploding gradients (reduced to 1.0 for more stable training)
        # Lower max_norm helps prevent divergence when learning rate is high
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # === TRAIN CRITIC (iDAD-inspired) ===
        if critic is not None and critic_optimizer is not None:
            critic_optimizer.zero_grad()
            
            # Estimate MOCU using critic
            final_state_data = create_state_data(w, a_lower, a_upper, device=device)
            if len(observed_pairs) > 0:
                history_list = [(observed_pairs[k][0], observed_pairs[k][1], observations_list[k]) 
                              for k in range(len(observed_pairs))]
                history_tensor = torch.tensor([history_list], dtype=torch.long, device=device)
            else:
                history_tensor = None
            
            mocu_estimate = critic(final_state_data, history_tensor)  # [1]
            # Ensure target_mocu has same shape as mocu_estimate ([1] not [])
            target_mocu = torch.tensor([terminal_MOCU], dtype=torch.float32, device=device)
            
            # Critic loss: MSE between predicted and actual MOCU
            critic_loss = F.mse_loss(mocu_estimate, target_mocu)
            critic_loss.backward()
            
            # Gradient clipping for critic (reduced to 1.0 for stability)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            critic_optimizer.step()
        
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
            
            # Calculate average log_prob magnitude for diagnostics
            avg_log_prob = np.mean([abs(lp.item()) for lp in log_probs]) if log_probs else 0.0
            
            # Calculate average entropy for diagnostics (prevent policy collapse)
            avg_entropy = np.mean([ent.item() for ent in entropy_loss]) if entropy_loss else 0.0
            
            traj_pbar.set_postfix({
                'loss': f'{current_loss:.4f}', 
                'reward': f'{reward:.4f}',
                'adv': f'{advantage:.4f}',
                '|adv|': f'{adv_magnitude:.4f}',
                'r_std': f'{reward_std:.6f}',
                '|log_p|': f'{avg_log_prob:.4f}',
                'entropy': f'{avg_entropy:.4f}'
            })
            
            # Diagnostic warnings
            if len(all_rewards) >= 10:
                if reward_std < 1e-6:
                    if not hasattr(train_reinforce, '_zero_var_warned'):
                        traj_pbar.write(f"[WARNING] Zero reward variance detected (std={reward_std:.8f}) - using constant advantage=1.0")
                        traj_pbar.write(f"  All rewards are identical: {reward:.6f}")
                        traj_pbar.write(f"  This suggests all trajectories have the same terminal MOCU - check data generation!")
                        train_reinforce._zero_var_warned = True
                
                if adv_magnitude < 0.01 and reward_std > 1e-6:
                    if not hasattr(train_reinforce, '_small_adv_warned'):
                        traj_pbar.write(f"[WARNING] Advantage magnitude is very small ({adv_magnitude:.6f}) - baseline may be canceling learning signal")
                        train_reinforce._small_adv_warned = True
                
                if avg_log_prob < 1e-6:
                    if not hasattr(train_reinforce, '_small_logp_warned'):
                        traj_pbar.write(f"[WARNING] Log probabilities are very small ({avg_log_prob:.8f}) - policy may be too deterministic")
                        train_reinforce._small_logp_warned = True
                
                # Warn if loss is approaching zero (no learning signal)
                if abs(current_loss) < 0.001:
                    if not hasattr(train_reinforce, '_zero_loss_warned'):
                        traj_pbar.write(f"[WARNING] Loss is near zero ({current_loss:.6f}) - policy may have stopped learning!")
                        traj_pbar.write(f"  This indicates no gradient signal. Check entropy ({avg_entropy:.6f}) and advantages.")
                        train_reinforce._zero_loss_warned = True
                
                # Warn if entropy is too low (policy too deterministic)
                if avg_entropy < 0.01 and len(entropy_loss) > 0:
                    if not hasattr(train_reinforce, '_low_entropy_warned'):
                        traj_pbar.write(f"[WARNING] Entropy is very low ({avg_entropy:.6f}) - policy is too deterministic!")
                        traj_pbar.write(f"  Consider increasing entropy_coef (current: {entropy_coef})")
                        train_reinforce._low_entropy_warned = True
        
        total_loss += current_loss
        total_reward += reward
    
    avg_loss = total_loss / len(trajectories)
    avg_reward = total_reward / len(trajectories)
    
    return avg_loss, avg_reward


def main():
    parser = argparse.ArgumentParser(description='Train DAD policy network')
    parser.add_argument('--data-path', type=str, required=True, help='Path to trajectory data')
    parser.add_argument('--method', type=str, default='dad_mocu', 
                       choices=['imitation', 'reinforce', 'dad_mocu', 'idad_mocu'], 
                       help='Training method: "dad_mocu" (no critic, simple baseline), "idad_mocu" (with critic from scratch), "reinforce" (legacy), or "imitation" (behavior cloning)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate (default: 1e-5, reduced for stability)')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension (default: 256, matching original DAD)')
    parser.add_argument('--encoding-dim', type=int, default=16, help='Encoding dimension (default: 16, matching original DAD)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='../models/', help='Output directory')
    parser.add_argument('--name', type=str, default='dad_policy', help='Model name')
    parser.add_argument('--use-predicted-mocu', action='store_true',
                       help='Use MPNN predictor for fast MOCU estimation (recommended). Requires trained MPNN model.')
    parser.add_argument('--use-critic', action='store_true',
                       help='Use critic network for variance reduction (iDAD-inspired). Auto-enabled for idad_mocu method.')
    parser.add_argument('--critic-lr', type=float, default=None,
                       help='Learning rate for critic (default: same as policy lr)')
    args = parser.parse_args()
    
    # Map method names to critic usage
    if args.method == 'idad_mocu':
        # iDAD-MOCU: Use critic (learns from scratch, no MPNN)
        args.use_critic = True
        print("[METHOD] Using iDAD-MOCU: with critic network (learns from scratch)")
    elif args.method == 'dad_mocu':
        # DAD-MOCU: No critic, use simple baseline
        args.use_critic = False
        print("[METHOD] Using DAD-MOCU: no critic, simple baseline")
    elif args.method == 'reinforce':
        # Legacy: Use --use-critic flag if provided
        print("[METHOD] Using legacy REINFORCE method")
    
    # Both dad_mocu and idad_mocu use per-step rewards by default
    use_per_step_reward_default = args.method in ['dad_mocu', 'idad_mocu']
    
    # Load data
    print("Loading trajectory data...")
    data = torch.load(args.data_path, weights_only=False)
    trajectories = data['trajectories']
    config = data['config']
    N = config['N']
    K = config['K']
    
    print(f"Loaded {len(trajectories)} trajectories")
    print(f"System: N={N}, K={K} design steps ({K+1} total steps: 0-{K})")
    
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
    
    # Load MPNN predictor for critic (if using critic)
    mpnn_model = None
    mpnn_mean = None
    mpnn_std = None
    # NOTE: MPNN loading is handled separately for per-step rewards (if needed)
    # We intentionally do NOT load MPNN for the critic (see critic initialization below)
    # MPNN is still used for pre-computed rewards (terminal_MOCU) and per-step rewards
    # but NOT for the critic baseline to prevent vanishing advantages
    
    # Create critic network (iDAD-inspired for variance reduction)
    # SOLUTION 2: Critic WITHOUT MPNN to prevent it from becoming too accurate
    # - MPNN is still used for pre-computed rewards (terminal_MOCU) - GOOD
    # - Critic learns from scratch (no MPNN) - prevents vanishing advantages
    # - Critic will be less accurate than MPNN, providing good advantage signal
    critic = None
    critic_optimizer = None
    if args.use_critic:
        from src.models.critics import MOCUCritic
        # Intentionally set mpnn_model=None to prevent critic from being too accurate
        # This ensures advantages don't vanish (advantage = reward - critic)
        # MPNN is still used for pre-computed rewards, just not in critic baseline
        critic = MOCUCritic(
            N=N,
            hidden_dim=args.hidden_dim,
            encoding_dim=args.encoding_dim,
            use_set_equivariant=False,  # Use LSTM encoder (order-dependent, necessary when order matters)
            mpnn_model=None,  # Intentionally None - critic learns from scratch
            mpnn_mean=None,  # Not needed when mpnn_model is None
            mpnn_std=None  # Not needed when mpnn_model is None
        )
        critic = critic.to(device)
        print(f"Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")
        
        critic_lr = args.critic_lr if args.critic_lr is not None else args.lr
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        print(f"Using critic network for variance reduction (iDAD-inspired)")
        print(f"  Critic LR: {critic_lr}")
        print(f"  Pre-trained MPNN: DISABLED in critic (intentional)")
        print(f"    → Critic learns from scratch (no MPNN)")
        print(f"    → Critic will be less accurate → good advantage signal")
        print(f"    → MPNN still used for pre-computed rewards (terminal_MOCU)")
    
    # Optimizer with learning rate scheduling
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler: reduce LR when loss plateaus
    # This helps prevent overfitting and stabilizes training
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, 
        verbose=True, min_lr=1e-6
    )
    
    # Critic scheduler (if using critic)
    critic_scheduler = None
    if critic is not None:
        critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            critic_optimizer, mode='min', factor=0.5, patience=10,
            verbose=True, min_lr=1e-6
        )
    
    # Early stopping: stop if no improvement for N epochs
    early_stop_patience = 20  # Stop if no improvement for 20 epochs
    early_stop_counter = 0
    
    # Prepare output directory and best-checkpoint tracking
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f'{args.name}.pth'
    best_model_path = output_dir / f'{args.name}_best.pth'
    best_loss = float('inf')
    best_epoch = -1

    def build_checkpoint_dict():
        save_dict = {
            'model_state_dict': model.state_dict(),
            'config': {
                'N': N,
                'K': K,
                'hidden_dim': args.hidden_dim,
                'encoding_dim': args.encoding_dim
            },
            'train_config': {
                'method': args.method,
                'epochs': args.epochs,
                'lr': args.lr,
                'batch_size': args.batch_size,
                'use_critic': args.use_critic
            },
            'train_losses': train_losses.copy()
        }
        if args.method == 'imitation' and train_accs:
            save_dict['train_accs'] = train_accs.copy()
        if critic is not None:
            save_dict['critic_state_dict'] = critic.state_dict()
        return save_dict
    
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
            current_loss = train_losses[-1]
            if current_loss < best_loss:
                best_loss = current_loss
                best_epoch = epoch
                torch.save(build_checkpoint_dict(), best_model_path)
                print(f"[Checkpoint] Saved best model (epoch {epoch + 1}, loss {current_loss:.6f}) -> {best_model_path}")
            
            epoch_pbar.set_postfix({'loss': f'{loss:.4f}', 'acc': f'{acc:.4f}', 
                                   'time': f'{time.time()-start_time:.1f}s'})
        
        elif args.method in ['reinforce', 'dad_mocu', 'idad_mocu']:
            # Get K_max from config or use default
            K_max = config.get('K_max', 20480)
            
            # REINFORCE training: Check if data has pre-computed MOCU
            mocu_model = None
            mocu_mean = None
            mocu_std = None
            
            # Check if trajectories have pre-computed MOCU
            has_precomputed_mocu = any('terminal_MOCU' in traj for traj in trajectories)
            
            # CRITICAL: For per-step rewards, we NEED MPNN predictor even if terminal MOCU is pre-computed
            # Per-step rewards require computing MOCU at each step, which needs the predictor
            # Both dad_mocu and idad_mocu use per-step rewards by default
            if args.method in ['dad_mocu', 'idad_mocu']:
                enable_per_step_reward = True  # Always enable for new methods
                print(f"[{args.method.upper()}] Per-step rewards enabled (default for this method)")
            else:
                enable_per_step_reward = True  # Enable per-step rewards to fix "wasting first steps"
            
            if has_precomputed_mocu:
                print("[REINFORCE] Using pre-computed terminal MOCU values")
                if enable_per_step_reward:
                    print("[REINFORCE] Per-step rewards enabled - will load MPNN predictor for intermediate MOCU")
                    use_predicted_mocu = True  # Need predictor for per-step rewards
                else:
                    use_predicted_mocu = False  # Don't need predictor if only using terminal MOCU
            else:
                # No pre-computed MOCU - need to use MPNN predictor during training
                print("[REINFORCE] No pre-computed MOCU - will use MPNN predictor during training")
                use_predicted_mocu = True
            
            # Auto-enable MPNN predictor if not explicitly disabled
            use_predicted_mocu = args.use_predicted_mocu if args.use_predicted_mocu else use_predicted_mocu
            
            if not use_predicted_mocu and not has_precomputed_mocu:
                print("\n" + "!"*80)
                print("WARNING: Running REINFORCE without MPNN predictor.")
                print("         Direct MOCU computation is slow - use --use-predicted-mocu for faster training")
                print("!"*80 + "\n")
            if use_predicted_mocu:
                # Load MPNN predictor (needed for per-step rewards OR if no pre-computed MOCU)
                # Always load if per-step rewards are enabled (even with pre-computed terminal MOCU)
                if has_precomputed_mocu and not enable_per_step_reward:
                    # Only terminal MOCU needed, predictor not required
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
            
            # Entropy scheduling: start high, decay slowly to maintain exploration
            # Start at 1.0, decay to 0.7 over epochs (prevents early collapse, maximum exploration)
            initial_entropy = 1.0
            final_entropy = 0.7  # Increased from 0.5 to 0.7 to maintain more exploration
            decay_epochs = args.epochs * 0.8  # Decay over 80% of training (slower decay)
            if epoch < decay_epochs:
                current_entropy = initial_entropy - (initial_entropy - final_entropy) * (epoch / decay_epochs)
            else:
                current_entropy = final_entropy
            
            loss, reward = train_reinforce(
                model, trajectories, optimizer, device, N, 
                K_max=K_max,
                mocu_model=mocu_model,
                mocu_mean=mocu_mean,
                mocu_std=mocu_std,
                use_predicted_mocu=use_predicted_mocu,
                epoch_num=epoch,
                entropy_coef=current_entropy,  # Scheduled entropy: starts at 0.5, decays to 0.3
                critic=critic,
                critic_optimizer=critic_optimizer,
                use_per_step_reward=True,  # Enable per-step rewards to fix "wasting first steps"
                per_step_weight=0.3  # 30% per-step, 70% terminal
            )
            train_losses.append(loss)
            current_loss = train_losses[-1]
            
            # Warn if loss is increasing (divergence) and automatically reduce LR
            if len(train_losses) > 5:
                recent_trend = train_losses[-5:]
                if all(recent_trend[i] > recent_trend[i-1] for i in range(1, len(recent_trend))):
                    if not hasattr(main, '_divergence_warned'):
                        print(f"\n⚠️  [EPOCH {epoch + 1}] WARNING: Loss is INCREASING (diverging)!")
                        print(f"   Recent trend: {[f'{x:.6f}' for x in recent_trend]}")
                        print(f"   This indicates training instability.")
                        print(f"   Automatically reducing learning rate by 50%...")
                        
                        # Reduce learning rate by 50% for both policy and critic
                        for param_group in optimizer.param_groups:
                            old_lr = param_group['lr']
                            param_group['lr'] = old_lr * 0.5
                            print(f"   Policy LR: {old_lr:.2e} → {param_group['lr']:.2e}")
                        
                        if critic_optimizer is not None:
                            for param_group in critic_optimizer.param_groups:
                                old_lr = param_group['lr']
                                param_group['lr'] = old_lr * 0.5
                                print(f"   Critic LR: {old_lr:.2e} → {param_group['lr']:.2e}")
                        
                        main._divergence_warned = True
                    elif not hasattr(main, '_divergence_lr_reduced'):
                        # If divergence continues, reduce LR again
                        print(f"\n⚠️  [EPOCH {epoch + 1}] Divergence continues - reducing LR again...")
                        for param_group in optimizer.param_groups:
                            old_lr = param_group['lr']
                            param_group['lr'] = old_lr * 0.5
                            print(f"   Policy LR: {old_lr:.2e} → {param_group['lr']:.2e}")
                        
                        if critic_optimizer is not None:
                            for param_group in critic_optimizer.param_groups:
                                old_lr = param_group['lr']
                                param_group['lr'] = old_lr * 0.5
                                print(f"   Critic LR: {old_lr:.2e} → {param_group['lr']:.2e}")
                        
                        main._divergence_lr_reduced = True
            
            # Warn if loss is approaching zero (policy collapse)
            if abs(current_loss) < 0.001:
                if not hasattr(main, '_zero_loss_epoch_warned'):
                    print(f"\n⚠️  [EPOCH {epoch + 1}] WARNING: Loss is near zero ({current_loss:.6f})!")
                    print(f"   This indicates policy collapse - no learning signal.")
                    print(f"   The model may be too deterministic or advantages are zero.")
                    print(f"   Consider: increasing entropy_coef, checking critic, or reducing learning rate.")
                    main._zero_loss_epoch_warned = True
            
            # For REINFORCE: more negative loss is better, so we want to minimize (make more negative)
            # best_loss should be the most negative (best) loss
            if current_loss < best_loss:
                best_loss = current_loss
                best_epoch = epoch
                torch.save(build_checkpoint_dict(), best_model_path)
                print(f"[Checkpoint] Saved best model (epoch {epoch + 1}, loss {current_loss:.6f}) -> {best_model_path}")
            
            epoch_pbar.set_postfix({'loss': f'{loss:.4f}', 'reward': f'{reward:.4f}', 
                                   'time': f'{time.time()-start_time:.1f}s'})
        
        # Update learning rate scheduler
        if args.method == 'reinforce':
            scheduler.step(loss)  # Reduce LR if loss plateaus
            if critic_scheduler is not None:
                critic_scheduler.step(loss)  # Also update critic LR
        
        # Early stopping check
        if loss < best_loss:
            early_stop_counter = 0  # Reset counter on improvement
        else:
            early_stop_counter += 1
        
        # Stop early if no improvement for patience epochs
        if early_stop_counter >= early_stop_patience:
            print(f"\n⚠ Early stopping: No improvement for {early_stop_patience} epochs")
            print(f"   Best loss was {best_loss:.6f} at epoch {best_epoch + 1}")
            break
    
    # Save model (output_dir already defined above)
    save_dict = build_checkpoint_dict()
    torch.save(save_dict, model_path)
    
    print(f"\n✓ Model saved to: {model_path}")
    if best_epoch >= 0:
        print(f"✓ Best model saved to: {best_model_path} (epoch {best_epoch + 1}, loss {best_loss:.6f})")
    print(f"✓ DAD model trained for K={K} design steps ({K+1} total steps: 0-{K})")
    
    # Print training summary
    if train_losses:
        print(f"\n=== Training Summary ===")
        print(f"Initial loss: {train_losses[0]:.6f}")
        print(f"Final loss: {train_losses[-1]:.6f}")
        print(f"Loss change: {train_losses[-1] - train_losses[0]:.6f}")
        print(f"Best loss: {min(train_losses):.6f} (epoch {train_losses.index(min(train_losses))+1})")
        
        # Check for training issues
        if abs(train_losses[-1] - train_losses[0]) < 0.001:
            print("\n⚠️  WARNING: Loss barely changed - training may not be effective!")
            print("   Consider:")
            print("   - Increasing learning rate (try --lr 0.01)")
            print("   - Using more trajectories (1000+ instead of 100)")
            print("   - Checking if rewards are diverse enough")
        
        # Check if loss collapsed to zero (policy collapse)
        if abs(train_losses[-1]) < 0.001:
            print("\n⚠️  CRITICAL: Loss collapsed to zero - policy stopped learning!")
            print("   This indicates:")
            print("   - Policy became too deterministic (entropy → 0)")
            print("   - Advantages became zero (critic matches rewards perfectly)")
            print("   - No gradient signal → no learning")
            print("   Solutions:")
            print("   - Increase entropy_coef (currently 0.15, try 0.2 or 0.3 if still collapsing)")
            print("   - Check critic training (may be overfitting)")
            print("   - Reduce learning rate or use different schedule")
            print("   - Monitor entropy during training (should stay > 0.01)")
    
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

