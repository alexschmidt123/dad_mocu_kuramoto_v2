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


def train_reinforce(model, trajectories, optimizer, device, N, gamma=0.99):
    """
    Train using REINFORCE policy gradient.
    
    Reward: Negative terminal MOCU (want to minimize MOCU)
    """
    model.train()
    total_loss = 0
    total_reward = 0
    
    for traj in trajectories:
        optimizer.zero_grad()
        
        w = traj['w']
        log_probs = []
        
        # Run through trajectory and collect log probabilities
        for step in range(len(traj['actions'])):
            # Create state
            a_lower, a_upper = traj['states'][step]
            state_data = create_state_data(w, a_lower, a_upper, device=device)
            
            # History
            if step == 0:
                history = None
            else:
                history = [[traj['actions'][k][0], 
                           traj['actions'][k][1], 
                           traj['observations'][k]] 
                          for k in range(step)]
                history = torch.tensor([history], dtype=torch.long, device=device)
            
            # Available mask
            available_mask = torch.tensor(
                [traj['available_masks'][step]],
                dtype=torch.float32,
                device=device
            )
            
            # Forward pass
            action_logits, action_probs = model(state_data, history, available_mask)
            
            # Get log prob of expert action
            action_i, action_j = traj['actions'][step]
            action_idx = model.pair_to_idx(action_i, action_j)
            log_prob = F.log_softmax(action_logits, dim=-1)[0, action_idx]
            
            log_probs.append(log_prob)
        
        # Reward: negative terminal MOCU (want to maximize this = minimize MOCU)
        terminal_MOCU = traj['terminal_MOCU']
        reward = -terminal_MOCU
        
        # Discounted rewards (optional, for multi-step credit assignment)
        # For now, use same reward for all steps
        returns = [reward] * len(log_probs)
        
        # Policy gradient loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        loss = torch.stack(policy_loss).sum()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_reward += reward
    
    avg_loss = total_loss / len(trajectories)
    avg_reward = total_reward / len(trajectories)
    
    return avg_loss, avg_reward


def main():
    parser = argparse.ArgumentParser(description='Train DAD policy network')
    parser.add_argument('--data-path', type=str, required=True, help='Path to trajectory data')
    parser.add_argument('--method', type=str, default='imitation', 
                       choices=['imitation', 'reinforce'], help='Training method')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--encoding-dim', type=int, default=32, help='Encoding dimension')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='../models/', help='Output directory')
    parser.add_argument('--name', type=str, default='dad_policy', help='Model name')
    args = parser.parse_args()
    
    # Load data
    print("Loading trajectory data...")
    data = torch.load(args.data_path)
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
            loss, reward = train_reinforce(model, trajectories, optimizer, device, N)
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

