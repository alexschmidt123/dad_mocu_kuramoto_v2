"""
DAD-MOCU Method (Deep Adaptive Design for MOCU)

Uses a trained policy network to sequentially select experiments
that minimize terminal MOCU. The policy is learned via REINFORCE 
(reinforcement learning) that directly optimizes terminal MOCU as the loss.

Training: REINFORCE policy gradient with terminal MOCU as reward signal.
This directly optimizes the true objective (minimize terminal MOCU) rather
than mimicking a suboptimal expert policy.

In the paper: "DAD" method (proposed)
"""

import time
import numpy as np
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.methods.base import OEDMethod
from src.models.policy_networks import DADPolicyNetwork, create_state_data
from src.core.mocu_cuda import MOCU


class DAD_MOCU_Method(OEDMethod):
    """
    Deep Adaptive Design method with MOCU objective.
    
    Uses a learned policy network to make sequential experimental selections.
    The policy is trained to minimize terminal MOCU over K steps.
    
    This is analogous to the original DAD paper (Foster et al. 2021) but with
    MOCU as the objective instead of Expected Information Gain (EIG).
    """
    
    def __init__(self, N, K_max, deltaT, MReal, TReal, it_idx, 
                 policy_model_path=None, gpu_id=0):
        """
        Args:
            N: Number of oscillators
            K_max: Number of Monte Carlo samples for MOCU
            deltaT: Time step
            MReal: Number of time steps
            TReal: Time horizon
            it_idx: Number of MOCU averaging iterations
            policy_model_path: Path to trained policy checkpoint (.pth)
            gpu_id: GPU device ID
        """
        super().__init__(N, K_max, deltaT, MReal, TReal, it_idx)
        
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.policy_net = None
        
        # Load policy network
        self._load_policy(policy_model_path)
        
        print(f"[DAD-MOCU] Initialized with policy on {self.device}")
    
    def _load_policy(self, policy_model_path):
        """Load trained DAD policy network."""
        if policy_model_path is None:
            # Try default location
            policy_model_path = PROJECT_ROOT / 'models' / f'dad_policy_N{self.N}.pth'
            
            if not policy_model_path.exists():
                raise FileNotFoundError(
                    f"Policy model not found at {policy_model_path}\n"
                    f"Please train a DAD policy first using:\n"
                    f"  python scripts/train_dad_policy.py --data-path <data> --name dad_policy_N{self.N}"
                )
        
        print(f"[DAD-MOCU] Loading policy from: {policy_model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(policy_model_path, map_location=self.device)
        model_config = checkpoint['config']
        
        # Create model
        self.policy_net = DADPolicyNetwork(
            N=model_config['N'],
            hidden_dim=model_config.get('hidden_dim', 64),
            encoding_dim=model_config.get('encoding_dim', 32),
            num_message_passing=model_config.get('num_message_passing', 3)
        )
        
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.policy_net.to(self.device)
        self.policy_net.eval()
        
        print(f"[DAD-MOCU] Policy loaded successfully")
    
    def select_experiment(self, w, a_lower_bounds, a_upper_bounds, criticalK, isSynchronized, history):
        """
        Select next experiment using learned DAD policy.
        
        The policy network takes the current state (w, bounds, history)
        and outputs a probability distribution over available experiments.
        We select the experiment with highest probability (greedy/deterministic).
        
        Args:
            w: Natural frequencies [N]
            a_lower_bounds: Current lower bounds [N, N]
            a_upper_bounds: Current upper bounds [N, N]
            criticalK: Critical coupling strengths (not used by policy)
            isSynchronized: Synchronization status (not used by policy)
            history: List of ((i, j), observation) tuples from run_episode
        
        Returns:
            (i, j): Selected experiment pair
        """
        # Convert history format: from [((i,j), obs), ...] to [(i, j, obs), ...]
        history_list = []
        observed_pairs = set()
        if history:
            for pair_obs in history:
                if isinstance(pair_obs, tuple) and len(pair_obs) == 2:
                    (i, j), obs = pair_obs
                    history_list.append((i, j, int(obs)))
                    observed_pairs.add((i, j))
        
        # Create state data for policy network
        state_data = create_state_data(w, a_lower_bounds, a_upper_bounds, device=self.device)
        
        # Get available pairs (not yet observed)
        available_pairs = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if (i, j) not in observed_pairs:
                    available_pairs.append((i, j))
        
        if not available_pairs:
            print("[DAD-MOCU] Warning: No available experiments left!")
            return -1, -1
        
        # Create available actions mask (1 = available, 0 = observed)
        num_actions = self.N * (self.N - 1) // 2
        available_mask = np.ones(num_actions, dtype=np.float32)
        for (i_obs, j_obs) in observed_pairs:
            action_idx = self.policy_net.pair_to_idx(i_obs, j_obs)
            available_mask[action_idx] = 0.0
        
        available_mask_tensor = torch.tensor([available_mask], dtype=torch.float32, device=self.device)
        
        # Convert history to tensor format expected by policy network
        if len(history_list) == 0:
            history_tensor = None
        else:
            history_tensor = torch.tensor([history_list], dtype=torch.long, device=self.device)
        
        # Get policy action probabilities (greedy/deterministic)
        with torch.no_grad():
            self.policy_net.eval()
            action_logits, action_probs = self.policy_net(state_data, history_tensor, available_mask_tensor)
        
        # Select action with highest probability (deterministic)
        action_probs = action_probs.squeeze(0)  # [num_actions]
        action_idx = torch.argmax(action_probs).item()
        
        # Convert action index to (i, j) pair using policy network method
        i, j = self.policy_net.idx_to_pair(action_idx)
        
        return i, j

