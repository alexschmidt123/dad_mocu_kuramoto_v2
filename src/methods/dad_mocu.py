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
from src.models.predictors.utils import get_edge_attr_from_bounds, get_edge_index, pre2R_mpnn
from torch_geometric.data import Data, DataLoader
# MOCU imported lazily in base.run_episode() to maintain separate usage pattern
# DAD uses policy network (PyTorch) for selection, base.run_episode() uses PyTorch CUDA for MOCU computation


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
        self.mpnn_model = None
        self.mpnn_mean = None
        self.mpnn_std = None
        self.use_expected_mocu = True  # Enable expected MOCU features by default
        
        # Load policy network
        self._load_policy(policy_model_path)
        
        # Try to load MPNN predictor for expected MOCU computation (like INN/NN)
        self._load_mpnn_predictor()
        
        print(f"[DAD-MOCU] Initialized with policy on {self.device}")
        if self.mpnn_model is not None:
            print(f"[DAD-MOCU] MPNN predictor loaded - will use expected MOCU features (enhanced mode)")
        else:
            print(f"[DAD-MOCU] MPNN predictor not available - using standard mode")
    
    def _load_policy(self, policy_model_path):
        """Load trained DAD policy network."""
        if policy_model_path is None:
            # Try to find policy in new structure: models/{config_name}/dad_policy_N{N}_K{K}.pth or dad_policy_N{N}.pth
            # Search in all config folders
            models_root = PROJECT_ROOT / 'models'
            found_paths = []
            
            if models_root.exists():
                # Search for both patterns: with K and without K
                for config_dir in models_root.iterdir():
                    if config_dir.is_dir():
                        # Try pattern with K: dad_policy_N{N}_K{K}*.pth (includes method suffix)
                        for k_file in config_dir.glob(f'dad_policy_N{self.N}_K*.pth'):
                            # Prefer best model if exists, otherwise regular model
                            if '_best.pth' in k_file.name:
                                found_paths.append((k_file, k_file.stat().st_mtime, True))
                            else:
                                found_paths.append((k_file, k_file.stat().st_mtime, False))
                        
                        # Try pattern without K: dad_policy_N{N}.pth (backward compatibility)
                        candidate = config_dir / f'dad_policy_N{self.N}.pth'
                        if candidate.exists():
                            found_paths.append((candidate, candidate.stat().st_mtime, False))
            
            # Use most recently modified if found (prefer best models)
            if found_paths:
                # Sort: best models first, then by modification time
                found_paths.sort(key=lambda x: (not x[2], -x[1]))
                policy_model_path = found_paths[0][0]
                print(f"[DAD-MOCU] Found policy at: {policy_model_path}")
            else:
                # Fall back to old flat structure
                policy_model_path = PROJECT_ROOT / 'models' / f'dad_policy_N{self.N}.pth'
                
                if not policy_model_path.exists():
                    raise FileNotFoundError(
                        f"Policy model not found. Searched in:\n"
                        f"  - models/*/dad_policy_N{self.N}_K*.pth (with K)\n"
                        f"  - models/*/dad_policy_N{self.N}.pth (without K)\n"
                        f"  - models/dad_policy_N{self.N}.pth (old structure)\n"
                        f"Please train a DAD policy first using:\n"
                        f"  python scripts/train_dad_policy.py --data-path <data> --name dad_policy_N{self.N}_K<K>"
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
    
    def _load_mpnn_predictor(self):
        """Load MPNN predictor for computing expected MOCU features (like INN/NN)."""
        try:
            from src.models.predictors.predictor_utils import load_mpnn_predictor
            import os
            
            # Get model name from environment variable or auto-detect
            model_name = os.getenv('MOCU_MODEL_NAME', f'cons{self.N}')
            
            self.mpnn_model, self.mpnn_mean, self.mpnn_std = load_mpnn_predictor(
                model_name=model_name, device=str(self.device)
            )
            
            if self.mpnn_model is not None:
                self.mpnn_model.eval()
                self.mpnn_model = self.mpnn_model.to(self.device)
        except Exception as e:
            print(f"[DAD-MOCU] Warning: Could not load MPNN predictor: {e}")
            self.mpnn_model = None
            self.mpnn_mean = None
            self.mpnn_std = None
    
    def _compute_expected_mocu_matrix(self, w, a_lower_bounds, a_upper_bounds):
        """
        Compute R matrix (expected remaining MOCU) for all possible experiments.
        
        This is the same computation that INN/NN uses - gives DAD the same information.
        """
        if self.mpnn_model is None:
            return None
        
        data_list = []
        P_syn_list = []
        
        x = torch.from_numpy(w.astype(np.float32)).unsqueeze(dim=1).to(self.device)
        edge_index = get_edge_index(self.N).long().to(self.device)
        dummy_y = torch.tensor(0.0).unsqueeze(0).unsqueeze(0).to(self.device)
        
        for i in range(self.N):
            for j in range(i + 1, self.N):
                # Compute f_inv (critical coupling for this pair)
                w_i = w[i]
                w_j = w[j]
                f_inv = 0.5 * np.abs(w_i - w_j)
                
                # Scenario 1: Assume observation is synchronized
                a_upper_syn = a_upper_bounds.copy()
                a_lower_syn = a_lower_bounds.copy()
                
                a_tilde = min(max(f_inv, a_lower_bounds[i, j]), a_upper_bounds[i, j])
                a_lower_syn[j, i] = a_tilde
                a_lower_syn[i, j] = a_tilde
                
                P_syn = (a_upper_bounds[i, j] - a_tilde) / (
                    a_upper_bounds[i, j] - a_lower_bounds[i, j] + 1e-10
                )
                P_syn_list.append(P_syn)
                
                edge_attr_syn = get_edge_attr_from_bounds(a_lower_syn, a_upper_syn, self.N).to(self.device)
                data_syn = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_syn, y=dummy_y)
                data_list.append(data_syn)
                
                # Scenario 2: Assume observation is non-synchronized
                a_upper_nonsyn = a_upper_bounds.copy()
                a_lower_nonsyn = a_lower_bounds.copy()
                
                a_upper_nonsyn[i, j] = a_tilde
                a_upper_nonsyn[j, i] = a_tilde
                
                edge_attr_nonsyn = get_edge_attr_from_bounds(a_lower_nonsyn, a_upper_nonsyn, self.N).to(self.device)
                data_nonsyn = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_nonsyn, y=dummy_y)
                data_list.append(data_nonsyn)
        
        if not data_list:
            return np.zeros((self.N, self.N))
        
        # Batch prediction
        dataloader = DataLoader(data_list, batch_size=128, shuffle=False)
        predictions = []
        
        with torch.no_grad():
            for batch_data in dataloader:
                batch_data = batch_data.to(self.device)
                pred = self.mpnn_model(batch_data).cpu().numpy()
                predictions.extend(pred)
        
        predictions = np.array(predictions)
        predictions = predictions * self.mpnn_std + self.mpnn_mean  # Denormalize
        
        # Convert predictions to R matrix
        R_matrix = pre2R_mpnn(predictions, P_syn_list, self.N)
        
        return R_matrix
    
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
        
        # Compute expected MOCU features (like INN/NN) if MPNN is available
        expected_mocu_features = None
        if self.use_expected_mocu and self.mpnn_model is not None:
            try:
                R_matrix = self._compute_expected_mocu_matrix(w, a_lower_bounds, a_upper_bounds)
                if R_matrix is not None:
                    # Extract expected MOCU values for available actions
                    expected_mocu_values = []
                    for i, j in available_pairs:
                        expected_mocu_values.append(R_matrix[i, j])
                    
                    # Convert to tensor: [1, num_actions]
                    # Pad with zeros for unavailable actions (will be masked anyway)
                    num_actions = self.N * (self.N - 1) // 2
                    expected_mocu_array = np.zeros(num_actions, dtype=np.float32)
                    for idx, (i, j) in enumerate(available_pairs):
                        action_idx = self.policy_net.pair_to_idx(i, j)
                        expected_mocu_array[action_idx] = expected_mocu_values[idx]
                    
                    # Normalize expected MOCU features for better training stability
                    # Use min-max normalization: (x - min) / (max - min + eps)
                    if len(expected_mocu_values) > 0:
                        mocu_min = min(expected_mocu_values)
                        mocu_max = max(expected_mocu_values)
                        mocu_range = mocu_max - mocu_min
                        if mocu_range > 1e-6:
                            # Normalize to [0, 1] range
                            expected_mocu_array = (expected_mocu_array - mocu_min) / mocu_range
                    
                    expected_mocu_features = torch.tensor([expected_mocu_array], dtype=torch.float32, device=self.device)
            except Exception as e:
                print(f"[DAD-MOCU] Warning: Failed to compute expected MOCU features: {e}")
                expected_mocu_features = None
        
        # Get policy action probabilities (greedy/deterministic)
        with torch.no_grad():
            self.policy_net.eval()
            action_logits, action_probs = self.policy_net(
                state_data, history_tensor, available_mask_tensor, 
                expected_mocu_features=expected_mocu_features
            )
        
        # Select action with highest probability (deterministic)
        action_probs = action_probs.squeeze(0)  # [num_actions]
        action_idx = torch.argmax(action_probs).item()
        
        # Convert action index to (i, j) pair using policy network method
        i, j = self.policy_net.idx_to_pair(action_idx)
        
        return i, j

