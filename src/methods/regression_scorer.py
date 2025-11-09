"""
Regression Scorer OED Method

Simple regressive sequencer that scores candidate designs using MPNN predictor
and greedily selects the top-scoring action.

This method uses the same MPNN predictor as iNN/NN methods, but instead of
computing an R-matrix, it directly scores each candidate design by predicting
the MOCU after applying that design.

Trade-offs vs iNN/NN:
- Performance: Slightly worse than iNN/NN because it only scores available pairs
  (not the full matrix), which may miss some global optimization opportunities.
- Speed: Faster than iNN/NN, especially for large N, because:
  - iNN/NN compute R-matrix for ALL pairs (N*(N-1)/2) at each step
  - regression_scorer only scores AVAILABLE pairs (decreases as experiments progress)
  - As N grows, the speed advantage becomes more significant

In the paper: Similar to "NN" but with direct design scoring instead of R-matrix.
"""

import time
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.methods.base import OEDMethod
from src.models.predictors.mpnn_plus import MPNNPlusPredictor
from src.models.predictors.utils import get_edge_attr_from_bounds, get_edge_index
# MOCU imported lazily in run_episode() to maintain separate usage pattern (original paper 2023)


class RegressionScorer_Method(OEDMethod):
    """
    Regression Scorer method for OED.
    
    Uses MPNN predictor to score each candidate design by predicting MOCU
    after applying that design, then greedily selects the best one.
    
    This is simpler than iNN/NN (no R-matrix computation) and uses the same
    MPNN predictor, so it shares the same training data and evaluation pipeline.
    
    Performance vs Speed Trade-off:
    - Computes expected MOCU only for available (unobserved) pairs
    - Faster than iNN/NN for large N, but may have slightly worse performance
      due to not considering the full matrix structure
    """
    
    def __init__(self, N, K_max, deltaT, MReal, TReal, it_idx, model_name, gpu_id=0):
        """
        Args:
            N: Number of oscillators
            K_max: Number of Monte Carlo samples for MOCU
            deltaT: Time step
            MReal: Number of time steps
            TReal: Time horizon
            it_idx: Number of MOCU averaging iterations
            model_name: Name of trained MPNN model (same as iNN/NN)
            gpu_id: GPU device ID
        """
        super().__init__(N, K_max, deltaT, MReal, TReal, it_idx)
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.model = None
        self.mean = None
        self.std = None
        # Force CPU when PyCUDA is used (steps 1-3) to avoid CUDA context conflicts
        import os
        use_pycuda = os.getenv('USE_PYCUDA_FOR_BASELINES', '0') == '1'
        if use_pycuda:
            self.device = torch.device('cpu')  # Use CPU when PyCUDA is active
        else:
            self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        # Load MPNN model (same as iNN/NN)
        self._load_model_and_stats()
    
    def _load_model_and_stats(self):
        """Load trained MPNN model and normalization statistics (same as iNN/NN)."""
        # New structure: models/{config_name}/model.pth and statistics.pth
        # model_name is just the config name (e.g., "N5_config")
        model_path = PROJECT_ROOT / 'models' / self.model_name / 'model.pth'
        stats_path = PROJECT_ROOT / 'models' / self.model_name / 'statistics.pth'
        
        # Fallback to old structure for backward compatibility
        if not model_path.exists() or not stats_path.exists():
            # Try old flat structure: models/{model_name}/
            old_model_path = PROJECT_ROOT / 'models' / self.model_name / 'model.pth'
            old_stats_path = PROJECT_ROOT / 'models' / self.model_name / 'statistics.pth'
            if old_model_path.exists() and old_stats_path.exists():
                model_path = old_model_path
                stats_path = old_stats_path
        
        if not model_path.exists() or not stats_path.exists():
            raise FileNotFoundError(
                f"MPNN model not found for RegressionScorer. Searched:\n"
                f"  - models/{self.model_name}/model.pth (old structure)\n"
                f"  - models/{config_name}/{timestamp}/model.pth (new structure)\n"
                f"Please train MPNN predictor first (same as iNN/NN methods)."
            )
        
        # Load model (same approach as iNN/NN methods)
        # MPNNPlusPredictor uses 'dim' parameter, not node_features/edge_features
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Infer dim from checkpoint (same as iNN/NN)
        state_dict = checkpoint if isinstance(checkpoint, dict) else (checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint)
        
        if 'lin0.weight' in state_dict:
            saved_dim = state_dict['lin0.weight'].shape[0]
        else:
            # Default dim=32 (matching original paper implementation)
            saved_dim = 32
        
        self.model = MPNNPlusPredictor(dim=saved_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False), strict=True)
        self.model.eval()
        
        # Load statistics
        stats = torch.load(stats_path, map_location=self.device, weights_only=False)
        self.mean = stats['mean'] if isinstance(stats['mean'], (int, float)) else stats['mean'].item()
        self.std = stats['std'] if isinstance(stats['std'], (int, float)) else stats['std'].item()
        
        print(f"[RegressionScorer] Loaded MPNN model: {self.model_name} (dim={saved_dim})")
    
    def _score_design(self, w, a_lower_bounds, a_upper_bounds, i, j, criticalK):
        """
        Score a candidate design (i, j) by predicting expected MOCU after applying it.
        
        Computes expected MOCU over both sync and non-sync outcomes, similar to iNN/NN R-matrix.
        
        Args:
            w: Natural frequencies [N]
            a_lower_bounds: Current lower bounds [N, N]
            a_upper_bounds: Current upper bounds [N, N]
            i, j: Candidate design pair
            criticalK: Critical coupling thresholds [N, N]
        
        Returns:
            score: Expected MOCU after applying design (i, j) (lower is better)
        """
        # Compute expected MOCU for both sync and non-sync outcomes
        # Similar to iNN/NN R-matrix computation
        
        # Get critical coupling threshold for this pair
        # Compute f_inv (critical coupling) same as iNN/NN
        w_i = w[i]
        w_j = w[j]
        f_inv = 0.5 * np.abs(w_i - w_j)
        
        # Current bounds for this pair
        a_lower_ij = a_lower_bounds[i, j]
        a_upper_ij = a_upper_bounds[i, j]
        
        # Compute a_tilde (same as iNN/NN)
        a_tilde = min(max(f_inv, a_lower_ij), a_upper_ij)
        
        # Probability of sync: P(a_ij >= a_tilde) = (a_upper - a_tilde) / (a_upper - a_lower)
        # Same computation as iNN/NN
        P_sync = (a_upper_ij - a_tilde) / (a_upper_ij - a_lower_ij + 1e-10)
        P_nonsync = 1.0 - P_sync
        
        # Create state data for MPNN
        x = torch.from_numpy(w.astype(np.float32)).unsqueeze(-1).to(self.device)  # [N, 1]
        edge_index = get_edge_index(self.N).to(self.device)
        
        # Predict MOCU for sync outcome
        # Same logic as iNN/NN: a_lower = a_tilde
        a_lower_syn = a_lower_bounds.copy()
        a_upper_syn = a_upper_bounds.copy()
        a_lower_syn[i, j] = a_tilde
        a_lower_syn[j, i] = a_tilde
        
        edge_attr_syn = get_edge_attr_from_bounds(a_lower_syn, a_upper_syn, self.N).to(self.device)
        data_syn = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_syn)
        
        # Predict MOCU for non-sync outcome
        # Same logic as iNN/NN: a_upper = a_tilde
        a_lower_nonsyn = a_lower_bounds.copy()
        a_upper_nonsyn = a_upper_bounds.copy()
        a_upper_nonsyn[i, j] = a_tilde
        a_upper_nonsyn[j, i] = a_tilde
        
        edge_attr_nonsyn = get_edge_attr_from_bounds(a_lower_nonsyn, a_upper_nonsyn, self.N).to(self.device)
        data_nonsyn = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_nonsyn)
        
        # Batch prediction
        data_list = [data_syn, data_nonsyn]
        dataloader = DataLoader(data_list, batch_size=2, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch_data in dataloader:
                batch_data = batch_data.to(self.device)
                pred = self.model(batch_data).cpu().numpy()
                predictions.extend(pred)
        
        predictions = np.array(predictions)
        predictions = predictions * self.std + self.mean  # Denormalize
        
        mocu_sync = predictions[0]
        mocu_nonsync = predictions[1]
        
        # Expected MOCU
        expected_mocu = P_sync * mocu_sync + P_nonsync * mocu_nonsync
        
        return expected_mocu
    
    def select_experiment(self, w, a_lower_bounds, a_upper_bounds, criticalK, isSynchronized, history):
        """
        Select next experiment by scoring all candidate designs.
        
        Scores each available design using MPNN predictor and greedily
        selects the one with lowest predicted MOCU.
        
        Args:
            w: Natural frequencies [N]
            a_lower_bounds: Current lower bounds [N, N]
            a_upper_bounds: Current upper bounds [N, N]
            criticalK: Critical coupling strengths (not used directly)
            isSynchronized: Synchronization status (not used directly)
            history: List of ((i, j), observation) tuples
        
        Returns:
            (i, j): Selected experiment pair
        """
        # Get observed pairs
        observed_pairs = set()
        if history:
            for pair_obs in history:
                if isinstance(pair_obs, tuple) and len(pair_obs) == 2:
                    (i, j), obs = pair_obs
                    observed_pairs.add((i, j))
        
        # Get available pairs
        available_pairs = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if (i, j) not in observed_pairs:
                    available_pairs.append((i, j))
        
        if not available_pairs:
            print("[RegressionScorer] Warning: No available experiments left!")
            return -1, -1
        
        # Score all available designs (batch for efficiency)
        # Instead of scoring individually, batch all pairs together like NN/iNN
        # Batch all available pairs for efficient prediction
        data_list = []
        P_sync_list = []
        pair_indices = []  # Track which pair each prediction corresponds to
        
        x = torch.from_numpy(w.astype(np.float32)).unsqueeze(-1).to(self.device)  # [N, 1]
        edge_index = get_edge_index(self.N).to(self.device)
        
        # Pre-allocate arrays to avoid repeated allocations
        num_pairs = len(available_pairs)
        
        for idx, (i, j) in enumerate(available_pairs):
            # Compute f_inv (critical coupling for this pair)
            w_i = w[i]
            w_j = w[j]
            f_inv = 0.5 * np.abs(w_i - w_j)
            
            # Current bounds for this pair
            a_lower_ij = a_lower_bounds[i, j]
            a_upper_ij = a_upper_bounds[i, j]
            
            # Compute a_tilde
            a_tilde = min(max(f_inv, a_lower_ij), a_upper_ij)
            
            # Probability of sync
            P_sync = (a_upper_ij - a_tilde) / (a_upper_ij - a_lower_ij + 1e-10)
            P_sync_list.append(P_sync)
            pair_indices.append((i, j))
            
            # Scenario 1: Synchronized observation
            # Use np.copy() explicitly for clarity (slightly faster than .copy())
            a_lower_syn = np.copy(a_lower_bounds)
            a_upper_syn = np.copy(a_upper_bounds)
            a_lower_syn[i, j] = a_tilde
            a_lower_syn[j, i] = a_tilde
            
            edge_attr_syn = get_edge_attr_from_bounds(a_lower_syn, a_upper_syn, self.N).to(self.device)
            data_syn = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_syn)
            data_list.append(data_syn)
            
            # Scenario 2: Non-synchronized observation
            a_lower_nonsyn = np.copy(a_lower_bounds)
            a_upper_nonsyn = np.copy(a_upper_bounds)
            a_upper_nonsyn[i, j] = a_tilde
            a_upper_nonsyn[j, i] = a_tilde
            
            edge_attr_nonsyn = get_edge_attr_from_bounds(a_lower_nonsyn, a_upper_nonsyn, self.N).to(self.device)
            data_nonsyn = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_nonsyn)
            data_list.append(data_nonsyn)
        
        # Batch prediction (much faster than individual predictions)
        dataloader = DataLoader(data_list, batch_size=128, shuffle=False)
        predictions = []
        
        with torch.no_grad():
            for batch_data in dataloader:
                batch_data = batch_data.to(self.device)
                pred = self.model(batch_data).cpu().numpy()
                predictions.extend(pred)
        
        predictions = np.array(predictions)
        predictions = predictions * self.std + self.mean  # Denormalize
        
        # Compute expected MOCU for each pair
        scores = []
        for idx, (i, j) in enumerate(pair_indices):
            mocu_sync = predictions[idx * 2]
            mocu_nonsync = predictions[idx * 2 + 1]
            P_sync = P_sync_list[idx]
            P_nonsync = 1.0 - P_sync
            expected_mocu = P_sync * mocu_sync + P_nonsync * mocu_nonsync
            scores.append(expected_mocu)
        
        # Select design with minimum predicted MOCU (greedy)
        best_idx = np.argmin(scores)
        return available_pairs[best_idx]

