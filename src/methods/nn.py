"""
NN (Static Neural Network) OED Method

This is the static (non-iterative) version of the MPNN-based greedy method.
It computes the expected MOCU matrix once and reuses it for all selections.

In the paper: "NN" method
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
from src.models.predictors.predictors import MPNNPlusPredictor, get_edge_attr_from_bounds, get_edge_index, pre2R_mpnn
# MOCU imported lazily in run_episode() to maintain separate usage pattern (original paper 2023)


class NN_Method(OEDMethod):
    """
    Static Neural Network (NN) method for OED.
    
    Uses MPNNPlusPredictor to compute expected MOCU once at the beginning,
    then greedily selects experiments without re-evaluation.
    
    This is faster than iNN but less adaptive.
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
            model_name: Name of trained model directory
            gpu_id: GPU device ID
        """
        super().__init__(N, K_max, deltaT, MReal, TReal, it_idx)
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.model = None
        self.mean = None
        self.std = None
        self.R_matrix = np.zeros((N, N))
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        # Load model once
        self._load_model_and_stats()
    
    def _load_model_and_stats(self):
        """Load trained MPNN model and normalization statistics."""
        # Support both new structure (models/{config}/{timestamp}/) and old (models/{name}/)
        model_path = None
        stats_path = None
        
        # Check if model_name contains timestamp (format: config_MMDDYYYY_HHMMSS)
        # Timestamp format: MMDDYYYY_HHMMSS (e.g., 11012025_163858)
        if '_' in self.model_name:
            parts = self.model_name.split('_')
            # Check if last part is 6 digits (HHMMSS) and second-to-last is 8 digits (MMDDYYYY)
            if len(parts) >= 3 and len(parts[-1]) == 6 and parts[-1].isdigit() and len(parts[-2]) == 8 and parts[-2].isdigit():
                # Last two parts form timestamp: MMDDYYYY_HHMMSS
                timestamp = f"{parts[-2]}_{parts[-1]}"
                config_name = '_'.join(parts[:-2])
                # Try timestamped path first
                candidate_model = PROJECT_ROOT / 'models' / config_name / timestamp / 'model.pth'
                candidate_stats = PROJECT_ROOT / 'models' / config_name / timestamp / 'statistics.pth'
                if candidate_model.exists() and candidate_stats.exists():
                    model_path = candidate_model
                    stats_path = candidate_stats
                else:
                    # Fall back to flat structure
                    model_path = PROJECT_ROOT / 'models' / self.model_name / 'model.pth'
                    stats_path = PROJECT_ROOT / 'models' / self.model_name / 'statistics.pth'
            else:
                # Doesn't match timestamp pattern, try flat structure
                model_path = PROJECT_ROOT / 'models' / self.model_name / 'model.pth'
                stats_path = PROJECT_ROOT / 'models' / self.model_name / 'statistics.pth'
        else:
            # Old structure: models/{model_name}/
            model_path = PROJECT_ROOT / 'models' / self.model_name / 'model.pth'
            stats_path = PROJECT_ROOT / 'models' / self.model_name / 'statistics.pth'
        
        if not model_path.exists() or not stats_path.exists():
            raise FileNotFoundError(
                f"Model or statistics not found for {self.model_name}. "
                f"Please ensure the model is trained and saved at {model_path}"
            )
        
        # Original code always used dim=32 (hardcoded in legacy_mpnn.py)
        # The model architecture is independent of N - it works for any graph size
        # Load checkpoint to infer hidden dimension from saved model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # MPNNPlusPredictor uses 'dim' (hidden dimension), not N
        # Infer dim from lin0.weight shape: [dim, 1]
        state_dict = checkpoint if isinstance(checkpoint, dict) else (checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint)
        
        if 'lin0.weight' in state_dict:
            saved_dim = state_dict['lin0.weight'].shape[0]
        else:
            # Default dim=32 (matching original paper implementation)
            saved_dim = 32
        
        self.model = MPNNPlusPredictor(dim=saved_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False), strict=True)
        self.model.eval()
        
        stats = torch.load(stats_path, map_location=self.device, weights_only=False)
        self.mean = stats['mean']
        self.std = stats['std']
        
        print(f"[NN] Loaded MPNNPlusPredictor model '{self.model_name}' (dim={saved_dim}) on {self.device}")
    
    def _compute_expected_mocu_matrix(self, w, a_lower_bounds, a_upper_bounds):
        """
        Compute R matrix (expected remaining MOCU) for all possible experiments.
        
        This is only called ONCE at the beginning for NN (static) method.
        """
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
                
                edge_attr_syn = get_edge_attr_from_bounds(a_lower_syn, a_upper_syn, self.N).t().to(self.device)
                data_syn = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_syn, y=dummy_y)
                data_list.append(data_syn)
                
                # Scenario 2: Assume observation is non-synchronized
                a_upper_nonsyn = a_upper_bounds.copy()
                a_lower_nonsyn = a_lower_bounds.copy()
                
                a_upper_nonsyn[i, j] = a_tilde
                a_upper_nonsyn[j, i] = a_tilde
                
                edge_attr_nonsyn = get_edge_attr_from_bounds(a_lower_nonsyn, a_upper_nonsyn, self.N).t().to(self.device)
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
                pred = self.model(batch_data).cpu().numpy()
                predictions.extend(pred)
        
        predictions = np.array(predictions)
        predictions = predictions * self.std + self.mean  # Denormalize
        
        # Convert predictions to R matrix
        R_matrix = pre2R_mpnn(predictions, P_syn_list, self.N)
        
        return R_matrix
    
    def select_experiment(self, w, a_lower_bounds, a_upper_bounds, criticalK, isSynchronized, history):
        """
        Select next experiment using static NN strategy.
        
        Computes R matrix only once, then greedily selects from it.
        """
        # Compute R matrix only on first call
        if not np.any(self.R_matrix):
            print("[NN] Computing expected MOCU matrix (static, once only)...")
            self.R_matrix = self._compute_expected_mocu_matrix(w, a_lower_bounds, a_upper_bounds)
        
        # Mask out already selected experiments
        for (i, j), _ in history:
            self.R_matrix[i, j] = 0.0
            self.R_matrix[j, i] = 0.0
        
        # Find experiment with minimum expected MOCU
        valid_R_values = self.R_matrix[np.nonzero(self.R_matrix)]
        
        if valid_R_values.size == 0:
            print("[NN] Warning: No valid experiments left!")
            return -1, -1
        
        min_val = np.min(valid_R_values)
        min_indices = np.where(self.R_matrix == min_val)
        
        if len(min_indices[0]) > 1:
            min_i = int(min_indices[0][0])
            min_j = int(min_indices[1][0])
        else:
            min_i = int(min_indices[0])
            min_j = int(min_indices[1])
        
        return min_i, min_j

