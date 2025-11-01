"""
Utility functions for using MPNN predictor to predict MOCU values.

Reuses the predictor loading logic from iNN/NN methods (paper 2023).
This allows fast MOCU prediction for DAD training instead of slow CUDA computation.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.predictors.all_predictors import MPNNPlusPredictor, get_edge_index, get_edge_attr_from_bounds


def load_mpnn_predictor(model_name, device='cuda'):
    """
    Load MPNN predictor model and statistics (same as iNN/NN methods).
    
    This reuses the exact loading logic from paper 2023 code.
    
    Args:
        model_name: Name of trained model (e.g., 'cons5', 'cons7')
        device: torch device
    
    Returns:
        model: Loaded MPNNPlusPredictor model (in eval mode)
        mean: Normalization mean
        std: Normalization std
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    model_path = PROJECT_ROOT / 'models' / model_name / 'model.pth'
    stats_path = PROJECT_ROOT / 'models' / model_name / 'statistics.pth'
    
    if not model_path.exists() or not stats_path.exists():
        raise FileNotFoundError(
            f"Model or statistics not found for {model_name}. "
            f"Please train MPNN predictor first:\n"
            f"  python scripts/train_mocu_predictor.py --name {model_name}"
        )
    
    # Reuse loading logic from iNN/NN (same as paper 2023)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint if isinstance(checkpoint, dict) else (
        checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
    )
    
    # Infer dim from saved model (same as inn.py and nn.py)
    if 'lin0.weight' in state_dict:
        saved_dim = state_dict['lin0.weight'].shape[0]
    else:
        saved_dim = 32  # Default
    
    model = MPNNPlusPredictor(dim=saved_dim).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    stats = torch.load(stats_path, map_location=device, weights_only=False)
    mean = stats['mean']
    std = stats['std']
    
    return model, mean, std


def predict_mocu(model, mean, std, w, a_lower, a_upper, device='cuda'):
    """
    Predict MOCU for given state using loaded MPNN model.
    
    Args:
        model: Loaded MPNNPlusPredictor model
        mean: Normalization mean
        std: Normalization std
        w: Natural frequencies [N]
        a_lower: Lower bounds [N, N]
        a_upper: Upper bounds [N, N]
        device: torch device
    
    Returns:
        mocu_pred: Predicted MOCU value (scalar)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    N = len(w)
    
    # Create PyG Data object (same format as iNN/NN methods)
    x = torch.from_numpy(w.astype(np.float32)).unsqueeze(-1)  # [N, 1]
    edge_index = get_edge_index(N).to(device)
    edge_attr = get_edge_attr_from_bounds(a_lower, a_upper, N).to(device)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data = data.to(device)
    
    # Predict (same as iNN/NN)
    with torch.no_grad():
        pred_normalized = model(data).cpu().item()
    
    # Denormalize (same as iNN/NN)
    mocu_pred = pred_normalized * std + mean
    
    return float(mocu_pred)
