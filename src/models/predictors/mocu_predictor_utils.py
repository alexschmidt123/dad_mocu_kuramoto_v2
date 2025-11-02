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

# File is at: src/models/predictors/mocu_predictor_utils.py
# Go up 4 levels to reach repo root: predictors -> models -> src -> repo_root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
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
    
    # New structure: models/{config_name}/{timestamp}/model.pth
    # But also support old structure: models/{model_name}/model.pth
    # Try new structure first (timestamped), then fall back to old
    model_path = None
    stats_path = None
    
    # New structure: models/{config_name}/{timestamp}/model.pth
    # Model name format from run.sh: {config_name}_{MMDDYYYY_HHMMSS}
    # Timestamp format: MMDDYYYY_HHMMSS (e.g., 11012025_163858)
    # So splitting by '_' gives: ['config', 'name', 'MMDDYYYY', 'HHMMSS']
    # We need to recognize that last 2 parts form the timestamp
    if '_' in model_name:
        parts = model_name.split('_')
        # Check if last part is 6 digits (HHMMSS format) and second-to-last is 8 digits (MMDDYYYY format)
        if len(parts) >= 3 and len(parts[-1]) == 6 and parts[-1].isdigit() and len(parts[-2]) == 8 and parts[-2].isdigit():
            # Last two parts form timestamp: MMDDYYYY_HHMMSS
            timestamp = f"{parts[-2]}_{parts[-1]}"
            config_name = '_'.join(parts[:-2])
            # Try timestamped path: models/{config_name}/{timestamp}/model.pth
            candidate_model = PROJECT_ROOT / 'models' / config_name / timestamp / 'model.pth'
            candidate_stats = PROJECT_ROOT / 'models' / config_name / timestamp / 'statistics.pth'
            if candidate_model.exists() and candidate_stats.exists():
                model_path = candidate_model
                stats_path = candidate_stats
            else:
                # Fall back to flat structure
                model_path = PROJECT_ROOT / 'models' / model_name / 'model.pth'
                stats_path = PROJECT_ROOT / 'models' / model_name / 'statistics.pth'
        else:
            # Doesn't match timestamp pattern, try flat structure
            model_path = PROJECT_ROOT / 'models' / model_name / 'model.pth'
            stats_path = PROJECT_ROOT / 'models' / model_name / 'statistics.pth'
    else:
        # Old structure: models/{model_name}/
        model_path = PROJECT_ROOT / 'models' / model_name / 'model.pth'
        stats_path = PROJECT_ROOT / 'models' / model_name / 'statistics.pth'
    
    if not model_path.exists() or not stats_path.exists():
        # Provide detailed error message with searched paths
        searched_paths = []
        if '_' in model_name:
            parts = model_name.split('_')
            if len(parts) >= 3 and len(parts[-1]) == 6 and parts[-1].isdigit() and len(parts[-2]) == 8 and parts[-2].isdigit():
                timestamp = f"{parts[-2]}_{parts[-1]}"
                config_name = '_'.join(parts[:-2])
                searched_paths.append(f"  - {PROJECT_ROOT / 'models' / config_name / timestamp / 'model.pth'}")
                searched_paths.append(f"  - {PROJECT_ROOT / 'models' / config_name / timestamp / 'statistics.pth'}")
        searched_paths.append(f"  - {PROJECT_ROOT / 'models' / model_name / 'model.pth'}")
        searched_paths.append(f"  - {PROJECT_ROOT / 'models' / model_name / 'statistics.pth'}")
        
        raise FileNotFoundError(
            f"Model or statistics not found for {model_name}.\n"
            f"Searched paths:\n" + "\n".join(searched_paths) + "\n"
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
