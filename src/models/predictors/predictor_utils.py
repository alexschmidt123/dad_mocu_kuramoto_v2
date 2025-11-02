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

# File is at: src/models/predictors/predictor_utils.py
# Go up 4 levels to reach repo root: predictors -> models -> src -> repo_root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.predictors.mpnn_plus import MPNNPlusPredictor
from src.models.predictors.utils import get_edge_index, get_edge_attr_from_bounds


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
    # torch is already imported at module level
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
            f"  python scripts/train_predictor.py --name {model_name}"
        )
    
    # Reuse loading logic from iNN/NN (same as paper 2023)
    # Ensure clean CUDA state before loading
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats (same as inn.py and nn.py)
    if isinstance(checkpoint, dict):
        # New format: {'model_state_dict': ..., 'config': ...}
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model_config = checkpoint.get('config', {})
    else:
        # Old format: direct state_dict or model object
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
            model_config = {}
        else:
            state_dict = checkpoint
            model_config = {}
    
    # Infer dim from saved model (same as inn.py and nn.py)
    if 'lin0.weight' in state_dict:
        saved_dim = state_dict['lin0.weight'].shape[0]
    else:
        saved_dim = model_config.get('dim', 32)  # Use config dim if available, else default
    
    model = MPNNPlusPredictor(dim=saved_dim).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # Ensure model is fully loaded and on correct device
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    stats = torch.load(stats_path, map_location=device, weights_only=False)
    mean = stats['mean']
    std = stats['std']
    
    return model, mean, std


def predict_mocu(model, mean, std, w, a_lower, a_upper, device='cuda'):
    """
    Predict MOCU for given state using loaded MPNN model.
    
    IMPORTANT: This function uses PyTorch CUDA for MPNN prediction.
    
    Args:
        model: Loaded MPNNPlusPredictor model (should already be on correct device)
        mean: Normalization mean
        std: Normalization std
        w: Natural frequencies [N]
        a_lower: Lower bounds [N, N]
        a_upper: Upper bounds [N, N]
        device: torch device string
    
    Returns:
        mocu_pred: Predicted MOCU value (scalar)
    """
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    N = len(w)
    
    # Ensure model is valid
    if model is None:
        raise ValueError("MPNN model is None - cannot predict MOCU")
    
    # Model should already be on device from load_mpnn_predictor
    # CRITICAL: Do NOT move model if it's already on the correct device
    # Repeated device transfers can cause segmentation faults
    model_device = next(model.parameters()).device
    if model_device != device_obj:
        # Only move if absolutely necessary, and synchronize before/after
        if device_obj.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        model = model.to(device_obj)
        if device_obj.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # CRITICAL: Ensure model is in eval mode and no gradients
    model.eval()
    
    # Ensure CUDA is synchronized before creating data
    # This is CRITICAL to prevent conflicts with policy network operations
    if device_obj.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Create PyG Data object (same format as iNN/NN methods)
    # Use pin_memory=False to avoid potential memory issues
    x = torch.from_numpy(w.astype(np.float32)).unsqueeze(-1)  # [N, 1]
    edge_index = get_edge_index(N).to(device_obj, non_blocking=False)
    edge_attr = get_edge_attr_from_bounds(a_lower, a_upper, N).to(device_obj, non_blocking=False)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data = data.to(device_obj, non_blocking=False)
    
    # Ensure data is fully on device before prediction
    if device_obj.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Predict (same as iNN/NN)
    # CRITICAL: Use torch.no_grad() to avoid gradient computation
    # This also prevents any autograd graph issues
    try:
        with torch.no_grad():
            # Ensure model is in eval mode (redundant check but safe)
            model.eval()
            pred_normalized = model(data)
            
            # CRITICAL: Synchronize before moving to CPU
            if device_obj.type == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Move to CPU before converting to item (safer, prevents device errors)
            pred_normalized = pred_normalized.cpu().item()
            
    except RuntimeError as e:
        # Handle CUDA errors specifically
        if device_obj.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        raise RuntimeError(f"MPNN prediction failed (RuntimeError): {e}") from e
    except Exception as e:
        # If prediction fails, synchronize CUDA and re-raise
        if device_obj.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        raise RuntimeError(f"MPNN prediction failed: {type(e).__name__}: {e}") from e
    
    # Ensure prediction is complete before returning
    if device_obj.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Denormalize (same as iNN/NN)
    mocu_pred = pred_normalized * std + mean
    
    return float(mocu_pred)
