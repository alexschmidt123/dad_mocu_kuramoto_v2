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
    
    # New structure: models/{config_name}/model.pth and statistics.pth
    # model_name is just the config name (e.g., "N5_config")
    model_path = PROJECT_ROOT / 'models' / model_name / 'model.pth'
    stats_path = PROJECT_ROOT / 'models' / model_name / 'statistics.pth'
    
    # Fallback to old structure for backward compatibility
    if not model_path.exists() or not stats_path.exists():
        # Try old flat structure: models/{model_name}/
        old_model_path = PROJECT_ROOT / 'models' / model_name / 'model.pth'
        old_stats_path = PROJECT_ROOT / 'models' / model_name / 'statistics.pth'
        if old_model_path.exists() and old_stats_path.exists():
            model_path = old_model_path
            stats_path = old_stats_path
    
    if not model_path.exists() or not stats_path.exists():
        raise FileNotFoundError(
            f"Model or statistics not found for {model_name}.\n"
            f"Searched paths:\n"
            f"  - {PROJECT_ROOT / 'models' / model_name / 'model.pth'}\n"
            f"  - {PROJECT_ROOT / 'models' / model_name / 'statistics.pth'}\n"
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
    
    # Debug: Print prediction details for DAD troubleshooting
    # This helps identify if predictor is seeing different inputs
    if hasattr(model, '_debug_prediction'):
        # Show multiple bound samples to check if matrix is updating
        sample1 = f"bounds[0,1]=({a_lower[0,1]:.4f},{a_upper[0,1]:.4f})"
        sample2 = f"bounds[2,3]=({a_lower[2,3]:.4f},{a_upper[2,3]:.4f})"
        sample3 = f"bounds[1,4]=({a_lower[1,4]:.4f},{a_upper[1,4]:.4f})"
        # Check if bounds matrix has any variation
        bounds_min = a_lower.min()
        bounds_max = a_upper.max()
        print(f"[predict_mocu] Normalized: {pred_normalized:.6f}, Denormalized: {mocu_pred:.6f}")
        print(f"[predict_mocu] Samples: {sample1}, {sample2}, {sample3}")
        print(f"[predict_mocu] Bounds range: [{bounds_min:.4f}, {bounds_max:.4f}]")
    
    return float(mocu_pred)
