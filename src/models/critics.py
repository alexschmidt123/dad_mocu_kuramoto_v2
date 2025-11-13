"""
MOCU Critic Network for Variance Reduction in REINFORCE

Inspired by iDAD's critic network approach for implicit models.
The critic estimates MOCU given state and history, providing a baseline
to reduce variance in REINFORCE policy gradient updates.

Enhanced with pre-trained MPNN predictor:
- Uses existing trained MPNN model for accurate state-based MOCU estimation
- Combines MPNN prediction with history encoding for context-aware adjustment
- No need to train MPNN again - reuses existing trained model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch.nn import Sequential, Linear, ReLU
import numpy as np


class MOCUCritic(nn.Module):
    """
    Critic network that estimates terminal MOCU given current state and history.
    
    Used as a baseline in REINFORCE to reduce variance:
    advantage = reward - critic(state, history)
    
    Architecture:
    1. Pre-trained MPNN predictor: Gets base MOCU estimate from state (no training needed)
    2. History encoder: Set-equivariant sum (from iDAD) or LSTM
    3. Adjustment head: MLP that adjusts MPNN prediction based on history
    
    Key advantage: Uses existing trained MPNN model on-the-fly, no retraining needed.
    The critic learns to adjust MPNN predictions based on history context.
    """
    
    def __init__(self, N, hidden_dim=256, encoding_dim=16, 
                 use_set_equivariant=True, mpnn_model=None, mpnn_mean=None, mpnn_std=None):
        """
        Args:
            N: Number of oscillators
            hidden_dim: Hidden dimension for MLPs
            encoding_dim: Dimension for history encodings
            use_set_equivariant: If True, use set-equivariant history encoder (iDAD style)
                                If False, use LSTM (current approach)
            mpnn_model: Pre-trained MPNNPlusPredictor model (will be used in eval mode)
            mpnn_mean: Normalization mean for MPNN predictions
            mpnn_std: Normalization std for MPNN predictions
        """
        super(MOCUCritic, self).__init__()
        self.N = N
        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim
        self.use_set_equivariant = use_set_equivariant
        
        # Store pre-trained MPNN model (frozen, eval mode)
        self.mpnn_model = mpnn_model
        self.mpnn_mean = mpnn_mean
        self.mpnn_std = mpnn_std
        
        if self.mpnn_model is not None:
            # Freeze MPNN model - we don't train it, just use it
            for param in self.mpnn_model.parameters():
                param.requires_grad = False
            self.mpnn_model.eval()
        
        # ========== History Encoder ==========
        if use_set_equivariant:
            # Set-equivariant: sum of independent encodings (iDAD style)
            self.history_embed = nn.Embedding(N * N + 2, encoding_dim // 2)  # For i, j indices
            self.obs_embed = nn.Embedding(2, encoding_dim // 2)  # For sync/non-sync
            self.pair_encoder = nn.Sequential(
                Linear(encoding_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, encoding_dim)
            )
        else:
            # LSTM-based (current approach)
            self.history_embed = nn.Embedding(N * N + 2, hidden_dim // 4)
            self.obs_embed = nn.Embedding(2, hidden_dim // 4)
            from torch.nn import LSTM
            self.history_lstm = LSTM(
                input_size=hidden_dim // 2,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.1
            )
        
        # ========== History Projection (for set-equivariant) ==========
        if use_set_equivariant:
            # Project history encoding to hidden_dim
            self.history_proj = nn.Linear(encoding_dim, hidden_dim)
        else:
            self.history_proj = nn.Identity()
        
        # ========== Adjustment Head ==========
        if mpnn_model is not None:
            # Takes MPNN prediction + history, outputs adjusted MOCU estimate
            # Input: [mpnn_pred (1) + history_emb (hidden_dim)] = [hidden_dim + 1]
            self.adjustment_head = nn.Sequential(
                Linear(hidden_dim + 1, hidden_dim),  # MPNN pred (scalar) + history embedding
                ReLU(),
                nn.Dropout(0.1),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, 1)  # Adjustment to MPNN prediction (scalar)
            )
        else:
            # Fallback: No MPNN, use history-only prediction
            # Input: [history_emb (hidden_dim)]
            self.adjustment_head = nn.Sequential(
                Linear(hidden_dim, hidden_dim),  # History embedding only
                ReLU(),
                nn.Dropout(0.1),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, 1)  # MOCU estimate (scalar)
            )
    
    def get_mpnn_prediction(self, state_data):
        """
        Get MOCU prediction from pre-trained MPNN model.
        
        Args:
            state_data: PyTorch Geometric Data/Batch for current state
        
        Returns:
            mpnn_pred: [batch_size] MOCU predictions from MPNN
        """
        if self.mpnn_model is None:
            raise ValueError("MPNN model not provided. Cannot get MPNN prediction.")
        
        device = next(self.parameters()).device
        self.mpnn_model.eval()  # Ensure eval mode
        
        # Ensure state_data is on correct device
        state_data = state_data.to(device)
        
        # Ensure MPNN model is on correct device
        if next(self.mpnn_model.parameters()).device != device:
            self.mpnn_model = self.mpnn_model.to(device)
        
        # Predict with MPNN (no gradients - MPNN is frozen)
        with torch.no_grad():
            pred_normalized = self.mpnn_model(state_data)  # [batch_size]
            
            # Denormalize if statistics provided
            if self.mpnn_mean is not None and self.mpnn_std is not None:
                # Ensure mean/std are on correct device and have right shape
                mean = self.mpnn_mean.to(device) if isinstance(self.mpnn_mean, torch.Tensor) else torch.tensor(self.mpnn_mean, device=device)
                std = self.mpnn_std.to(device) if isinstance(self.mpnn_std, torch.Tensor) else torch.tensor(self.mpnn_std, device=device)
                mpnn_pred = pred_normalized * std + mean
            else:
                mpnn_pred = pred_normalized
        
        return mpnn_pred  # [batch_size]
    
    def encode_history(self, history_data):
        """Encode history using set-equivariant sum or LSTM."""
        if history_data is None or len(history_data) == 0:
            batch_size = 1
            device = next(self.parameters()).device
            if self.use_set_equivariant:
                return torch.zeros(batch_size, self.encoding_dim, device=device)
            else:
                return torch.zeros(batch_size, self.hidden_dim, device=device)
        
        device = next(self.parameters()).device
        
        if self.use_set_equivariant:
            # Set-equivariant: encode each pair independently, then sum
            if isinstance(history_data, list):
                if len(history_data) == 0:
                    return torch.zeros(1, self.encoding_dim, device=device)
                
                history_tensor = torch.tensor(history_data, dtype=torch.long, device=device)
            else:
                history_tensor = history_data.to(device)
            
            # history_tensor: [batch_size, seq_len, 3] where 3 = (i, j, obs)
            batch_size, seq_len, _ = history_tensor.shape
            
            i_indices = history_tensor[:, :, 0]
            j_indices = history_tensor[:, :, 1]
            obs_indices = history_tensor[:, :, 2]
            
            i_emb = self.history_embed(i_indices)  # [batch, seq, encoding_dim//2]
            j_emb = self.history_embed(j_indices)
            obs_emb = self.obs_embed(obs_indices)
            
            # Concatenate: [batch, seq, encoding_dim]
            pair_emb = torch.cat([i_emb, j_emb], dim=-1)
            
            # Encode each pair
            pair_encodings = []
            for t in range(seq_len):
                pair_enc = self.pair_encoder(pair_emb[:, t, :])  # [batch, encoding_dim]
                pair_encodings.append(pair_enc)
            
            # Sum (set-equivariant operation)
            sum_encoding = sum(pair_encodings)  # [batch, encoding_dim]
            return sum_encoding
        
        else:
            # LSTM-based (current approach)
            if isinstance(history_data, list):
                if len(history_data) == 0:
                    return torch.zeros(1, self.hidden_dim, device=device)
                history_tensor = torch.tensor(history_data, dtype=torch.long, device=device)
            else:
                history_tensor = history_data
            
            history_tensor = history_tensor.to(device)
            batch_size, seq_len, _ = history_tensor.shape
            
            i_indices = history_tensor[:, :, 0]
            j_indices = history_tensor[:, :, 1]
            obs_indices = history_tensor[:, :, 2]
            
            i_emb = self.history_embed(i_indices)
            j_emb = self.history_embed(j_indices)
            obs_emb = self.obs_embed(obs_indices)
            
            history_emb = torch.cat([i_emb, j_emb], dim=-1)
            
            if not history_emb.is_contiguous():
                history_emb = history_emb.contiguous()
            
            lstm_out, (h_n, c_n) = self.history_lstm(history_emb)
            history_embedding = h_n[-1]  # [batch, hidden_dim]
            return history_embedding
    
    def forward(self, state_data, history_data=None):
        """
        Estimate terminal MOCU given state and history.
        
        Uses pre-trained MPNN for base MOCU prediction, then adjusts based on history.
        Falls back to history-only prediction if MPNN is not available.
        
        Args:
            state_data: PyTorch Geometric Data/Batch for current state
            history_data: History of (i, j, obs) tuples [batch_size, seq_len, 3]
        
        Returns:
            mocu_estimate: [batch_size] MOCU estimate (scalar per sample)
        """
        # Encode history
        history_emb = self.encode_history(history_data)  # [batch, encoding_dim or hidden_dim]
        
        # Project history to hidden_dim if needed (for set-equivariant)
        history_emb = self.history_proj(history_emb)  # [batch, hidden_dim]
        
        if self.mpnn_model is not None:
            # Get base MOCU prediction from pre-trained MPNN
            mpnn_pred = self.get_mpnn_prediction(state_data)  # [batch_size]
            
            # Combine MPNN prediction with history
            # mpnn_pred: [batch_size], history_emb: [batch, hidden_dim]
            mpnn_pred_expanded = mpnn_pred.unsqueeze(-1)  # [batch, 1]
            combined = torch.cat([mpnn_pred_expanded, history_emb], dim=-1)  # [batch, hidden_dim + 1]
            
            # Get adjustment to MPNN prediction based on history
            adjustment = self.adjustment_head(combined)  # [batch, 1]
            
            # Final estimate: MPNN prediction + adjustment
            mocu_estimate = mpnn_pred + adjustment.squeeze(-1)  # [batch]
        else:
            # Fallback: Use history-only prediction
            mocu_estimate = self.adjustment_head(history_emb).squeeze(-1)  # [batch]
        
        return mocu_estimate  # [batch]

