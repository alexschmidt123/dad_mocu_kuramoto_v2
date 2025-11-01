"""
Deep Adaptive Design Policy Network for MOCU-based Optimal Experimental Design

This module implements a policy network that learns to select optimal experiments
sequentially to minimize terminal MOCU (Model-based Objective-based Characterization of Uncertainty).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch.nn import Sequential, Linear, ReLU, GRU
import numpy as np


class DADPolicyNetwork(nn.Module):
    """
    Policy network that outputs a probability distribution over available experiment pairs.
    
    Architecture:
    1. Graph encoder: Encodes current state (w, a_lower, a_upper) using GNN
    2. History encoder: Encodes past (action, observation) pairs using LSTM
    3. Action decoder: Outputs logits for each possible (i,j) pair
    """
    
    def __init__(self, N, hidden_dim=64, encoding_dim=32, num_message_passing=3):
        """
        Args:
            N: Number of oscillators
            hidden_dim: Hidden dimension for LSTM and MLPs
            encoding_dim: Dimension for graph embeddings
            num_message_passing: Number of message passing layers
        """
        super(DADPolicyNetwork, self).__init__()
        self.N = N
        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim
        self.num_actions = N * (N - 1) // 2  # Number of unique pairs
        
        # ========== Graph State Encoder (similar to iNN) ==========
        self.lin0 = nn.Linear(1, encoding_dim)  # Node features: frequencies
        
        # Message passing with edge features [a_lower, a_upper]
        edge_nn = Sequential(
            Linear(2, 128),
            ReLU(),
            Linear(128, encoding_dim * encoding_dim)
        )
        self.conv = NNConv(encoding_dim, encoding_dim, edge_nn, aggr='mean')
        self.gru = GRU(encoding_dim, encoding_dim)
        
        # Graph-level pooling
        self.set2set = Set2Set(encoding_dim, processing_steps=3)
        self.graph_mlp = nn.Sequential(
            Linear(2 * encoding_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )
        
        # ========== History Encoder (LSTM for sequential decisions) ==========
        # Each history item: (i, j, observation) → embedding
        self.history_embed = nn.Embedding(N * N + 2, hidden_dim // 4)  # For i, j indices
        self.obs_embed = nn.Embedding(2, hidden_dim // 4)  # For sync/non-sync observation
        
        # LSTM to encode sequence of past decisions
        self.history_lstm = nn.LSTM(
            input_size=hidden_dim // 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # ========== Action Decoder ==========
        self.action_decoder = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),  # State + history
            ReLU(),
            nn.Dropout(0.1),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, self.num_actions)  # Logits for each pair
        )
    
    def encode_state(self, state_data):
        """
        Encode current state (w, a_lower, a_upper) using GNN.
        
        Args:
            state_data: PyTorch Geometric Data or Batch object
        
        Returns:
            state_embedding: [batch_size, hidden_dim]
        """
        # Ensure we're on the correct device (model's device)
        device = next(self.parameters()).device
        
        # Ensure all state_data components are on correct device BEFORE extracting
        state_data = state_data.to(device)
        
        # Node features
        x = state_data.x  # [batch_size * N, 1]
        edge_index = state_data.edge_index
        edge_attr = state_data.edge_attr  # [num_edges, 2]
        batch = state_data.batch if hasattr(state_data, 'batch') else None
        
        # Initial embedding (same pattern as MPNNPlusPredictor which works)
        out = F.relu(self.lin0(x))  # [batch_size * N, encoding_dim]
        h = out.unsqueeze(0)  # [1, batch_size * N, encoding_dim]
        
        # Message passing (same pattern as MPNNPlusPredictor)
        # Temporarily disable cuDNN to avoid stream mismatch issues
        # This is a workaround for CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH
        cudnn_enabled = torch.backends.cudnn.enabled
        try:
            # Disable cuDNN for this operation to avoid stream issues
            torch.backends.cudnn.enabled = False
            for _ in range(3):
                m = F.relu(self.conv(out, edge_index, edge_attr))
                m_input = m.unsqueeze(0)  # [1, batch_size * N, encoding_dim]
                out, h = self.gru(m_input, h)
                out = out.squeeze(0)
        finally:
            # Restore original cuDNN setting
            torch.backends.cudnn.enabled = cudnn_enabled
        
        # Graph-level pooling
        # Set2Set uses LSTM internally which can cause cuDNN stream mismatch errors
        # For single graphs (no batch), use simpler pooling to avoid cuDNN issues
        # For batched graphs, we still need Set2Set but with cuDNN disabled
        if batch is not None:
            # Batched case - use Set2Set with cuDNN disabled
            if device.type == 'cuda':
                torch.cuda.synchronize()
            cudnn_enabled = torch.backends.cudnn.enabled
            cudnn_benchmark = torch.backends.cudnn.benchmark
            try:
                torch.backends.cudnn.enabled = False
                torch.backends.cudnn.benchmark = False
                if batch.device != device:
                    batch = batch.to(device)
                if not batch.is_contiguous():
                    batch = batch.contiguous()
                if not out.is_contiguous():
                    out = out.contiguous()
                out = self.set2set(out, batch)  # [batch_size, 2 * encoding_dim]
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            finally:
                torch.backends.cudnn.enabled = cudnn_enabled
                torch.backends.cudnn.benchmark = cudnn_benchmark
        else:
            # Single graph case - use mean+max pooling instead of Set2Set to avoid cuDNN issues
            # This avoids the LSTM in Set2Set which causes stream mismatch
            batch_tensor = torch.zeros(out.size(0), dtype=torch.long, device=device)
            mean_pool = global_mean_pool(out, batch_tensor)  # [1, encoding_dim]
            max_pool = global_max_pool(out, batch_tensor)    # [1, encoding_dim]
            out = torch.cat([mean_pool, max_pool], dim=-1)    # [1, 2 * encoding_dim]
        
        state_embedding = self.graph_mlp(out)  # [batch_size, hidden_dim]
        
        return state_embedding
    
    def encode_history(self, history_data):
        """
        Encode history of past (action, observation) pairs using LSTM.
        
        Args:
            history_data: List of (i, j, obs) tuples, or batch of such lists
                Shape: [batch_size, seq_len, 3] where 3 = (i, j, obs)
        
        Returns:
            history_embedding: [batch_size, hidden_dim]
        """
        if history_data is None or len(history_data) == 0:
            # No history yet (first step)
            batch_size = 1
            device = next(self.parameters()).device
            return torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # Convert history to tensor
        if isinstance(history_data, list):
            if len(history_data) == 0:
                device = next(self.parameters()).device
                return torch.zeros(1, self.hidden_dim, device=device)
            
            # Single trajectory
            history_tensor = torch.tensor(history_data, dtype=torch.long)  # [seq_len, 3]
            history_tensor = history_tensor.unsqueeze(0)  # [1, seq_len, 3]
        else:
            history_tensor = history_data  # [batch_size, seq_len, 3]
        
        device = next(self.parameters()).device
        history_tensor = history_tensor.to(device)
        
        batch_size, seq_len, _ = history_tensor.shape
        
        # Embed i, j, obs separately
        i_indices = history_tensor[:, :, 0]  # [batch_size, seq_len]
        j_indices = history_tensor[:, :, 1]  # [batch_size, seq_len]
        obs_indices = history_tensor[:, :, 2]  # [batch_size, seq_len]
        
        i_emb = self.history_embed(i_indices)  # [batch_size, seq_len, hidden_dim//4]
        j_emb = self.history_embed(j_indices)
        obs_emb = self.obs_embed(obs_indices)  # [batch_size, seq_len, hidden_dim//4]
        
        # Concatenate embeddings
        history_emb = torch.cat([i_emb, j_emb], dim=-1)  # [batch_size, seq_len, hidden_dim//2]
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.history_lstm(history_emb)
        
        # Use final hidden state
        history_embedding = h_n[-1]  # [batch_size, hidden_dim]
        
        return history_embedding
    
    def forward(self, state_data, history_data=None, available_actions_mask=None):
        """
        Forward pass: compute action logits.
        
        Args:
            state_data: PyTorch Geometric Data/Batch for current state
            history_data: History of (i, j, obs) tuples [batch_size, seq_len, 3]
            available_actions_mask: Binary mask for available actions [batch_size, num_actions]
                                   1 = available, 0 = already observed
        
        Returns:
            action_logits: [batch_size, num_actions]
            action_probs: [batch_size, num_actions] (softmax over available actions)
        """
        # Encode current state
        state_emb = self.encode_state(state_data)  # [batch_size, hidden_dim]
        
        # Encode history
        history_emb = self.encode_history(history_data)  # [batch_size, hidden_dim]
        
        # Combine state and history
        combined = torch.cat([state_emb, history_emb], dim=-1)  # [batch_size, hidden_dim*2]
        
        # Decode to action logits
        action_logits = self.action_decoder(combined)  # [batch_size, num_actions]
        
        # Apply mask to prevent selecting already observed pairs
        if available_actions_mask is not None:
            # Set logits of unavailable actions to very negative value
            action_logits = action_logits.masked_fill(available_actions_mask == 0, -1e9)
        
        # Compute probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_logits, action_probs
    
    def select_action(self, state_data, history_data=None, available_actions_mask=None, 
                     deterministic=False):
        """
        Select an action based on the current policy.
        
        Args:
            state_data: Current state
            history_data: History of decisions
            available_actions_mask: Mask for available actions
            deterministic: If True, select argmax; if False, sample from distribution
        
        Returns:
            action_idx: Selected action index
            log_prob: Log probability of the selected action
            action_pair: (i, j) tuple corresponding to action_idx
        """
        action_logits, action_probs = self.forward(state_data, history_data, available_actions_mask)
        
        if deterministic:
            action_idx = torch.argmax(action_probs, dim=-1)
        else:
            # Sample from categorical distribution
            dist = torch.distributions.Categorical(probs=action_probs)
            action_idx = dist.sample()
        
        # Compute log probability
        log_prob = F.log_softmax(action_logits, dim=-1)
        log_prob_selected = log_prob.gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)
        
        # Convert action_idx to (i, j) pair
        action_pair = self.idx_to_pair(action_idx.item())
        
        return action_idx, log_prob_selected, action_pair
    
    def idx_to_pair(self, action_idx):
        """
        Convert action index to (i, j) oscillator pair.
        
        Maps: 0 → (0,1), 1 → (0,2), ..., to enumerate all N*(N-1)/2 pairs
        """
        # Enumerate pairs in order: (0,1), (0,2), ..., (0,N-1), (1,2), ...
        count = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if count == action_idx:
                    return (i, j)
                count += 1
        raise ValueError(f"Invalid action_idx: {action_idx}")
    
    def pair_to_idx(self, i, j):
        """Convert (i, j) pair to action index."""
        if i > j:
            i, j = j, i
        
        count = 0
        for ii in range(self.N):
            for jj in range(ii + 1, self.N):
                if ii == i and jj == j:
                    return count
                count += 1
        raise ValueError(f"Invalid pair: ({i}, {j})")


def create_state_data(w, a_lower, a_upper, device='cpu'):
    """
    Create PyTorch Geometric Data object from state.
    
    Args:
        w: Natural frequencies [N]
        a_lower: Lower bounds [N, N]
        a_upper: Upper bounds [N, N]
        device: torch device
    
    Returns:
        Data object for the state
    """
    N = len(w)
    
    # Node features: frequencies
    x = torch.from_numpy(w.astype(np.float32)).unsqueeze(-1)  # [N, 1]
    
    # Edge indices: fully connected graph
    edge_index = []
    edge_attr = []
    
    for i in range(N):
        for j in range(N):
            if i != j:
                edge_index.append([i, j])
                edge_attr.append([a_lower[i, j], a_upper[i, j]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  # [num_edges, 2]
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data = data.to(device)
    
    return data

