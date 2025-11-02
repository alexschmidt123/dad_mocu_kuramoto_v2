"""
Utility functions for MOCU predictors.

Shared utilities used by multiple predictor types.
"""

import torch
import numpy as np
from torch_geometric.data import Data


def get_edge_index(N):
    """
    Create edge indices for fully connected directed graph (excluding self-loops).
    
    Args:
        N: Number of nodes
    
    Returns:
        edge_index: [2, num_edges] tensor with edge connections
    """
    edge_index = []
    for i in range(N):
        for j in range(N):
            if i != j:
                edge_index.append([i, j])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()


def get_edge_attr_from_bounds(a_lower, a_upper, N):
    """
    Extract edge attributes from bound matrices.
    
    Args:
        a_lower: Lower bound matrix [N, N]
        a_upper: Upper bound matrix [N, N]
        N: Number of nodes
    
    Returns:
        edge_attr: [num_edges, 2] tensor with [a_lower, a_upper] for each edge
    """
    edge_attr = []
    for i in range(N):
        for j in range(N):
            if i != j:
                edge_attr.append([a_lower[i, j], a_upper[i, j]])
    return torch.tensor(edge_attr, dtype=torch.float32)  # [num_edges, 2]


def create_graph_data(w, a_lower, a_upper, device='cpu'):
    """
    Create PyTorch Geometric Data object from state.
    
    Args:
        w: Natural frequencies [N]
        a_lower: Lower bounds [N, N]
        a_upper: Upper bounds [N, N]
        device: torch device
    
    Returns:
        data: PyG Data object for MPNN predictors
    """
    N = len(w)
    
    # Node features
    x = torch.from_numpy(w.astype(np.float32)).unsqueeze(-1)  # [N, 1]
    
    # Edge indices and attributes
    edge_index = get_edge_index(N)
    edge_attr = get_edge_attr_from_bounds(a_lower, a_upper, N)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data = data.to(device)
    
    return data


def matrix_to_vector(x, N):
    """
    Convert N×N matrix to vector of lower triangular elements.
    
    Used for MLP input: [a_lower, a_upper] → flattened vector
    """
    x = np.tril(x, -1)  # Lower triangular
    x = x.ravel()[np.flatnonzero(x)]
    return x


def pre2R_mpnn(predictions, P_syn_list, N):
    """
    Convert MPNN predictions and probabilities to R matrix (expected remaining MOCU).
    
    Args:
        predictions: Array of predictions, pairs are [syn, non-syn] for each (i,j) pair
        P_syn_list: List of synchronization probabilities, one per (i,j) pair
        N: Number of nodes
    
    Returns:
        R_matrix: [N, N] matrix with expected remaining MOCU for each pair
    """
    R_matrix = np.zeros((N, N))
    pair_idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            # Each pair has 2 scenarios: syn (idx*2) and non-syn (idx*2+1)
            syn_idx = pair_idx * 2
            nonsyn_idx = pair_idx * 2 + 1
            
            if syn_idx < len(predictions) and nonsyn_idx < len(predictions):
                pred_syn = float(predictions[syn_idx])
                pred_nonsyn = float(predictions[nonsyn_idx])
                P_syn = float(P_syn_list[pair_idx]) if pair_idx < len(P_syn_list) else 0.5
                P_nonsyn = 1.0 - P_syn
                
                R = P_syn * pred_syn + P_nonsyn * pred_nonsyn
                R_matrix[i, j] = R
                R_matrix[j, i] = R
            pair_idx += 1
    return R_matrix

