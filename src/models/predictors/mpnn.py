"""
Basic Message Passing Neural Network (MPNN) Predictor for MOCU.

Basic MPNN architecture WITHOUT ranking constraint.
This is the baseline MPNN before adding ranking constraint (which becomes MP+).

Paper: "Neural Message Passing for Objective-Based Uncertainty Quantification 
        and Optimal Experimental Design" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, Set2Set


class MPNNPredictor(nn.Module):
    """
    Basic Message Passing Neural Network for MOCU prediction.
    
    Architecture:
    - Node features: natural frequencies w
    - Edge features: [a_lower, a_upper]
    - 3 layers of NNConv + GRU message passing
    - Set2Set graph-level pooling
    - NO ranking constraint (this is the difference from MPNNPlusPredictor)
    
    This is the baseline MPNN architecture before adding ranking constraint.
    """
    
    def __init__(self, dim=32, num_message_passing=3):
        """
        Args:
            dim: Hidden dimension
            num_message_passing: Number of message passing layers
        """
        super(MPNNPredictor, self).__init__()
        self.dim = dim
        self.num_mp = num_message_passing
        
        # Node feature projection
        self.lin0 = nn.Linear(1, dim)  # w: [N, 1] → [N, dim]
        
        # Edge network for message passing
        edge_nn = Sequential(
            Linear(2, 128),  # [a_lower, a_upper]
            ReLU(),
            Linear(128, dim * dim)
        )
        self.conv = NNConv(dim, dim, edge_nn, aggr='mean')
        self.gru = GRU(dim, dim)
        
        # Graph-level readout
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = nn.Linear(2 * dim, dim)
        self.lin2 = nn.Linear(dim, 1)
    
    def forward(self, data):
        """
        Args:
            data: PyTorch Geometric Data object
                - x: Node features (frequencies) [num_nodes, 1]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, 2] = [a_lower, a_upper]
                - batch: Batch assignment (for batching multiple graphs)
        
        Returns:
            mocu_pred: Predicted MOCU [batch_size]
        """
        # Initial node embedding
        out = F.relu(self.lin0(data.x))  # [num_nodes, dim]
        h = out.unsqueeze(0)  # [1, num_nodes, dim] for GRU
        
        # Message passing layers
        for _ in range(self.num_mp):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        
        # Graph-level pooling
        out = self.set2set(out, data.batch)  # [batch_size, 2*dim]
        
        # Final prediction
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        
        return out.view(-1)  # [batch_size]

