"""
MOCU Prediction Models (Paper Table 1 Comparison).

These architectures predict MOCU values given system state (w, a_lower, a_upper).
They are components used by OED methods in methods/.

Architectures:
- MLPPredictor: Multi-Layer Perceptron baseline
- CNNPredictor: Convolutional Neural Network baseline  
- MPNNPredictor: Basic Message Passing Neural Network
- MPNNPlusPredictor: MPNN with ranking constraint (WINNER - used in iNN/NN)

Paper: "Neural Message Passing for Objective-Based Uncertainty Quantification 
        and Optimal Experimental Design" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, Set2Set
import numpy as np


class MLPPredictor(nn.Module):
    """
    Multi-Layer Perceptron for MOCU prediction.
    
    Baseline architecture from message_passing.py.
    Treats state as flattened vector.
    """
    
    def __init__(self, n_feature, n_hidden=[400, 200], n_output=1):
        """
        Args:
            n_feature: Input dimension (N + N*(N-1)/2 + N*(N-1)/2)
            n_hidden: Hidden layer sizes
            n_output: Output dimension (1 for MOCU)
        """
        super(MLPPredictor, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden[0])
        self.hidden2 = nn.Linear(n_hidden[0], n_hidden[1])
        self.predict = nn.Linear(n_hidden[1], n_output)
    
    def forward(self, x):
        """
        Args:
            x: Flattened state vector [batch, n_feature]
        
        Returns:
            mocu_pred: Predicted MOCU [batch, 1]
        """
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.predict(x)
        return x


class CNNPredictor(nn.Module):
    """
    Convolutional Neural Network for MOCU prediction.
    
    Baseline architecture from message_passing.py.
    Treats state as 3-channel image [w, a_lower, a_upper] of size N×N.
    """
    
    def __init__(self, N):
        """
        Args:
            N: Number of oscillators (determines image size)
        """
        super(CNNPredictor, self).__init__()
        self.N = N
        
        # Adaptive number of layers based on system size
        if N == 7:
            self.nlayer = 3
        else:
            self.nlayer = 2
        
        # 3 input channels: [w tiled, a_lower, a_upper]
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        
        if self.nlayer == 3:
            self.layer3 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3),
                nn.ReLU(inplace=True)
            )
        
        # Final FC layers
        self.fc = nn.Sequential(
            nn.Linear(64 * 1 * 1, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: State as image [batch, 3, N, N]
        
        Returns:
            mocu_pred: Predicted MOCU [batch, 1]
        """
        x = self.layer1(x)
        x = self.layer2(x)
        if self.nlayer == 3:
            x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MPNNPlusPredictor(nn.Module):
    """
    Message Passing Neural Network with Ranking Constraint (MP+).
    
    WINNER from Paper Table 1 - used in iNN and NN methods.
    
    Architecture:
    - Node features: natural frequencies w
    - Edge features: [a_lower, a_upper]
    - 3 layers of NNConv + GRU message passing
    - Set2Set graph-level pooling
    - Ranking constraint: enforces monotonicity w.r.t. bounds
    
    From training.py::Net
    """
    
    def __init__(self, dim=32, num_message_passing=3):
        """
        Args:
            dim: Hidden dimension
            num_message_passing: Number of message passing layers
        """
        super(MPNNPlusPredictor, self).__init__()
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
    
    @staticmethod
    def compute_ranking_loss(prediction, edge_attr, use_l2=True):
        """
        Ranking constraint loss.
        
        Enforces: 
        - MOCU should increase when a_lower increases
        - MOCU should decrease when a_upper increases
        
        This is the key innovation in MP+ over basic MP.
        
        Args:
            prediction: Predicted MOCU [batch_size]
            edge_attr: Edge features [num_edges, 2] with requires_grad=True
            use_l2: Use L2 loss (vs L1)
        
        Returns:
            ranking_loss: Constraint violation penalty
        """
        # Compute gradients w.r.t. edge attributes
        grads = torch.autograd.grad(
            outputs=prediction,
            inputs=edge_attr,
            grad_outputs=torch.ones(prediction.size()).to(prediction.device),
            create_graph=True
        )[0]
        
        # Gradients: [num_edges, 2] = [∂MOCU/∂a_lower, ∂MOCU/∂a_upper]
        lower_grads = grads[:, 0]  # Should be positive (MOCU ↑ when a_lower ↑)
        upper_grads = grads[:, 1]  # Should be negative (MOCU ↓ when a_upper ↑)
        
        # Penalize violations
        lower_violations = F.relu(-lower_grads)  # Negative gradients are bad
        upper_violations = F.relu(upper_grads)   # Positive gradients are bad
        
        if use_l2:
            ranking_loss = lower_violations.square().sum() + upper_violations.square().sum()
        else:
            ranking_loss = lower_violations.sum() + upper_violations.sum()
        
        return ranking_loss


# Utility functions

def create_graph_data(w, a_lower, a_upper, device='cpu'):
    """
    Create PyTorch Geometric Data object from state.
    
    Args:
        w: Natural frequencies [N]
        a_lower: Lower bounds [N, N]
        a_upper: Upper bounds [N, N]
        device: torch device
    
    Returns:
        data: PyG Data object for MPNNPlusPredictor
    """
    from torch_geometric.data import Data
    
    N = len(w)
    
    # Node features
    x = torch.from_numpy(w.astype(np.float32)).unsqueeze(-1)  # [N, 1]
    
    # Edge indices (fully connected directed graph, excluding self-loops)
    edge_index = []
    edge_attr = []
    
    for i in range(N):
        for j in range(N):
            if i != j:
                edge_index.append([i, j])
                edge_attr.append([a_lower[i, j], a_upper[i, j]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    
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


class SamplingBasedMOCU:
    """
    Ground truth MOCU computation using Monte Carlo sampling.
    
    This is NOT a learned predictor - it's the exact (but slow) computation
    using CUDA-accelerated integration. Used by ODE method.
    
    This is what the paper calls "Sampling-based" in Table 1.
    """
    
    def __init__(self, K_max=20480, h=1.0/160.0, T=5.0):
        """
        Args:
            K_max: Number of Monte Carlo samples
            h: Time step for RK4 integration
            T: Time horizon
        """
        from ..core.mocu_cuda import MOCU
        self.MOCU = MOCU
        self.K_max = K_max
        self.h = h
        self.M = int(T / h)
        self.T = T
    
    def compute(self, w, a_lower, a_upper, num_iterations=10):
        """
        Compute MOCU using Monte Carlo sampling (ground truth).
        
        Args:
            w: Natural frequencies [N]
            a_lower: Lower bounds [N, N]
            a_upper: Upper bounds [N, N]
            num_iterations: Number of times to compute and average
        
        Returns:
            mocu: Ground truth MOCU value
        """
        N = len(w)
        mocu_vals = np.zeros(num_iterations)
        
        for i in range(num_iterations):
            mocu_vals[i] = self.MOCU(
                self.K_max, w, N, self.h, self.M, self.T,
                a_lower, a_upper, seed=0
            )
        
        return np.mean(mocu_vals)
    
    def __call__(self, w, a_lower, a_upper):
        """Allow calling as a function."""
        return self.compute(w, a_lower, a_upper)


class EnsemblePredictor(nn.Module):
    """
    Ensemble of multiple MOCU predictors.
    
    Combines predictions from multiple models for better accuracy.
    Used in paper comparison as "Ensemble methods".
    """
    
    def __init__(self, models, weights=None):
        """
        Args:
            models: List of predictor models
            weights: Optional weights for each model (defaults to equal weighting)
        """
        super(EnsemblePredictor, self).__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
    
    def forward(self, data):
        """
        Forward pass through all models and combine.
        
        Args:
            data: Input data (format depends on model type)
        
        Returns:
            ensemble_pred: Weighted average of predictions
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(data)
            predictions.append(pred)
        
        # Weighted average
        ensemble = sum(w * p for w, p in zip(self.weights, predictions))
        
        return ensemble

