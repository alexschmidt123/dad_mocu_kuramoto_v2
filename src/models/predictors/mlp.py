"""
Multi-Layer Perceptron (MLP) Predictor for MOCU.

Baseline architecture from paper 2023.
Treats state as flattened vector.

Paper: "Neural Message Passing for Objective-Based Uncertainty Quantification 
        and Optimal Experimental Design" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

