"""
Convolutional Neural Network (CNN) Predictor for MOCU.

Baseline architecture from paper 2023.
Treats state as 3-channel image [w, a_lower, a_upper] of size N×N.

Paper: "Neural Message Passing for Objective-Based Uncertainty Quantification 
        and Optimal Experimental Design" (2023)
"""

import torch
import torch.nn as nn


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

