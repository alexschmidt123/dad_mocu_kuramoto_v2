"""
Ensemble Predictor for MOCU.

Combines predictions from multiple models for better accuracy.
Used in paper comparison as "Ensemble methods".

Paper: "Neural Message Passing for Objective-Based Uncertainty Quantification 
        and Optimal Experimental Design" (2023)
"""

import torch
import torch.nn as nn


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

