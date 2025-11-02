"""
MOCU Predictors Package

This package contains all neural network models for MOCU prediction from paper 2023.

Predictors (6 total):
- MLPPredictor: Multi-Layer Perceptron baseline
- CNNPredictor: Convolutional Neural Network baseline
- MPNNPredictor: Basic Message Passing Neural Network
- MPNNPlusPredictor: MPNN with ranking constraint (WINNER - used in iNN/NN)
- EnsemblePredictor: Ensemble of multiple predictors
- SamplingBasedMOCU: Ground truth MOCU computation (separate file to avoid PyCUDA import)

Files:
- mlp.py: MLP predictor
- cnn.py: CNN predictor
- mpnn.py: Basic MPNN predictor
- mpnn_plus.py: MPNN+ predictor (winner)
- ensemble.py: Ensemble predictor
- sampling_mocu.py: Sampling-based ground truth (separate to avoid PyCUDA)
- utils.py: Shared utility functions
- predictor_utils.py: Utilities for loading/using MPNN predictor

Paper: "Neural Message Passing for Objective-Based Uncertainty Quantification 
        and Optimal Experimental Design" (2023)
"""

# Import all predictors from individual files
from .mlp import MLPPredictor
from .cnn import CNNPredictor
from .mpnn import MPNNPredictor
from .mpnn_plus import MPNNPlusPredictor
from .ensemble import EnsemblePredictor

# Import utilities
from .utils import (
    get_edge_index,
    get_edge_attr_from_bounds,
    create_graph_data,
    matrix_to_vector,
    pre2R_mpnn,
)

# SamplingBasedMOCU is NOT imported here to avoid PyCUDA context initialization
# It's only needed by compare_predictors.py for evaluation
# Import it from the separate file when needed:
#   from src.models.predictors.sampling_mocu import SamplingBasedMOCU
# 
# NOTE: ODE method uses MOCU() directly from mocu_cuda.py, NOT SamplingBasedMOCU

__all__ = [
    # Predictors
    'MLPPredictor',
    'CNNPredictor',
    'MPNNPredictor',
    'MPNNPlusPredictor',
    'EnsemblePredictor',
    # Utilities
    'get_edge_index',
    'get_edge_attr_from_bounds',
    'create_graph_data',
    'matrix_to_vector',
    'pre2R_mpnn',
]
