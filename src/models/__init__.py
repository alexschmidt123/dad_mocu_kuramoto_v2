"""
Models Package

Directory structure:
- predictors/: All MOCU prediction models
  - all_predictors.py: Unified implementations (recommended)
  - legacy_baselines.py: Original CNN/MLP (backward compatibility)
  - legacy_mpnn_train.py: Original MPNN training (backward compatibility)
  
- policy_networks.py: DAD policy network

For new projects, import from predictors package.
"""

# MOCU Predictors (recommended)
from .predictors import (
    MLPPredictor,
    CNNPredictor,
    MPNNPlusPredictor,
    SamplingBasedMOCU,
    EnsemblePredictor,
)

# Policy Networks
from .policy_networks import DADPolicyNetwork

__all__ = [
    # MOCU Predictors
    'MLPPredictor',
    'CNNPredictor',
    'MPNNPlusPredictor',
    'SamplingBasedMOCU',
    'EnsemblePredictor',
    # Policy Networks
    'DADPolicyNetwork',
]
