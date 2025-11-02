"""
MOCU Predictors Package

This package contains all neural network models for MOCU prediction.

Files:
- predictors.py: Unified implementations (MLP, CNN, MPNN+, Sampling, Ensemble)
- legacy_baselines.py: Original CNN/MLP from 2023 paper
- legacy_mpnn.py: Original MPNN training code from 2023 paper

For new projects, use predictors.py.
Legacy files are kept for backward compatibility with original paper experiments.
"""

# Import only MPNN-related predictors by default (most common use case)
# This avoids importing SamplingBasedMOCU which would trigger mocu_cuda import
from .predictors import (
    MLPPredictor,
    CNNPredictor,
    MPNNPlusPredictor,
    EnsemblePredictor,
)

# SamplingBasedMOCU is NOT imported here to avoid PyCUDA context initialization
# It's only needed by ODE method and compare_predictors.py
# Those scripts should import it directly when needed:
#   from src.models.predictors.predictors import SamplingBasedMOCU

__all__ = [
    'MLPPredictor',
    'CNNPredictor',
    'MPNNPlusPredictor',
    'EnsemblePredictor',
    # SamplingBasedMOCU removed from __all__ - import directly from predictors.py when needed
]

