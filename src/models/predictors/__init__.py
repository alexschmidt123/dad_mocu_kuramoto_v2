"""
MOCU Predictors Package

This package contains all neural network models for MOCU prediction.

Files:
- all_predictors.py: Unified implementations (MLP, CNN, MPNN+, Sampling, Ensemble)
- legacy_baselines.py: Original CNN/MLP from 2023 paper
- legacy_mpnn_train.py: Original MPNN training code from 2023 paper

For new projects, use all_predictors.py.
Legacy files are kept for backward compatibility with original paper experiments.
"""

from .all_predictors import (
    MLPPredictor,
    CNNPredictor,
    MPNNPlusPredictor,
    SamplingBasedMOCU,
    EnsemblePredictor,
)

__all__ = [
    'MLPPredictor',
    'CNNPredictor',
    'MPNNPlusPredictor',
    'SamplingBasedMOCU',
    'EnsemblePredictor',
]

