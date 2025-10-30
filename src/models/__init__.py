"""
Models Package

Directory structure:
- predictors/: All MOCU prediction models (import as `src.models.predictors`)
- policy_networks.py: DAD policy network

Note: Avoid eager imports of heavy predictor modules at package import time
to prevent unnecessary dependencies when only policy networks are needed.
"""

# Export policy networks directly
from .policy_networks import DADPolicyNetwork

# Expose subpackages without importing them eagerly
__all__ = [
    'DADPolicyNetwork',
    'predictors',  # subpackage
]
