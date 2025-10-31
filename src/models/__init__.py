"""
Models Package

Directory structure:
- predictors/: All MOCU prediction models (import as `src.models.predictors`)
- policy_networks.py: DAD policy network

Note: Import directly from submodules:
    from src.models.policy_networks import DADPolicyNetwork, create_state_data
    from src.models.predictors import MPNNPlusPredictor
"""

# Don't import anything eagerly - let users import directly from submodules
# This avoids import errors when dependencies are missing and allows
# direct imports like: from src.models.policy_networks import DADPolicyNetwork

__all__ = [
    # Subpackages/modules are accessible via direct import
    'predictors',  # subpackage
]
