"""
Unified OED Methods Package

All experimental design methods follow the OEDMethod base class interface.

Available methods:
- iNN: Iterative Neural Network (MPNN-based, re-computes at each step)
- NN: Static Neural Network (MPNN-based, computes once)
- iODE: Iterative ODE (sampling-based, re-computes at each step)
- ODE: Static ODE (sampling-based, computes once)
- ENTROPY: Greedy uncertainty-based selection
- RANDOM: Random baseline
- REGRESSION_SCORER: Regression Scorer (MPNN-based, scores designs directly, greedy selection)
- DAD_MOCU: Deep Adaptive Design with MOCU objective

NOTE: Methods are lazy-imported to avoid unnecessary PyTorch CUDA initialization.
Import methods directly from their modules:
    from src.methods.random import RANDOM_Method
    from src.methods.inn import iNN_Method
"""

from .base import OEDMethod
# NOTE: DO NOT import methods here - they trigger PyTorch import!
# Import methods lazily from their modules when needed.

__all__ = [
    'OEDMethod',
    # Methods should be imported directly from their modules
    # 'iNN_Method',  # from src.methods.inn import iNN_Method
    # 'NN_Method',   # from src.methods.nn import NN_Method
    # 'ODE_Method',  # from src.methods.ode import ODE_Method
    # 'iODE_Method', # from src.methods.ode import iODE_Method
    # 'ENTROPY_Method', # from src.methods.entropy import ENTROPY_Method
    # 'RANDOM_Method',  # from src.methods.random import RANDOM_Method
    # 'DAD_MOCU_Method', # from src.methods.dad_mocu import DAD_MOCU_Method
]
