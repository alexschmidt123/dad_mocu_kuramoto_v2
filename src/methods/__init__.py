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
- DAD_MOCU: Deep Adaptive Design with MOCU objective
"""

from .base import OEDMethod
from .inn import iNN_Method
from .nn import NN_Method
from .ode import ODE_Method, iODE_Method
from .entropy import ENTROPY_Method
from .random import RANDOM_Method
from .dad_mocu import DAD_MOCU_Method

__all__ = [
    'OEDMethod',
    'iNN_Method',
    'NN_Method',
    'ODE_Method',
    'iODE_Method',
    'ENTROPY_Method',
    'RANDOM_Method',
    'DAD_MOCU_Method',
]
