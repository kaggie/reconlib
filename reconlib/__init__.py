"""
ReconLib: A Python library for MRI reconstruction.
"""

__version__ = "0.1.0"

from .data import MRIData
from .operators import Operator, NUFFTOperator, CoilSensitivityOperator, MRIForwardOperator
from .regularizers import Regularizer, L1Regularizer, L2Regularizer, SparsityTransform
from .optimizers import Optimizer, FISTA
from .reconstructors import Reconstructor, IterativeReconstructor
from .utils import calculate_density_compensation

# Optionally, define __all__ to specify what `from reconlib import *` imports
__all__ = [
    'MRIData',
    'Operator', 'NUFFTOperator', 'CoilSensitivityOperator', 'MRIForwardOperator',
    'Regularizer', 'L1Regularizer', 'L2Regularizer', 'SparsityTransform',
    'Optimizer', 'FISTA',
    'Reconstructor', 'IterativeReconstructor',
    'calculate_density_compensation',
    '__version__'
]
