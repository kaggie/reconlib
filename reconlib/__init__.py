"""
ReconLib: A Python library for MRI reconstruction.
"""

__version__ = "0.1.0"

from .data import MRIData
from .operators import (Operator, NUFFTOperator, CoilSensitivityOperator, 
                        MRIForwardOperator, SlidingWindowNUFFTOperator) # Added SlidingWindowNUFFTOperator
from .regularizers import (Regularizer, L1Regularizer, L2Regularizer, 
                           SparsityTransform, TVRegularizer, GradientMatchingRegularizer)
from .optimizers import Optimizer, FISTA, ADMM
from .reconstructors import (Reconstructor, IterativeReconstructor, 
                             RegriddingReconstructor, ConstrainedReconstructor)
from .utils import calculate_density_compensation
from .csm import estimate_csm_from_central_kspace

# Optionally, define __all__ to specify what `from reconlib import *` imports
__all__ = [
    'MRIData',
    'Operator', 'NUFFTOperator', 'CoilSensitivityOperator', 'MRIForwardOperator', 'SlidingWindowNUFFTOperator', # Added
    'Regularizer', 'L1Regularizer', 'L2Regularizer', 'SparsityTransform', 'TVRegularizer', 'GradientMatchingRegularizer',
    'Optimizer', 'FISTA', 'ADMM',
    'Reconstructor', 'IterativeReconstructor', 'RegriddingReconstructor', 'ConstrainedReconstructor',
    'calculate_density_compensation',
    'estimate_csm_from_central_kspace',
    '__version__'
]
