# reconlib.regularizers module

from .base import Regularizer

from .common import (
    L1Regularizer,
    L2Regularizer,
    TVRegularizer
    # HuberRegularizer and CharbonnierRegularizer will be added here later
)

# Assuming the refactored reconlib/regularizers.py still exists with these classes
from .regularizers import (
    SparsityTransform,
    GradientMatchingRegularizer
)

from .functional import (
    l1_norm,
    l2_norm_squared,
    total_variation,
    charbonnier_penalty,
    huber_penalty
)

__all__ = [
    'Regularizer',
    'L1Regularizer',
    'L2Regularizer',
    'TVRegularizer',
    # 'HuberRegularizer', # To be added
    # 'CharbonnierRegularizer', # To be added
    'SparsityTransform',
    'GradientMatchingRegularizer',
    'l1_norm',
    'l2_norm_squared',
    'total_variation',
    'charbonnier_penalty',
    'huber_penalty'
]
