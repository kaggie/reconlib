"""
reconlib.modalities.hyperspectral
=================================

This module provides tools for Hyperspectral Imaging (HSI) reconstruction,
including a sensing matrix based forward/adjoint operator for compressed HSI
and 3D TV-regularized HSI cube reconstruction algorithms.
"""

from .operators import HyperspectralImagingOperator, create_sparse_sensing_matrix
from .reconstructors import tv_reconstruction_hsi

__all__ = [
    'HyperspectralImagingOperator',
    'create_sparse_sensing_matrix', # This helper is quite central to setting up the operator
    'tv_reconstruction_hsi',
]
