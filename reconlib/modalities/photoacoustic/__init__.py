"""
reconlib.modalities.photoacoustic
=================================

This module provides tools for Photoacoustic Tomography (PAT) reconstruction,
including a time-of-flight based forward/adjoint operator and
TV-regularized image reconstruction algorithms.
"""

from .operators import PhotoacousticOperator
from .reconstructors import tv_reconstruction_pat
# from .utils import generate_pat_phantom, plot_pat_results # Typically not exported at this level

__all__ = [
    'PhotoacousticOperator',
    'tv_reconstruction_pat',
]
