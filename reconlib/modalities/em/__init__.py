"""
reconlib.modalities.em
======================

This module provides tools for Electron Microscopy (EM) 3D tomographic
reconstruction, including a forward/adjoint operator (Radon transform based
on Z-axis rotations) and TV-regularized reconstruction algorithms.
"""

from .operators import EMForwardOperator
from .reconstructors import tv_reconstruction_em
# from .utils import ... # Import any core utilities if they become part of the public API

__all__ = [
    'EMForwardOperator',
    'tv_reconstruction_em',
]
