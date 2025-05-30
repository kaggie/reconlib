"""
reconlib.modalities.sar
=======================

This module provides tools for Synthetic Aperture Radar (SAR)
image reconstruction, including a Fourier-based forward/adjoint operator
and TV-regularized reconstruction algorithms.
"""

from .operators import SARForwardOperator
from .reconstructors import tv_reconstruction_sar
# Add .utils import here if utils.py for SAR becomes non-empty and exports anything

__all__ = [
    'SARForwardOperator',
    'tv_reconstruction_sar',
]
