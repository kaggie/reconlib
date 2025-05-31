"""
reconlib.modalities.microwave
=============================

This module provides tools for Microwave Imaging (MWI) reconstruction,
including a system-matrix based forward/adjoint operator for complex dielectric properties
and TV-regularized (complex) image reconstruction algorithms.
"""

from .operators import MicrowaveImagingOperator
from .reconstructors import tv_reconstruction_mwi, ComplexTotalVariationRegularizer

__all__ = [
    'MicrowaveImagingOperator',
    'tv_reconstruction_mwi',
    'ComplexTotalVariationRegularizer', # Exporting this as it's a significant component
]
