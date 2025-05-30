"""
reconlib.modalities.xray_phase_contrast
=======================================

This module provides tools for X-ray Phase-Contrast Imaging (XPCI)
reconstruction, including a differential phase contrast forward/adjoint
operator and TV-regularized phase reconstruction algorithms.
"""

from .operators import XRayPhaseContrastOperator
from .reconstructors import tv_reconstruction_xrpc
# Add .utils import here if utils.py for XPCI becomes non-empty and exports anything

__all__ = [
    'XRayPhaseContrastOperator',
    'tv_reconstruction_xrpc',
]
