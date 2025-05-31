"""
reconlib.modalities.infrared_thermography
=========================================

This module provides tools for Infrared Thermography (IRT) reconstruction,
including an iterative diffusion model based forward/adjoint operator for thermal data
and TV-regularized reconstruction of subsurface heat maps.
"""

from .operators import InfraredThermographyOperator
from .reconstructors import tv_reconstruction_irt

__all__ = [
    'InfraredThermographyOperator',
    'tv_reconstruction_irt',
]
