"""
reconlib.modalities.eit
=======================

This module provides placeholder tools for Electrical Impedance Tomography (EIT)
reconstruction. It currently uses a simplified linearized model based on a
placeholder sensitivity matrix (Jacobian).

Note: These are highly simplified placeholders. Real EIT reconstruction is
a non-linear, ill-posed problem requiring sophisticated forward modeling
(e.g., FEM to compute the sensitivity matrix or solve the full equations)
and robust regularization techniques.
"""

from .operators import EITOperator
from .reconstructors import tv_reconstruction_eit
from .utils import generate_eit_phantom_delta_sigma

__all__ = [
    'EITOperator',
    'tv_reconstruction_eit',
    'generate_eit_phantom_delta_sigma',
]
