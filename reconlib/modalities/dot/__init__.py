"""
reconlib.modalities.dot
=======================

This module provides placeholder tools for Diffuse Optical Tomography (DOT)
reconstruction. It currently uses a simplified linearized model based on a
placeholder sensitivity matrix (Jacobian) to relate internal optical property
changes to boundary measurements.

Note: These are highly simplified placeholders. Real DOT reconstruction is
a non-linear, highly ill-posed problem requiring sophisticated forward modeling
(e.g., solving the Diffusion Equation with FEM) and robust regularization.
"""

from .operators import DOTOperator
from .reconstructors import tv_reconstruction_dot
from .utils import generate_dot_phantom_delta_mu

__all__ = [
    'DOTOperator',
    'tv_reconstruction_dot',
    'generate_dot_phantom_delta_mu',
]
