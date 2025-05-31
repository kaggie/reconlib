"""
reconlib.modalities.xray_diffraction
====================================

This module provides placeholder tools for X-ray Diffraction Imaging,
focusing on the phase retrieval problem. It includes a basic operator
that models far-field diffraction and magnitude detection, and a
simplified iterative phase retrieval algorithm.

Note: These are highly simplified placeholders. Real phase retrieval
is a complex field requiring more sophisticated algorithms and constraints.
"""

from .operators import XRayDiffractionOperator
from .reconstructors import basic_phase_retrieval_gs
from .utils import generate_xrd_phantom
# plot_xrd_results is mainly for notebook use

__all__ = [
    'XRayDiffractionOperator',
    'basic_phase_retrieval_gs',
    'generate_xrd_phantom',
]
