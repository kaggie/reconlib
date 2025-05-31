"""
reconlib.modalities.sim
=======================

This module provides placeholder tools for Structured Illumination Microscopy (SIM)
reconstruction. SIM is a super-resolution technique that uses patterned illumination.

Note: The current implementations are highly simplified placeholders and do not
represent a full, accurate SIM reconstruction pipeline, which is considerably
more complex, typically involving Fourier-space processing.
"""

from .operators import SIMOperator
from .reconstructors import tv_reconstruction_sim
from .utils import generate_sim_patterns, generate_sim_phantom_hr
# plot_sim_results is mainly for notebook use, not typically exported here

__all__ = [
    'SIMOperator',
    'tv_reconstruction_sim',
    'generate_sim_patterns',
    'generate_sim_phantom_hr',
]
