"""
reconlib.modalities.pcct
========================

This module provides placeholder tools for Photon Counting CT (PCCT)
reconstruction. It includes a basic operator for simulating multi-energy
photon counts based on Beer-Lambert law and simplified Radon transforms,
and placeholder reconstruction algorithms.

Note: These are initial simplified placeholders. Accurate PCCT modeling and
reconstruction involve complex physics (detector response, material decomposition)
and advanced algorithms.
"""

from .operators import PCCTProjectorOperator
# Assuming simple_radon_transform and simple_back_projection are internal to operators.py for now

from .reconstructors import tv_reconstruction_pcct_mu_ref # Based on previous subtask report for reconstructors
# If there are other main reconstructors, they should be listed here.

from .utils import generate_pcct_phantom_material_maps, combine_material_maps_to_mu_ref, get_pcct_energy_scaling_factors
# plot_pcct_results is mainly for notebook use

__all__ = [
    'PCCTProjectorOperator',
    'tv_reconstruction_pcct_mu_ref',
    'generate_pcct_phantom_material_maps',
    'combine_material_maps_to_mu_ref',
    'get_pcct_energy_scaling_factors',
]
