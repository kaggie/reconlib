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
from .material_decomposition import MaterialDecompositionForwardOperator, IterativeMaterialDecompositionReconstructor
from .projection_domain_decomposition import (
    calculate_material_thickness_sinograms,
    reconstruct_thickness_maps_from_sinograms,
    LinearRadonOperatorPlaceholder # Added here
)
# Assuming simple_radon_transform and simple_back_projection are internal to operators.py for now

from .reconstructors import tv_reconstruction_pcct_mu_ref # Based on previous subtask report for reconstructors
# If there are other main reconstructors, they should be listed here.

from .utils import (
    generate_pcct_phantom_material_maps,
    combine_material_maps_to_mu_ref,
    get_pcct_energy_scaling_factors,
    estimate_scatter_sinogram_kernel_based,
    simulate_flux_scan_for_pileup_calibration # Added new function
)
# plot_pcct_results is mainly for notebook use

__all__ = [
    'PCCTProjectorOperator',
    'MaterialDecompositionForwardOperator',
    'IterativeMaterialDecompositionReconstructor',
    'tv_reconstruction_pcct_mu_ref',
    'generate_pcct_phantom_material_maps',
    'combine_material_maps_to_mu_ref',
    'get_pcct_energy_scaling_factors',
    'estimate_scatter_sinogram_kernel_based',
    'simulate_flux_scan_for_pileup_calibration',
    'calculate_material_thickness_sinograms',
    'reconstruct_thickness_maps_from_sinograms',
    'LinearRadonOperatorPlaceholder', # Added here
]
