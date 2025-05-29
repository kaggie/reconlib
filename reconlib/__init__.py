"""
ReconLib: A Python library for MRI reconstruction.
"""

__version__ = "0.1.0"

from .data import MRIData
from .operators import (Operator, NUFFTOperator, CoilSensitivityOperator, 
                        MRIForwardOperator, SlidingWindowNUFFTOperator) # Added SlidingWindowNUFFTOperator
# from .regularizers import (Regularizer, L1Regularizer, L2Regularizer, 
#                            SparsityTransform, TVRegularizer, GradientMatchingRegularizer)
# Updated import for regularizers to reflect their new structure
from .regularizers.base import Regularizer
from .regularizers.common import L1Regularizer, L2Regularizer, TVRegularizer, NonnegativityConstraint, HuberRegularizer, CharbonnierRegularizer
from .regularizers.functional import SparsityTransform # Assuming functional.py holds this
# GradientMatchingRegularizer might be in common or its own file, adjust as needed.
# For now, assuming it might be in common or needs specific import if elsewhere.
# If GradientMatchingRegularizer is in common.py, L1Regularizer etc. are there too.

from .optimizers import Optimizer, FISTA, ADMM, OrderedSubsetsExpectationMaximization, PenalizedLikelihoodReconstruction
# from .reconstructors import (Reconstructor, IterativeReconstructor, 
#                              RegriddingReconstructor, ConstrainedReconstructor)
# Updated import for reconstructors from the new submodule
from .reconstructors import ProximalGradientReconstructor, POCSENSEreconstructor
# The older reconstructors might need to be added to reconlib/reconstructors/__init__.py
# or imported directly if they are at the top level of a reconstructors.py file.
# For now, only importing the new ones as specified.


from .utils import calculate_density_compensation
from .csm import estimate_csm_from_central_kspace
from .io import DICOMIO # Added DICOMIO

# New PET/CT specific modules
from .geometry import ScannerGeometry, SystemMatrix
from .projectors import ForwardProjector, BackwardProjector
from .physics import AttenuationCorrection, ScatterCorrection, DetectorResponseModel
from .pet_ct_preprocessing import normalize_counts, randoms_correction, normalize_projection_data
from .pet_ct_pipeline import ReconstructionPipeline, convergence_monitor, metrics_calculator
from .pet_ct_simulation import PhantomGenerator, simulate_projection_data


from .coil_combination import (
    coil_combination_with_phase,
    estimate_phase_maps,
    estimate_sensitivity_maps,
    reconstruct_coil_images,
    compute_density_compensation
)

# Exports from wavelets_scratch.py
from .wavelets_scratch import WaveletTransform
from .wavelets_scratch import WaveletRegularizationTerm
from .wavelets_scratch import NUFFTWaveletRegularizedReconstructor

# Export from nufft_multi_coil.py
from .nufft_multi_coil import MultiCoilNUFFTOperator

# Exports from deeplearning submodule
from .deeplearning import SimpleWaveletDenoiser, LearnedRegularizationIteration

# Exports from simulation module (toy_datasets functions)
from .simulation.toy_datasets import generate_dynamic_phantom_data, generate_nlinv_data_stubs

# Import specific plotting functions
from .plotting import (
    plot_phase_image, 
    plot_unwrapped_phase_map, 
    plot_b0_field_map,
    plot_voronoi_diagram_2d,
    plot_density_weights_2d,
    plot_voronoi_diagram_3d_slice,
    plot_density_weights_3d_slice,
    plot_projection_data,       # Added
    visualize_reconstruction    # Added
)
from .pipeline_utils import preprocess_multi_coil_multi_echo_data

# Exports from voronoi_utils.py
from .voronoi_utils import compute_polygon_area
from .voronoi_utils import compute_convex_hull_volume
from .voronoi_utils import normalize_weights


# Optionally, define __all__ to specify what `from reconlib import *` imports
__all__ = [
    'MRIData', 'DICOMIO', # Added DICOMIO
    'Operator', 'NUFFTOperator', 'CoilSensitivityOperator', 'MRIForwardOperator', 'SlidingWindowNUFFTOperator',
    'Regularizer', 'L1Regularizer', 'L2Regularizer', 'SparsityTransform', 'TVRegularizer', 'NonnegativityConstraint', 'HuberRegularizer', 'CharbonnierRegularizer', # Added NonnegativityConstraint etc.
    'Optimizer', 'FISTA', 'ADMM', 'OrderedSubsetsExpectationMaximization', 'PenalizedLikelihoodReconstruction', # Added OSEM, PLR
    # 'Reconstructor', 'IterativeReconstructor', 'RegriddingReconstructor', 'ConstrainedReconstructor', # Old reconstructors
    'ProximalGradientReconstructor', # New reconstructor
    'POCSENSEreconstructor',         # New reconstructor
    'calculate_density_compensation',
    'estimate_csm_from_central_kspace',
    '__version__',
    # Added wavelet components
    'WaveletTransform',
    'WaveletRegularizationTerm',
    'NUFFTWaveletRegularizedReconstructor',
    # Added multi-coil NUFFT operator
    'MultiCoilNUFFTOperator',
    # Added deep learning components
    'SimpleWaveletDenoiser',
    'LearnedRegularizationIteration',
    # Added simulation functions
    'generate_dynamic_phantom_data',
    'generate_nlinv_data_stubs',
    # 'plotting', # Replaced with individual function exports
    'preprocess_multi_coil_multi_echo_data',
    "coil_combination_with_phase",
    "estimate_phase_maps",
    "estimate_sensitivity_maps",
    "reconstruct_coil_images",
    "compute_density_compensation",
    # Added Voronoi utils functions
    "compute_polygon_area",
    "compute_convex_hull_volume",
    "normalize_weights",
    # Added plotting functions
    "plot_phase_image",
    "plot_unwrapped_phase_map",
    "plot_b0_field_map",
    "plot_voronoi_diagram_2d",
    "plot_density_weights_2d",
    "plot_voronoi_diagram_3d_slice",
    "plot_density_weights_3d_slice",
    "plot_projection_data",       # Added
    "visualize_reconstruction",   # Added

    # PET/CT specific additions
    'ScannerGeometry', 'SystemMatrix',
    'ForwardProjector', 'BackwardProjector',
    'AttenuationCorrection', 'ScatterCorrection', 'DetectorResponseModel',
    'normalize_counts', 'randoms_correction', 'normalize_projection_data',
    'ReconstructionPipeline', 'convergence_monitor', 'metrics_calculator',
    'PhantomGenerator', 'simulate_projection_data',
]

