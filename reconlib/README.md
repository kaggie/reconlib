# ReconLib: MRI Reconstruction Library

`reconlib` is a Python library for advanced MRI reconstruction, providing modules for various tasks including NUFFT, coil sensitivity estimation, image regularization, and iterative reconstruction algorithms.

## Recently Added Modules

This library has been extended with modules for:

### B0 Mapping (`reconlib.b0_mapping`)
Provides tools for estimating B0 field inhomogeneities from multi-echo MRI data.
-   `dual_echo_gre.py`: Implements PyTorch-centric B0 mapping using phase difference (dual-echo) and multi-echo linear fitting.
-   `b0_nice.py`: Placeholder for the B0-NICE algorithm (not yet implemented).
-   `utils.py`: Utilities for B0 mapping, such as mask creation.

### Phase Unwrapping (`reconlib.phase_unwrapping`)
Offers algorithms for resolving 3D phase ambiguities.
-   `quality_guided_unwrap.py`: Implements a 3D quality-guided spatial phase unwrapping algorithm (PyTorch-based).
-   `least_squares_unwrap.py`: Implements a 3D least-squares spatial phase unwrapping algorithm via an FFT-based Poisson solver (PyTorch-based).
-   `goldstein_unwrap.py`: Implements a 3D Goldstein-style spatial phase unwrapping algorithm using k-space spectral filtering (PyTorch-based).
-   `reference_echo_unwrap.py`: Implements `unwrap_multi_echo_masked_reference` for multi-echo phase unwrapping using a reference echo strategy. This function expects coil-combined input phase and magnitude for each echo and utilizes a user-provided spatial unwrapper. For a full pipeline from multi-coil data, see `preprocess_multi_coil_multi_echo_data` in `reconlib.pipeline_utils`.
-   `puror.py`: Implements `unwrap_phase_puror`, a native PyTorch Voronoi-seeded region-growing phase unwrapping algorithm.
-   `romeo.py`: Implements `unwrap_phase_romeo`, a native PyTorch region-growing phase unwrapping algorithm based on ROMEO principles.
-   `deep_learning_unwrap.py`: Implements `unwrap_phase_deep_learning`, providing a framework to use a U-Net model (defined in `reconlib.deeplearning.models.unet_denoiser.py`) for phase unwrapping with pre-trained weights.
-   `voronoi_unwrap.py`: Placeholder for `unwrap_phase_voronoi_region_growing`, a detailed Voronoi-based region-growing algorithm.
-   `residue_unwrap.py`: Placeholder for `unwrap_phase_residue_guided`, a residue/branch-cut based phase unwrapping algorithm.
-   `utils.py`: Utilities for phase unwrapping, including mask generation.

### Data Handling and Preprocessing (`reconlib.data`, `reconlib.utils`, `reconlib.io`, `reconlib.pipeline_utils`)
-   **Data Representation (`reconlib.data`):**
    -   The `MRIData` class has been enhanced to support `echo_times`.
-   **Basic Utilities (`reconlib.utils`):**
    -   Includes `extract_phase`, `extract_magnitude`, `get_echo_data`.
    -   `combine_coils_complex_sum`: Combines multi-coil complex data for a single echo via complex sum, returning the combined phase and magnitude. This is a building block for more complex pipelines.
-   **I/O (`reconlib.io`):**
    -   Contains placeholder functions for ISMRMRD and NIfTI data formats.
-   **Pipeline Utilities (`reconlib.pipeline_utils`):**
    -   `preprocess_multi_coil_multi_echo_data`: A higher-level utility that processes raw multi-coil, multi-echo complex images. It orchestrates coil combination (using `combine_coils_complex_sum`) and advanced multi-echo phase unwrapping (using `unwrap_multi_echo_masked_reference` with a user-provided spatial unwrapper) to produce masked, coil-combined, and unwrapped phase images for each echo.

### Plotting (`reconlib.plotting`)
A new module for visualizing MRI data, including:
-   Phase images (`plot_phase_image`)
-   Unwrapped phase maps (`plot_unwrapped_phase_map`)
-   B0 field maps (`plot_b0_field_map`)

### ESPIRiT Sensitivity Maps (`reconlib.csm`)
-   `estimate_espirit_maps`: Implemented natively in PyTorch for ESPIRiT coil sensitivity map estimation, including ACS data gridding, calibration matrix construction, SVD, and pixel-wise eigenvalue analysis.

### Reconstructors (`reconlib.reconstructors`)
-   `phase_aided_reconstructor.py`: Placeholder for `PhaseAidedReconstruction`, a method for phase-aided MRI image reconstruction.
-   Proximal Gradient based iterative solvers (existing, e.g. `pg_reconstructor.py`).


## Core Features (Existing)
-   **NUFFT**: Non-Uniform Fast Fourier Transform operators (`reconlib.nufft`, `reconlib.operators.NUFFTOperator`).
-   **Coil Sensitivity Maps**: Basic CSM estimation and operator (`reconlib.csm` - see above for ESPIRiT, `reconlib.operators.CoilSensitivityOperator`).
-   **Forward Operator**: Composite MRI forward operator (`reconlib.operators.MRIForwardOperator`).
-   **Regularizers**: L1, L2, Total Variation, etc. (`reconlib.regularizers`).
-   **Reconstructors**: (See above for new placeholders and existing iterative solvers).

## Examples
Example scripts and Jupyter Notebooks demonstrating the use of these modules can be found in the main `examples/` directory of the repository. New Jupyter Notebook examples include:
- Demonstrations for 3D spatial phase unwrapping algorithms (`quality_guided_unwrap_3d_example.ipynb`, `least_squares_unwrap_3d_example.ipynb`, `goldstein_unwrap_3d_example.ipynb`).
- PyTorch-based B0 mapping, including use of unwrapping functions (`b0_mapping_pytorch_example.ipynb`).
- A comprehensive pipeline for multi-coil, multi-echo data processing (`multi_coil_multi_echo_unwrapping_pipeline_example.ipynb`).
