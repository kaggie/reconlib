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
-   `quality_guided_unwrap.py`: Implements a 3D quality-guided phase unwrapping algorithm (PyTorch-based).
-   `least_squares_unwrap.py`: Implements a 3D least-squares phase unwrapping algorithm via an FFT-based Poisson solver (PyTorch-based).
-   `goldstein_unwrap.py`: Implements a 3D Goldstein-style phase unwrapping algorithm using k-space spectral filtering (PyTorch-based).
-   `puror.py`: Placeholder for the PUROR algorithm (not yet implemented).
-   `romeo.py`: Placeholder for the ROMEO algorithm (not yet implemented).
-   `deep_learning_unwrap.py`: Placeholder for U-Net based phase unwrapping (not yet implemented).
-   `utils.py`: Utilities for phase unwrapping, including mask generation.

### Data Handling and I/O (`reconlib.data`, `reconlib.utils`, `reconlib.io`)
-   The `MRIData` class in `reconlib.data` has been enhanced to support `echo_times`.
-   `reconlib.utils` now includes `extract_phase`, `extract_magnitude`, and `get_echo_data`.
-   `reconlib.io` contains placeholder functions for ISMRMRD and NIfTI data formats.

### Plotting (`reconlib.plotting`)
A new module for visualizing MRI data, including:
-   Phase images (`plot_phase_image`)
-   Unwrapped phase maps (`plot_unwrapped_phase_map`)
-   B0 field maps (`plot_b0_field_map`)

### ESPIRiT Sensitivity Maps (`reconlib.csm`)
-   A placeholder function `estimate_espirit_maps` has been added for ESPIRiT coil sensitivity map estimation. The full algorithm implementation is pending.

## Core Features (Existing)
-   **NUFFT**: Non-Uniform Fast Fourier Transform operators (`reconlib.nufft`, `reconlib.operators.NUFFTOperator`).
-   **Coil Sensitivity Maps**: Basic CSM estimation and operator (`reconlib.csm`, `reconlib.operators.CoilSensitivityOperator`).
-   **Forward Operator**: Composite MRI forward operator (`reconlib.operators.MRIForwardOperator`).
-   **Regularizers**: L1, L2, Total Variation, etc. (`reconlib.regularizers`).
-   **Reconstructors**: Proximal Gradient based iterative solvers (`reconlib.reconstructors`).

## Examples
Example scripts and Jupyter Notebooks demonstrating the use of these modules can be found in the main `examples/` directory of the repository. New Jupyter Notebook examples for 3D phase unwrapping and PyTorch-based B0 mapping are now available.
