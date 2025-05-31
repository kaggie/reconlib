# ReconLib Overview

## 1. Introduction
`reconlib` is a comprehensive Python library for advanced image reconstruction, particularly focused on medical imaging modalities. It provides a modular framework encompassing operators, solvers, regularizers, NUFFT implementations, deep learning tools, Voronoi-based methods, and specific utilities for various imaging techniques. The library is designed to support research and development in tomographic reconstruction by offering a collection of reusable components and example workflows.

## 2. Core Modules
This section details the core, modality-agnostic modules of `reconlib`.

### 2.1. `reconlib.operators`
This module defines the abstract `Operator` class and various specific linear operators used in reconstruction.
- `Operator`: Abstract Base Class for linear operators, defining `op` (forward) and `op_adj` (adjoint) methods.
- `NUFFTOperator`: An operator for Non-Uniform Fast Fourier Transform (NUFFT) operations, often used in MRI.
- `CoilSensitivityOperator`: Operator to incorporate coil sensitivity maps in MRI.
- `MRIForwardOperator`: A comprehensive forward operator for MRI, potentially combining NUFFT and coil sensitivities.
- `SlidingWindowNUFFTOperator`: NUFFT operator for sliding window reconstructions (e.g., dynamic MRI).
- `IRadon`: Operator for the inverse Radon transform (e.g., for CT FBP).
- `PETForwardProjection`: Forward projection operator for Positron Emission Tomography.
- `PATForwardProjection`: Forward projection operator for Photoacoustic Tomography.
- `RadioInterferometryOperator`: Operator for radio interferometry in astronomical imaging.
- `FieldCorrectedNUFFTOperator`: NUFFT operator that includes B0 field inhomogeneity corrections.

### 2.2. `reconlib.nufft`
Provides implementations of the Non-Uniform Fast Fourier Transform.
- `NUFFT`: Abstract Base Class for NUFFT implementations.
- `NUFFT2D`: 2D NUFFT implementation.
- `NUFFT3D`: 3D NUFFT implementation.

### 2.3. `reconlib.nufft_multi_coil`
Handles NUFFT operations for multi-coil MRI data.
- `MultiCoilNUFFTOperator`: Combines NUFFT with multi-coil sensitivity information.

### 2.4. `reconlib.solvers`
Contains iterative algorithms to solve inverse problems in image reconstruction.
- `iterative_reconstruction`: A generic framework for iterative reconstruction algorithms.
- `fista_reconstruction`: Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).
- `admm_reconstruction`: Alternating Direction Method of Multipliers (ADMM).
- `conjugate_gradient_reconstruction`: Conjugate Gradient method, often for solving normal equations.

### 2.5. `reconlib.reconstructors`
Defines base classes and specific implementations for image reconstruction pipelines.
- `Reconstructor` (ABC): Abstract Base Class for reconstructors.
- `IterativeReconstructor`: Base class for iterative reconstruction algorithms.
- `RegriddingReconstructor`: Reconstructor based on gridding k-space data followed by FFT.
- `ConstrainedReconstructor`: Base for reconstructors incorporating constraints.
- `phase_aided_reconstructor.PhaseAidedReconstructor`: Reconstructor utilizing phase information. (from `reconlib/reconstructors/phase_aided_reconstructor.py`)
- `pocsense_reconstructor.POCSENSE`: Implementation of POCSENSE algorithm. (from `reconlib/reconstructors/pocsense_reconstructor.py`)
- `proximal_gradient_reconstructor.ProximalGradientReconstructor`: Reconstructor based on proximal gradient methods. (from `reconlib/reconstructors/proximal_gradient_reconstructor.py`)


### 2.6. `reconlib.regularizers`
Provides various regularization terms to incorporate prior knowledge and improve reconstruction quality.
- **`reconlib.regularizers.base`**
  - `Regularizer` (ABC): Abstract Base Class for regularizers.
- **`reconlib.regularizers.common`**
  - `L1Regularizer`: L1 norm penalty (promotes sparsity).
  - `L2Regularizer`: L2 norm penalty (ridge regression).
  - `TVRegularizer`: Total Variation penalty (promotes piecewise constant solutions).
  - `HuberRegularizer`: Huber penalty, combining L1 and L2 characteristics.
  - `CharbonnierRegularizer`: Charbonnier penalty, a smooth L1-like penalty.
  - `NonnegativityConstraint`: Enforces non-negativity.
- **`reconlib.regularizers.functional`**
  - `l1_norm`: Computes L1 norm.
  - `l2_norm_squared`: Computes L2 norm squared.
  - `total_variation`: Computes Total Variation.
  - `charbonnier_penalty`: Computes Charbonnier penalty value.
  - `huber_penalty`: Computes Huber penalty value.

### 2.7. `reconlib.wavelets_scratch`
Implementation of wavelet transforms and wavelet-based regularization.
- `WaveletTransform`: Performs wavelet decomposition and reconstruction.
- `WaveletRegularizationTerm`: A regularizer based on wavelet domain sparsity.
- `NUFFTWaveletRegularizedReconstructor`: Reconstructor combining NUFFT with wavelet regularization.

### 2.8. `reconlib.data`
Data structures for handling imaging data.
- `MRIData`: A class to encapsulate MRI k-space data, trajectory, and associated parameters.

### 2.9. `reconlib.csm`
Coil Sensitivity Map (CSM) estimation methods for MRI.
- `estimate_csm_from_central_kspace`: Estimates CSMs from the central region of k-space.
- `estimate_espirit_maps`: Estimates CSMs using ESPIRiT algorithm.

### 2.10. `reconlib.coil_combination`
Methods for combining data from multiple receiver coils in MRI.
- `coil_combination_with_phase`: Coil combination considering phase information.
- `estimate_phase_maps`: Estimates phase maps from multi-coil data.
- `estimate_sensitivity_maps`: Placeholder for sensitivity map estimation.
- `reconstruct_coil_images`: Placeholder for reconstructing individual coil images.
- `compute_density_compensation`: Placeholder for Voronoi-based density compensation (noted as not fully implemented here).

### 2.11. `reconlib.geometry`
Utilities for defining scanner geometry and system matrices.
- `ScannerGeometry`: Class to define scanner geometric parameters.
- `SystemMatrix`: Class to represent system matrices (e.g., for PET, SPECT).

### 2.12. `reconlib.projectors`
Forward and backward projection operators.
- `ForwardProjector`: Generic forward projector class.
- `BackwardProjector`: Generic backward projector class.

### 2.13. `reconlib.physics`
Modules for modeling physical effects in imaging.
- **`reconlib.physics.models`**
    - `AttenuationCorrection`: Placeholder for attenuation correction methods.
    - `ScatterCorrection`: Placeholder for scatter correction methods.
    - `DetectorResponseModel`: Placeholder for detector response modeling.

### 2.14. `reconlib.io`
Input/output utilities for various imaging data formats.
- Placeholder functions for ISMRMRD and NIfTI formats.
- `DICOMIO`: Class with placeholder methods for DICOM reading/writing.

### 2.15. `reconlib.metrics`
Metrics for evaluating image quality.
- **`reconlib.metrics.image_metrics`**
  - `mse`: Mean Squared Error.
  - `psnr`: Peak Signal-to-Noise Ratio.
  - `ssim`: Structural Similarity Index.

### 2.16. `reconlib.pipeline_utils`
Utilities for image reconstruction pipelines.
- `preprocess_multi_coil_multi_echo_data`: Preprocessing for multi-coil, multi-echo MRI data.

### 2.17. `reconlib.simulation_utils`
Utilities for simulating imaging data. (Note: `reconlib.simulation.toy_datasets.py` exists)
 - **`reconlib.simulation.toy_datasets`**
    - `create_shepp_logan_phantom`
    - `simulate_kspace_data`


### 2.18. `reconlib.utils`
General utility functions used across the library.
- `extract_phase`, `extract_magnitude`: For complex data.
- `get_echo_data`: Utility for multi-echo data.
- `calculate_density_compensation`: General density compensation.
- `combine_coils_complex_sum`: Simple coil combination.

### 2.19. `reconlib.plotting`
Plotting utilities for visualizing images, k-space data, etc.
- (Summarize key plotting functions if details are available from previous steps, e.g., `plot_image_grid`, `plot_sinogram`).
- `plot_image_callbacks`, `plot_convergence_curve`, `plot_phase_unwrap_comparison`, etc.

## 3. Deep Learning Submodule (`reconlib.deeplearning`)
This submodule provides tools and models for deep learning-based image reconstruction.
- **`reconlib.deeplearning.datasets`**
  - `MoDLDataset`: PyTorch Dataset class for Model-Based Deep Learning (MoDL).
- **`reconlib.deeplearning.denoisers`**
  - `SimpleWaveletDenoiser`: A denoiser using wavelet thresholding.
- **`reconlib.deeplearning.layers`**
  - **`reconlib.deeplearning.layers.data_consistency`**
    - `DataConsistencyMoDL`: Data consistency layer for MoDL networks.
- **`reconlib.deeplearning.losses`**
  - **`reconlib.deeplearning.losses.common_losses`**
    - (Currently empty, but intended for common loss functions like MSE, L1).
- **`reconlib.deeplearning.models`**
  - `MoDLNet`: Implementation of the MoDL reconstruction network.
  - `SimpleResNetDenoiser`: ResNet-based denoiser model.
  - `UNet`: UNet architecture, often used for image-to-image tasks including denoising/reconstruction.
- **`reconlib.deeplearning.unrolled`**
  - `LearnedRegularizationIteration`: Represents an iteration of an unrolled optimization algorithm with learned components.
- **`reconlib.deeplearning.utils.data_utils`**
  - `generate_cartesian_undersampling_mask`: Generates k-space undersampling masks.
  - `apply_cartesian_mask`: Applies undersampling masks to k-space data.
  - `get_zero_filled_reconstruction`: Performs zero-filled inverse FFT.

## 4. Voronoi Submodule (`reconlib.voronoi`)
This submodule provides tools for computing and analyzing Voronoi diagrams and Delaunay triangulations, primarily for use in defining non-pixel-based image representations in reconstruction.
Key features include:
- Core geometric primitives (convex hull, clipping, predicates).
- 2D and 3D Delaunay triangulation/tetrahedralization.
- Construction of Voronoi diagrams from Delaunay duals.
- Analysis functions: density weights, centroids, neighbors, shape descriptors.
- Plotting utilities for 2D and 3D Voronoi diagrams.
- Raster-based Voronoi tessellation.
- Seed selection and region growing algorithms.
- **Applications in Medical Image Reconstruction**:
    - **CT:** `VoronoiCTReconstructor2D` using SART.
    - **SPECT:** `VoronoiSPECTReconstructor2D` using OSEM.
    - **PCCT:** `VoronoiPCCTReconstructor2D` for material decomposition.

- **Key files and components:**
  - **`reconlib.voronoi.geometry_core`**: `EPSILON`, `ConvexHull`, clipping functions, geometric predicates.
  - **`reconlib.voronoi.delaunay_2d`**: `delaunay_triangulation_2d`, `get_triangle_circumcircle_details_2d`, `is_point_in_circumcircle`.
  - **`reconlib.voronoi.delaunay_3d`**: `delaunay_triangulation_3d`.
  - **`reconlib.voronoi.voronoi_from_delaunay`**: `construct_voronoi_polygons_2d`, `construct_voronoi_polyhedra_3d`.
  - **`reconlib.voronoi.density_weights_pytorch`**: `compute_voronoi_density_weights_pytorch`.
  - **`reconlib.voronoi.circumcenter_calculations`**: `compute_triangle_circumcenter_2d`, `compute_tetrahedron_circumcenter_3d`.
  - **`reconlib.voronoi.voronoi_tessellation`**: `compute_voronoi_tessellation`.
  - **`reconlib.voronoi.voronoi_analysis`**: `compute_voronoi_density_weights`, `compute_cell_centroid`, `get_cell_neighbors`, shape descriptors.
  - **`reconlib.voronoi.voronoi_plotting`**: `plot_voronoi_diagram_2d`, `plot_voronoi_wireframe_3d`.
  - **`reconlib.voronoi.seed_selection`**: `select_voronoi_seeds`.
  - **`reconlib.voronoi.region_growing`**: `voronoi_region_growing`.
  - **`reconlib.voronoi.cell_merging_paths`**: Conceptual `merge_voronoi_cells_and_optimize_paths`.
  - **`reconlib.voronoi.find_nearest_seed`**: `find_nearest_seed`.

## 5. Modalities (`reconlib.modalities`)
This section covers modules tailored for specific imaging modalities. Each submodule typically contains modality-specific operators, reconstructors, and utility functions.

- **`reconlib.modalities.astronomical`**: For astronomical imaging.
  - `operators.RadioInterferometryOperator`
  - `reconstructors.AstronomicalReconstructor` (placeholder)
  - `astro_reconstruction_notebook.ipynb`: Example workflow.
- **`reconlib.modalities.ct`**: Computed Tomography.
  - `operators.CTProjectorOperator`: Basic Radon transform and back-projection.
  - `voronoi_reconstructor.VoronoiCTReconstructor2D`: SART-based Voronoi reconstruction.
  - `preprocessing.CTPreprocessing`: (Placeholder)
- **`reconlib.modalities.dot`**: Diffuse Optical Tomography.
  - `operators.DOTForwardOperator`
  - `reconstructors.DOTReconstructor`
  - `dot_reconstruction_notebook.ipynb`: Example workflow.
- **`reconlib.modalities.eit`**: Electrical Impedance Tomography.
  - `operators.EITForwardOperator`
  - `reconstructors.EITReconstructor`
  - `eit_reconstruction_notebook.ipynb`: Example workflow.
- **`reconlib.modalities.em`**: Electron Microscopy.
  - (operators, reconstructors, notebook placeholders)
- **`reconlib.modalities.fluorescence_microscopy`**:
  - `operators.FluorescenceMicroscopyOperator`
  - `reconstructors.FluorescenceMicroscopyReconstructor`
  - `fluorescence_microscopy_reconstruction_notebook.ipynb`: Example workflow.
- **`reconlib.modalities.hyperspectral`**: Hyperspectral Imaging.
  - (operators, reconstructors, notebook placeholders)
- **`reconlib.modalities.infrared_thermography`**:
  - (operators, reconstructors, notebook placeholders)
- **`reconlib.modalities.microwave`**: Microwave Imaging.
  - (operators, reconstructors, notebook placeholders)
- **`reconlib.modalities.oct`**: Optical Coherence Tomography.
  - (operators, reconstructors, notebook placeholders)
- **`reconlib.modalities.pcct`**: Photon Counting CT.
  - `operators.PCCTProjectorOperator`: Multi-energy bin projector.
  - `reconstructors.PCCTReconstructor`: (Placeholder for pixel-based)
  - `voronoi_reconstructor.VoronoiPCCTReconstructor2D`: SART-like Voronoi material decomposition.
  - `material_decomposition.py`, `projection_domain_decomposition.py`
  - `pcct_reconstruction_notebook.ipynb`: Example workflow.
- **`reconlib.modalities.pet`**: Positron Emission Tomography.
  - `voronoi_reconstructor.VoronoiPETReconstructor2D`: MLEM-based Voronoi reconstruction.
  - `voronoi_reconstructor_3d.VoronoiPETReconstructor3D`: (Conceptual 3D extension)
  - `simulation.PETSimulation`, `preprocessing.PETPreprocessing`
- **`reconlib.modalities.photoacoustic`**: Photoacoustic Tomography.
  - `operators.PATForwardOperator`
  - `reconstructors.PATReconstructor`
  - `photoacoustic_reconstruction_notebook.ipynb`: Example workflow.
- **`reconlib.modalities.sar`**: Synthetic Aperture Radar.
  - (operators, reconstructors, notebook placeholders)
- **`reconlib.modalities.seismic`**: Seismic Imaging.
  - (operators, reconstructors, notebook placeholders)
- **`reconlib.modalities.sim`**: Structured Illumination Microscopy.
  - `operators.SIMOperator`
  - `reconstructors.SIMReconstructor`
  - `sim_reconstruction_notebook.ipynb`: Example workflow.
- **`reconlib.modalities.spect`**: Single Photon Emission Computed Tomography.
  - `operators.SPECTProjectorOperator` (placeholder)
  - `reconstructors.SPECTFBPReconstructor` (from `reconstructors.py`)
  - `voronoi_reconstructor.VoronoiSPECTReconstructor2D`: OSEM-based Voronoi reconstruction.
  - `spect_reconstruction_notebook.ipynb`: Example workflow.
- **`reconlib.modalities.terahertz`**:
  - (operators, reconstructors, notebook placeholders)
- **`reconlib.modalities.ultrasound`**:
  - `operators.UltrasoundOperator`
  - `reconstructors.UltrasoundReconstructor`
  - `regularizers.UltrasoundRegularizer`
  - `ultrasound_reconstruction_notebook.ipynb`: Example workflow.
- **`reconlib.modalities.xray_diffraction`**:
  - (operators, reconstructors, notebook placeholders)
- **`reconlib.modalities.xray_phase_contrast`**:
  - (operators, reconstructors, notebook placeholders)

## 6. Phase Unwrapping Submodule (`reconlib.phase_unwrapping`)
Provides algorithms for 2D and 3D phase unwrapping, a common problem in MRI and other phase-sensitive imaging modalities.
- `goldstein_unwrap.goldstein_phase_unwrap`: Branch-cut based algorithm.
- `least_squares_unwrap.least_squares_phase_unwrap`: Iterative least-squares solver.
- `quality_guided_unwrap.quality_guided_phase_unwrap`: Unwraps based on quality maps.
- `reference_echo_unwrap.reference_echo_phase_unwrap`: Uses a reference echo for unwrapping.
- `voronoi_unwrap.voronoi_phase_unwrap`: Voronoi-based unwrapping (experimental/research).
- `deep_learning_unwrap.DeepLearningPhaseUnwrap` (placeholder).
- `puror.PURORUnwrap` (placeholder).
- `romeo.ROMEOUnwrap` (placeholder for integration of ROMEO).
- `utils.py`: Mask generation utilities for phase unwrapping.

## 7. Examples (Top-Level `examples/` directory)
This directory contains Jupyter notebooks demonstrating example workflows using `reconlib`.
*(Summaries for each notebook would go here, based on previous analysis. If specific summaries weren't generated for all, this section might list the notebooks and their general topics.)*

- **`examples/ADMM_Example_Cartesian_MRI.ipynb`**: Demonstrates ADMM for Cartesian MRI reconstruction.
- **`examples/FISTA_Example_Cartesian_MRI.ipynb`**: Demonstrates FISTA for Cartesian MRI reconstruction.
- **`examples/Iterative_Reconstruction_Framework_Example.ipynb`**: Showcases the generic iterative reconstruction framework.
- **`examples/MoDL_Example_Cartesian_MRI.ipynb`**: Example of Model-Based Deep Learning (MoDL) for MRI.
- **`examples/NUFFT_Example.ipynb`**: Demonstrates usage of the NUFFT operators.
- **`examples/Phase_Unwrapping_Examples.ipynb`**: Examples of various phase unwrapping algorithms.
- **`examples/Regularization_Examples.ipynb`**: Illustrates different regularization terms.
- **`examples/Sliding_Window_NUFFT_Example.ipynb`**: Example of sliding window NUFFT reconstruction.
- **`examples/UNet_Denoiser_Example.ipynb`**: Demonstrates training/using a UNet denoiser.
- **`examples/Wavelet_Regularization_Example.ipynb`**: Example of wavelet-based regularization in reconstruction.
- **`examples/Working_with_MRIData_Object.ipynb`**: Shows how to use the `MRIData` class.
- **`examples/B0_Mapping_Examples.ipynb`**: (from `reconlib/b0_mapping`) Examples of B0 field map estimation.

*(Note: `examples/_no_notebook_reconstruction_with_estimated_csm.ipynb` was previously identified as a Python script, not a notebook. Its purpose would be reconstructing MRI data with estimated coil sensitivity maps using an iterative algorithm, likely demonstrating components from `reconlib.csm`, `reconlib.operators`, and `reconlib.solvers`.)*
