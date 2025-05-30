# ReconLib: MRI Reconstruction Library

`reconlib` is a comprehensive Python library designed for advanced Magnetic Resonance Imaging (MRI) reconstruction. It provides a suite of tools and algorithms for various complex reconstruction tasks, catering to both research and development in the field of medical imaging. The library is built with a focus on modularity and extensibility, allowing users to integrate and customize reconstruction pipelines.

## Key Feature Areas

*   **Forward and Adjoint Operators**: Including Non-Uniform Fast Fourier Transform (NUFFT), coil sensitivity modeling, and MRI forward operators.
*   **Regularizers**: A collection of regularization techniques such as L1 norm, L2 norm, Total Variation (TV), and more advanced methods like gradient matching.
*   **Optimizers**: Iterative optimization algorithms including FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) and ADMM (Alternating Direction Method of Multipliers).
*   **Coil Combination**: Methods for combining data from multiple receiver coils.
*   **Phase Unwrapping**: Algorithms to resolve phase ambiguities in MRI phase data, including quality-guided, Goldstein, and region-growing approaches.
*   **B0 Mapping**: Tools for estimating and correcting B0 field inhomogeneities.
*   **PET/CT Reconstruction**: Modules for Positron Emission Tomography (PET) and Computed Tomography (CT) reconstruction, including projectors, geometry definition, and simulation tools.
*   **Deep Learning Integration**: Components for integrating deep learning models into reconstruction workflows, such_as learned denoisers and unrolled networks.
*   **Simulation Utilities**: Tools for generating synthetic phantom data and simulating MRI acquisitions.
*   **Plotting and Visualization**: Utilities for displaying MRI images, phase maps, k-space data, and other relevant visualizations.
*   **Voronoi-Based Tools**: Advanced functionalities leveraging Voronoi diagrams for tasks like density compensation and phase unwrapping.

## Voronoi Functionality

The `reconlib` library includes specialized tools based on Voronoi tessellation, primarily located within the `reconlib.voronoi` submodule. These offer powerful methods for k-space analysis and phase processing.

### Voronoi Density Weights

The function `reconlib.voronoi.voronoi_analysis.compute_voronoi_density_weights` calculates density compensation factors for non-Cartesian k-space sampling patterns. These weights are crucial for accurate image reconstruction by accounting for the non-uniform sampling density.

**Usage Example (Conceptual):**
```python
import torch
from reconlib.voronoi.voronoi_analysis import compute_voronoi_density_weights

# Assume k_space_points is a (N, 2) or (N, 3) tensor of k-space coordinates
# Assume bounds is a (2, 2) or (2, 3) tensor defining the k-space boundaries
# k_space_points = torch.rand((1000, 2)) * 0.5 - 0.25 # Example random k-space points
# bounds = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])    # Example bounds

# density_weights = compute_voronoi_density_weights(k_space_points, bounds=bounds)
# These weights can then be used in NUFFT operations or iterative reconstruction algorithms.
```
For more details, refer to the documentation within `reconlib.voronoi.voronoi_analysis`.

### Voronoi-Based Phase Unwrapping

The function `reconlib.phase_unwrapping.voronoi_unwrap.unwrap_phase_voronoi_region_growing` provides a phase unwrapping algorithm that utilizes Voronoi cells generated from high-quality seed points to guide the unwrapping process.

**Usage Example (Conceptual):**
```python
import torch
from reconlib.phase_unwrapping.voronoi_unwrap import unwrap_phase_voronoi_region_growing

# Assume wrapped_phase_data is a 2D or 3D tensor of wrapped phase values
# wrapped_phase_data = ... # e.g., torch.rand((64, 64)) * (2 * torch.pi) - torch.pi
# mask = torch.ones_like(wrapped_phase_data, dtype=torch.bool) # Optional
# voxel_size = (1.0, 1.0)

# unwrapped_phase = unwrap_phase_voronoi_region_growing(
#     phase_data=wrapped_phase_data,
#     mask=mask,
#     voxel_size=voxel_size,
#     quality_threshold=0.1,
#     min_seed_distance=5.0
# )
```
A detailed usage example can be found in the Jupyter Notebook: `examples/example_voronoi_phase_unwrapping.ipynb`.

### Detailed Voronoi Submodule Documentation

For in-depth information on the Voronoi submodule's architecture, algorithms, and individual components, please refer to its dedicated wiki:
[Voronoi Submodule Wiki](../reconlib/voronoi/wiki/README.md)

*(Note: The relative path `../reconlib/voronoi/wiki/README.md` assumes this main wiki README is in `<repo_root>/wiki/README.md` and the submodule wiki is at `<repo_root>/reconlib/voronoi/wiki/README.md`)*.

## Examples and Tutorials

*   **[Voronoi-Weighted Non-Cartesian MRI Reconstruction](./VoronoiReconstructionExample.md)**: Demonstrates how to use Voronoi density compensation for non-Cartesian MRI reconstruction and compares it with other methods. Includes a link to a [Jupyter Notebook example](../examples/voronoi_recon_comparison.ipynb).
*   **[User Guide: Basic Non-Cartesian Reconstruction Pipeline](../examples/tutorial_basic_pipeline.ipynb)**: A step-by-step guide to setting up a simple 2D non-Cartesian reconstruction from data simulation to image display using NUFFT and a conjugate gradient solver.
*   **[User Guide: Reconstruction with Regularizers](../examples/tutorial_regularizers.ipynb)**: Demonstrates how to use L1 and Total Variation (TV) regularizers with the FISTA algorithm to improve reconstruction quality, especially for noisy or undersampled data.

## Getting Started

[Instructions on how to install and use `reconlib` will be added here.]

## Contributing

[Information for developers wishing to contribute to `reconlib`.]

## License

[`reconlib` is typically released under an open-source license, e.g., MIT License. Details here.]
