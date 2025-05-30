# Voronoi-Weighted Non-Cartesian MRI Reconstruction Example

This page describes an example demonstrating how to perform non-Cartesian MRI reconstruction using Voronoi-based density compensation with the `reconlib` library.

## Overview

Non-Cartesian k-space sampling trajectories (e.g., radial, spiral) can offer several benefits in MRI. However, reconstructing images from such data requires careful handling of varying sample densities across k-space. Voronoi tessellation provides a robust method to calculate Density Compensation Factors (DCF) by assigning a weight to each k-space sample inversely proportional to the area/volume of its Voronoi cell.

The example notebook `examples/voronoi_recon_comparison.ipynb` walks through the process of:
1.  Generating a 2D Shepp-Logan phantom.
2.  Simulating a non-Cartesian (radial) k-space trajectory and acquiring noisy k-space data.
3.  Calculating Voronoi density weights using `reconlib.voronoi.density_weights_pytorch.compute_voronoi_density_weights_pytorch`.
4.  Reconstructing the image using three methods:
    *   Conjugate Gradient (CG) with Voronoi weights.
    *   Simple gridding (adjoint NUFFT with default/no explicit DCF).
    *   Conjugate Gradient (CG) with default/no explicit DCF.
5.  Comparing the results visually and quantitatively (MSE, PSNR, SSIM).

**You can find the full interactive example here: [`examples/voronoi_recon_comparison.ipynb`](../examples/voronoi_recon_comparison.ipynb).**

## Key Steps & Code Snippets

### 1. Computing Voronoi Weights

After generating your `k_trajectory` (e.g., shape `(num_points, 2)` for 2D) and defining `bounds` for your k-space (e.g., `torch.tensor([[-0.5, -0.5], [0.5, 0.5]])`), you can compute the weights:

```python
from reconlib.voronoi.density_weights_pytorch import compute_voronoi_density_weights_pytorch

# k_trajectory should be a PyTorch tensor on the desired device
# bounds should also be a PyTorch tensor
voronoi_weights = compute_voronoi_density_weights_pytorch(
    points=k_trajectory,
    bounds=voronoi_bounds,
    space_dim=2 # Or 3 for 3D
)
```

### 2. Setting up NUFFT with Voronoi Weights

The computed `voronoi_weights` are passed to the NUFFT operator (e.g., `NUFFT2D`) via the `density_comp_weights` argument in its constructor.

```python
from reconlib.nufft import NUFFT2D

nufft_params_voronoi = {
    'image_shape': (IMAGE_SIZE, IMAGE_SIZE),
    'k_trajectory': k_trajectory,
    'oversamp_factor': (2.0, 2.0),
    'kb_J': (4, 4),
    'kb_alpha': (2.34 * 4, 2.34 * 4),
    'Ld': (1024, 1024),
    'kb_m': (0.0, 0.0),
    'device': device,
    'density_comp_weights': voronoi_weights # Key addition
}
nufft_op_voronoi = NUFFT2D(**nufft_params_voronoi)
```

### 3. Performing Reconstruction (e.g., Conjugate Gradient)

The `reconlib.solvers` module contains various reconstruction algorithms. For example, to use Conjugate Gradient:

```python
from reconlib.solvers import conjugate_gradient_reconstruction

recon_image = conjugate_gradient_reconstruction(
    kspace_data=k_space_data_noisy,
    nufft_operator_class=NUFFT2D,
    nufft_kwargs=nufft_params_voronoi, # Contains the Voronoi weights
    max_iters=20,
    tol=1e-6
)
```
The `conjugate_gradient_reconstruction` solver (and others like `fista_reconstruction`, `admm_reconstruction`) will internally use the `density_comp_weights` provided within `nufft_kwargs` when creating and using the NUFFT operator for its forward and adjoint operations.

## Benefits

Using Voronoi-based density compensation, especially with iterative reconstruction algorithms like Conjugate Gradient, generally leads to:
*   Reduced aliasing artifacts.
*   Improved image sharpness and contrast.
*   More accurate quantitative values.

Refer to the [example notebook](../examples/voronoi_recon_comparison.ipynb) for a full implementation and comparison with other methods.
