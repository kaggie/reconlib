# reconlib

**reconlib** is a Python library for advanced k-space trajectory generation, **self-contained Non-Uniform Fast Fourier Transform (NUFFT) operations**, regridding (though NUFFT adjoint often replaces explicit regridding), and iterative image reconstruction, mainly for MRI applications. It also provides visualization tools for k-space data and reconstructed images.

---

## Features

- **Trajectory Generation**: Create various k-space sampling trajectories (e.g., spiral, radial).
- **Self-Contained NUFFT**: Internal Python-based 2D and 3D NUFFT (table-based, MIRT-inspired) for CPU/GPU, removing dependencies on external libraries like SigPy or PyNUFFT. The `NUFFTOperator` class provides a unified interface.
- **Regridding**: Convert non-uniform k-space data to a uniform grid suitable for FFT. (Note: The adjoint operation of the NUFFT engine inherently performs gridding).
- **Iterative Reconstruction**: Implement algorithms like L1-L2 minimization for image reconstruction, leveraging the `NUFFTOperator`.
- **Visualization**: Plot k-space trajectories and reconstructed images.

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/kaggie/reconlib.git
cd reconlib
pip install -r requirements.txt 
```
(Ensure `numpy`, `scipy`, and `torch` are listed in `requirements.txt`. No external NUFFT libraries are needed for the core table-based NUFFT functionality.)

---

## Usage Examples

### 1. Generate a Spiral Trajectory

```python
from reconlib.trajectorygen import generate_spiral # Assuming this module exists

trajectory = generate_spiral(num_points=1024, num_arms=8)
```

---

### 2. Using the NUFFTOperator

The `NUFFTOperator` provides a high-level interface for NUFFT operations, internally using `NUFFT2D` or `NUFFT3D` which are table-based.

```python
from reconlib.operators import NUFFTOperator
import torch
import numpy as np # For np.pi if used in example (though math.pi is preferred)
import math 

# Example 2D Setup
image_shape_2d = (128, 128)
# Dummy k-space trajectory (replace with actual trajectory)
# k_traj_2d should be normalized to [-0.5, 0.5] for each dimension
num_points = 512
k_coords = torch.rand(num_points, 2, dtype=torch.float32) - 0.5 

# MIRT-style parameters for NUFFTOperator
oversamp_factor_2d = (2.0, 2.0)
kb_J_2d = (4, 4) # Kernel width for each dimension
# Alpha (shape parameter) for Kaiser-Bessel kernel, often J * some_factor
kb_alpha_2d = tuple(2.34 * J for J in kb_J_2d) 
Ld_2d = (2**10, 2**10) # Table oversampling factor

# These can be often left to defaults or calculated by NUFFTOperator:
# Kd_2d (oversampled grid size) will be calculated if None.
# kb_m_2d (KB order) defaults to (0.0, 0.0) if None.
# n_shift_2d (FFT shift for image center) defaults to (0.0, 0.0) if None.

nufft_op = NUFFTOperator(
    k_trajectory=k_coords,
    image_shape=image_shape_2d,
    oversamp_factor=oversamp_factor_2d,
    kb_J=kb_J_2d,
    kb_alpha=kb_alpha_2d,
    Ld=Ld_2d,
    device='cpu' # or 'cuda' if available
)

# Now nufft_op.op(image_data) and nufft_op.op_adj(kspace_data) can be used.
print("NUFFTOperator for 2D initialized and ready for use.")

# For 3D, the setup is similar, with 3-element tuples for shape and MIRT params,
# and k_trajectory of shape [num_points, 3].
# nufft_op_3d = NUFFTOperator(..., nufft_type_3d='table') # or 'direct'
```

---

### 3. Regrid Non-Uniform Data (via NUFFT Adjoint)

The adjoint operation of NUFFT inherently performs gridding.

```python
# Assuming nufft_op is initialized as above
# non_uniform_kspace_data: your acquired k-space data (1D tensor)
# gridded_data_oversampled = nufft_op.op_adj(non_uniform_kspace_data) 
# Note: The output of op_adj is an image. If you need k-space gridded data,
# you would typically use a dedicated gridding algorithm or the internal
# 'gridded_k_space' from the NUFFT3D.adjoint method before IFFT.
# The direct output of op_adj is the image after gridding and IFFT.
# For simple regridding to Cartesian k-space, external tools might be more direct
# if the full NUFFT image reconstruction is not the goal.
```
The example for `regrid.py` is removed as its functionality is now part of the NUFFT adjoint or would require a different approach.

---

### 4. Perform L1-L2 Reconstruction

The `l1l2recon.py` module can use the `NUFFTOperator` as its linear operator.
(Assuming `l1l2recon.reconstruct` is adapted or a new example is shown using `L1Reconstruction` or `L2Reconstruction` classes directly with `NUFFTOperator`.)

```python
from l1l2recon import L2Reconstruction # Or L1Reconstruction
# Assuming nufft_op (e.g., nufft_op_2d from above) and kspace_data_2d_noisy are defined

l2_recon_module = L2Reconstruction(linear_operator=nufft_op, num_iterations=10)
reconstructed_image = l2_recon_module.forward(kspace_data_2d_noisy) 

# Or, if using a simplified top-level 'reconstruct' function that now uses NUFFTOperator:
# image = reconstruct(kspace_data_2d_noisy, nufft_op, lambda_reg=0.01) 
# This would require 'reconstruct' in l1l2recon.py to be updated.
```

---

### 5. Plot Trajectory

```python
from reconlib.plot_trajectory import plot # Assuming this module exists

# trajectory: a 2D or 3D k-space trajectory tensor (e.g., k_coords from NUFFTOperator example)
plot(k_coords.cpu()) # Ensure plotting function can handle the trajectory format
```

---
### 6. Comprehensive Pipeline Examples

For detailed examples of 2D and 3D reconstruction pipelines, including data generation, NUFFT setup, iterative reconstruction (CG), and L1/L2 regularized reconstruction, please see the Jupyter Notebook: `examples/recon_pipeline_demo.ipynb`.
