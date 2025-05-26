# reconlib

**reconlib** is a Python library for advanced k-space trajectory generation, regridding, and iterative image reconstruction, mainly for MRI applications. It also provides visualization tools for k-space data and reconstructed images.

---

## Features

- **Trajectory Generation**: Create various k-space sampling trajectories (e.g., spiral, radial).
- **Regridding**: Convert non-uniform k-space data to a uniform grid suitable for FFT.
- **Iterative Reconstruction**: Implement algorithms like L1-L2 minimization for image reconstruction.
- **Visualization**: Plot k-space trajectories and reconstructed images.

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/kaggie/reconlib.git
cd reconlib
pip install -r requirements.txt
```

---

## Usage Examples

### 1. Generate a Spiral Trajectory

```python
from reconlib.trajectorygen import generate_spiral

trajectory = generate_spiral(num_points=1024, num_arms=8)
```

---

### 2. Regrid Non-Uniform Data

```python
from reconlib.regrid import regrid_data

# non_uniform_data: your acquired k-space data
# trajectory: sampling trajectory, e.g., from generate_spiral
uniform_data = regrid_data(non_uniform_data, trajectory)
```

---

### 3. Perform L1-L2 Reconstruction

```python
from reconlib.l1l2recon import reconstruct

# k_space_data: your (possibly regridded) k-space data
# trajectory: the trajectory used for acquisition
# lambda_reg: regularization parameter
image = reconstruct(k_space_data, trajectory, lambda_reg=0.01)
```

---

### 4. Plot Trajectory

```python
from reconlib.plot_trajectory import plot

plot(trajectory)
