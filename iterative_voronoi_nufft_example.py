# %% [markdown]
# # Iterative NUFFT Reconstruction with Voronoi Density Compensation
#
# This notebook demonstrates iterative Magnetic Resonance Imaging (MRI) reconstruction using a Non-Uniform Fast Fourier Transform (NUFFT). It particularly focuses on applying Voronoi-based density compensation to the k-space samples. We will:
# 1. Set up parameters for a 2D imaging scenario.
# 2. Generate a k-space trajectory and a phantom image.
# 3. Simulate k-space data from the phantom using a NUFFT operator.
# 4. Calculate Voronoi density compensation weights based on the k-space trajectory.
# 5. Visualize the k-space samples and their Voronoi cells.
# 6. Perform iterative reconstruction using gradient descent, both with and without Voronoi weights.
# 7. Compare the reconstructed images.
# 8. Optionally, demonstrate some 3D geometric plotting utilities available in `reconlib`.

# %%
# Cell 2: Imports
import torch
import numpy as np
import matplotlib.pyplot as plt

from reconlib.solvers import iterative_reconstruction
from reconlib.nufft import NUFFT2D # Using 2D for this example
from reconlib.utils import calculate_density_compensation
from reconlib.plotting import (
    plot_voronoi_kspace, 
    plot_3d_hull, 
    plot_3d_delaunay
    # plot_3d_voronoi_with_hull # Can be added if a specific 3D Voronoi example is constructed
)
from reconlib.voronoi_utils import ConvexHull, delaunay_triangulation_3d, EPSILON

# Configure Matplotlib for inline plotting in Jupyter
# %matplotlib inline # This is a magic command for Jupyter, not for .py script

# %%
# Cell 3: Setup Parameters and Device
torch.manual_seed(0)
np.random.seed(0) # For NumPy operations if any

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Image parameters
image_shape_2d = (64, 64) # (Height, Width)

# K-space parameters
num_k_points = 512 # Total number of k-space samples
# For a radial-like trajectory example, one might define spokes and samples per spoke.
# For simplicity, we'll use random points first, then a simple radial pattern.

# NUFFT parameters for NUFFT2D
nufft_kwargs_2d = {
    'oversamp_factor': (2.0, 2.0),
    'kb_J': (5, 5), # Kaiser-Bessel kernel width
    'kb_alpha': (2.34 * 5, 2.34 * 5), # Kaiser-Bessel shape parameter (common choice: 2.34 * J)
    'Ld': (2**10, 2**10) # Table oversampling factor for interpolation
    # 'kb_m': (0.0, 0.0) # Default is 0.0, standard Kaiser-Bessel
}
print(f"Image shape: {image_shape_2d}")
print(f"Number of k-space points: {num_k_points}")
print(f"NUFFT parameters: {nufft_kwargs_2d}")

# %%
# Cell 4: Generate K-space Trajectory

# Option 1: Simple Random Trajectory
# sampling_points_2d = (torch.rand(num_k_points, 2, device=device, dtype=torch.float32) - 0.5) * 2 
# This creates points in [-1, 1], but NUFFT typically expects [-0.5, 0.5] if k-space is normalized that way.
# Let's assume normalization to [-0.5, 0.5] which is common.
sampling_points_2d = (torch.rand(num_k_points, 2, device=device, dtype=torch.float32) - 0.5)

# Option 2: Simple Radial-like Trajectory
num_spokes = 32
samples_per_spoke = num_k_points // num_spokes 
if samples_per_spoke * num_spokes != num_k_points: # Adjust num_k_points if not perfectly divisible
    num_k_points = samples_per_spoke * num_spokes
    print(f"Adjusted num_k_points to {num_k_points} for radial trajectory.")

angles = torch.linspace(0, np.pi - (np.pi / num_spokes), num_spokes, device=device, dtype=torch.float32)
radii = torch.linspace(0, 0.5, samples_per_spoke, device=device, dtype=torch.float32)

kx_radial = torch.outer(radii, torch.cos(angles)).flatten()
ky_radial = torch.outer(radii, torch.sin(angles)).flatten()
sampling_points_2d = torch.stack((kx_radial, ky_radial), dim=1)

print(f"K-space trajectory shape: {sampling_points_2d.shape}")
print(f"K-space trajectory min/max: {sampling_points_2d.min().item():.2f} / {sampling_points_2d.max().item():.2f}")

# Plot the k-space trajectory
plt.figure(figsize=(6, 6))
plt.scatter(sampling_points_2d[:, 0].cpu().numpy(), sampling_points_2d[:, 1].cpu().numpy(), s=5, alpha=0.7)
plt.title('2D K-space Trajectory')
plt.xlabel('kx')
plt.ylabel('ky')
plt.axis('equal')
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()

# %%
# Cell 5: Create Phantom Image and Simulate K-space Data

# Create a simple 2D phantom image (a square)
phantom_2d = torch.zeros(image_shape_2d, device=device, dtype=torch.complex64)
center_x, center_y = image_shape_2d[0] // 2, image_shape_2d[1] // 2
square_half_size = image_shape_2d[0] // 8 # A smaller square
phantom_2d[
    center_x - square_half_size : center_x + square_half_size,
    center_y - square_half_size : center_y + square_half_size
] = 1.0 + 0.0j # Complex data

plt.figure(figsize=(5,5))
plt.imshow(torch.abs(phantom_2d).cpu().numpy(), cmap='gray')
plt.title('Original Phantom (Magnitude)')
plt.axis('off')
plt.show()

# Instantiate NUFFT2D operator for phantom data generation
# For data generation, density compensation is not strictly needed in NUFFT forward.
# The NUFFT classes have been modified to accept density_comp_weights,
# but if None, NUFFT2D uses its internal _estimate_density_compensation for adjoint,
# and NUFFT3D does not apply DCF in adjoint if not provided.
# Forward operation does not use DCF.
nufft_op_phantom = NUFFT2D(
    image_shape=image_shape_2d,
    k_trajectory=sampling_points_2d,
    device=device,
    **nufft_kwargs_2d
)

# Compute clean k-space data
kspace_data_clean = nufft_op_phantom.forward(phantom_2d)
# Ensure kspace_data is 1D (num_k_points,) as expected by iterative_reconstruction
if kspace_data_clean.ndim > 1:
    kspace_data_clean = kspace_data_clean.flatten()

print(f"Clean k-space data shape: {kspace_data_clean.shape}")

# Optionally, add complex Gaussian noise
noise_level_percentage = 5 # Percentage of max signal intensity
if noise_level_percentage > 0:
    # Calculate noise standard deviation based on a percentage of the max k-space signal
    # This is a heuristic for setting a reasonable noise level.
    if kspace_data_clean.numel() > 0 :
        max_abs_kspace = torch.max(torch.abs(kspace_data_clean))
        noise_std = (noise_level_percentage / 100.0) * max_abs_kspace
        # Generate complex Gaussian noise
        noise_real = torch.randn_like(kspace_data_clean.real) * noise_std / np.sqrt(2) # Divide by sqrt(2) for real and imag parts
        noise_imag = torch.randn_like(kspace_data_clean.imag) * noise_std / np.sqrt(2)
        complex_noise = torch.complex(noise_real, noise_imag)
        kspace_data_noisy = kspace_data_clean + complex_noise
        print(f"Added noise with std: {noise_std.item():.2e}")
    else:
        kspace_data_noisy = kspace_data_clean # No noise if no signal
        print("No k-space signal, no noise added.")
else:
    kspace_data_noisy = kspace_data_clean
    print("Using noise-free k-space data.")

# Use noisy data for reconstruction
kspace_data_to_reconstruct = kspace_data_noisy.clone() # Clone to avoid modifying it later if needed

# %% [markdown]
# ## Density Compensation
# Density compensation is crucial in NUFFT reconstruction to account for non-uniform sampling densities in k-space. Higher density regions should contribute less per sample to the image reconstruction, and vice-versa.
#
# ### Voronoi-based Density Compensation
# Voronoi tessellation divides the k-space into regions, where each region consists of points closest to one sample. The area (in 2D) or volume (in 3D) of a sample's Voronoi cell is inversely proportional to the sampling density around that sample. Thus, these areas/volumes can be used to calculate density compensation weights (often, weight = area/volume or 1/area/volume depending on definition, here we use 1/measure).

# %%
# Cell 6: Calculate Voronoi Density Compensation Weights

# Define bounds for Voronoi calculation, typically covering the k-space sampling area
# For sampling_points_2d in [-0.5, 0.5], bounds can be slightly larger or match this.
bounds_2d = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], device=device, dtype=torch.float32)

print("Calculating Voronoi density compensation weights...")
voronoi_weights = calculate_density_compensation(
    k_trajectory=sampling_points_2d, 
    image_shape=image_shape_2d, # image_shape is not used by 'voronoi' method in calculate_density_compensation
    method='voronoi', 
    bounds=bounds_2d, 
    device=device # calculate_density_compensation will pass this to compute_voronoi_density_weights
)

print(f"Voronoi weights shape: {voronoi_weights.shape}")
if voronoi_weights.numel() > 0:
    print(f"Min Voronoi weight: {voronoi_weights.min().item():.2e}")
    print(f"Max Voronoi weight: {voronoi_weights.max().item():.2e}")
    print(f"Mean Voronoi weight: {voronoi_weights.mean().item():.2e}")
else:
    print("Voronoi weights tensor is empty.")

# %%
# Cell 7: Visualize K-space Samples with Voronoi Cells
print("Plotting Voronoi cells (this may take a moment for many points)...")
fig_vor, ax_vor = plt.subplots(figsize=(8,8))
plot_voronoi_kspace(
    kspace_points=sampling_points_2d.cpu(), # Plotting function expects CPU tensors
    weights=voronoi_weights.cpu(), 
    bounds=bounds_2d.cpu(), 
    ax=ax_vor,
    title='K-space Samples with Voronoi Cells (Colored by Weight)'
)
plt.show()

# %% [markdown]
# ## Iterative Reconstruction
# We will now perform iterative reconstruction using a simple gradient descent algorithm. The core update rule is:
# $ x_{k+1} = x_k - \alpha \nabla J(x_k) $
# where $ J(x) = \frac{1}{2} \| A(x) - y \|_2^2 $ is the data consistency cost function.
# The gradient is $ \nabla J(x_k) = A^H (A(x_k) - y) $, where $A$ is the NUFFT forward operator and $A^H$ is its adjoint.
#
# We will compare reconstruction with and without applying the Voronoi weights.
# When Voronoi weights ($w$) are used, the cost function can be thought of as $ J(x) = \frac{1}{2} \| \sqrt{w} (A(x) - y) \|_2^2 $ or the weights are incorporated into the adjoint $A^H$. The `iterative_reconstruction` function expects `kspace_data` to be pre-weighted if `use_voronoi` is true, and also passes weights to NUFFT if the NUFFT class handles them (which ours now do).

# %%
# Cell 8: Perform Iterative Reconstruction without Voronoi Weights

print("Performing iterative reconstruction WITHOUT Voronoi weights...")
# Ensure kspace_data_to_reconstruct is on the correct device
img_recon_no_voronoi = iterative_reconstruction(
    kspace_data=kspace_data_to_reconstruct.to(device), # Pass the (potentially noisy) k-space data
    sampling_points=sampling_points_2d,
    image_shape=image_shape_2d,
    nufft_operator_class=NUFFT2D,
    nufft_kwargs=nufft_kwargs_2d,
    use_voronoi=False,
    max_iters=20, # Number of iterations
    fixed_alpha=0.01 # Step size
)

plt.figure(figsize=(6,6))
plt.imshow(torch.abs(img_recon_no_voronoi).cpu().numpy(), cmap='gray')
plt.title('Reconstruction without Voronoi Weights (Magnitude)')
plt.axis('off')
plt.show()

# %%
# Cell 9: Perform Iterative Reconstruction with Voronoi Weights

print("\nPerforming iterative reconstruction WITH Voronoi weights...")
# The iterative_reconstruction function handles weighting kspace_data internally if use_voronoi=True
# and passes voronoi_weights to the NUFFT operator.
img_recon_with_voronoi = iterative_reconstruction(
    kspace_data=kspace_data_to_reconstruct.to(device), # Pass the original (potentially noisy) k-space data
    sampling_points=sampling_points_2d,
    image_shape=image_shape_2d,
    nufft_operator_class=NUFFT2D,
    nufft_kwargs=nufft_kwargs_2d,
    use_voronoi=True,
    voronoi_weights=voronoi_weights, # Provide the calculated weights
    max_iters=20,
    fixed_alpha=0.01
)

plt.figure(figsize=(6,6))
plt.imshow(torch.abs(img_recon_with_voronoi).cpu().numpy(), cmap='gray')
plt.title('Reconstruction with Voronoi Weights (Magnitude)')
plt.axis('off')
plt.show()

# %% [markdown]
# ### Comparison of Results
# Let's display the original phantom and the two reconstructions side-by-side for comparison.

# %%
fig_comp, axes_comp = plt.subplots(1, 3, figsize=(18, 6))
axes_comp[0].imshow(torch.abs(phantom_2d).cpu().numpy(), cmap='gray')
axes_comp[0].set_title('Original Phantom')
axes_comp[0].axis('off')

axes_comp[1].imshow(torch.abs(img_recon_no_voronoi).cpu().numpy(), cmap='gray')
axes_comp[1].set_title('Recon without Voronoi DCF')
axes_comp[1].axis('off')

axes_comp[2].imshow(torch.abs(img_recon_with_voronoi).cpu().numpy(), cmap='gray')
axes_comp[2].set_title('Recon with Voronoi DCF')
axes_comp[2].axis('off')

plt.suptitle('Comparison of Reconstructions', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
plt.show()

# %% [markdown]
# # Cell 10: (Optional) Demonstrate 3D Geometric Plotting
# This section demonstrates the 3D plotting utilities for visualizing convex hulls and Delaunay triangulations.

# %%
# Cell 10 Code: Demonstrate 3D Plotting Utilities
print("\nDemonstrating 3D plotting utilities...")

# Generate 3D points
torch.manual_seed(42) # Seed for this cell's random data
points_3d = torch.rand(15, 3, device=device, dtype=torch.float32) * 10 # Scale for better visualization

# 1. Convex Hull Plot
print("Computing and plotting 3D Convex Hull...")
try:
    hull_3d = ConvexHull(points_3d, tol=EPSILON) # Use EPSILON from voronoi_utils
    if hull_3d.simplices is not None and hull_3d.simplices.numel() > 0:
        plot_3d_hull(points_3d, hull_3d.vertices, hull_3d.simplices, 
                     show_points=True, show_hull=True)
        plt.title("3D Convex Hull Demonstration")
        plt.show()
    else:
        print("Could not generate 3D convex hull for plotting (degenerate input or too few points).")
except Exception as e:
    print(f"Error during 3D Convex Hull plotting: {e}")


# 2. Delaunay Triangulation Plot
print("\nComputing and plotting 3D Delaunay Triangulation (Placeholder)...")
# Note: delaunay_triangulation_3d in voronoi_utils is a placeholder.
# Its output might be minimal or based on ConvexHull internally for demonstration.
try:
    tetrahedra = delaunay_triangulation_3d(points_3d, tol=EPSILON)
    # The plot_3d_delaunay function can compute its own hull if not provided
    # For robustness, compute it here if needed for the plot function argument.
    if points_3d.shape[0] >=4 : # Need enough points for a hull
        hull_for_delaunay_plot = ConvexHull(points_3d, tol=EPSILON)
        if tetrahedra.numel() > 0 :
             plot_3d_delaunay(points_3d, tetrahedra, convex_hull=hull_for_delaunay_plot, 
                             show_tetrahedra=(tetrahedra.numel() > 0), show_hull=True)
             plt.title("3D Delaunay Triangulation Demonstration")
             plt.show()
        else:
            print("Delaunay triangulation resulted in no tetrahedra to plot.")
            # Optionally, still plot points and hull:
            if hull_for_delaunay_plot.simplices is not None and hull_for_delaunay_plot.simplices.numel() > 0:
                 plot_3d_delaunay(points_3d, tetrahedra, convex_hull=hull_for_delaunay_plot, 
                                  show_tetrahedra=False, show_hull=True)
                 plt.title("3D Points and their Convex Hull (No Delaunay Tetrahedra)")
                 plt.show()

    else:
        print("Not enough points to attempt Delaunay triangulation or hull for plotting.")
except Exception as e:
    print(f"Error during 3D Delaunay plotting: {e}")


# %% [markdown]
# # Cell 11: Conclusion
# This notebook demonstrated an iterative NUFFT-based MRI reconstruction pipeline.
# Key steps included:
# - Setting up a 2D phantom and k-space trajectory.
# - Simulating k-space data using the NUFFT forward operation.
# - Calculating density compensation weights using the Voronoi method, which involves computing the area/volume of Voronoi cells corresponding to each k-space sample.
# - Visualizing the k-space samples and their Voronoi cells.
# - Performing iterative reconstruction using gradient descent with and without these Voronoi weights.
# - Qualitatively comparing the reconstruction results, where appropriate DCF typically leads to improved image quality (e.g., reduced artifacts, better uniformity).
# - Optionally, visualizations of 3D geometric concepts like Convex Hulls and Delaunay triangulations were shown using the library's plotting utilities.
#
# This example provides a basic framework that can be extended for more advanced reconstruction techniques and k-space sampling strategies.

# %%
print("Notebook execution complete.")
