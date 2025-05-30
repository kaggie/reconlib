import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# This allows running the example directly from the 'examples' folder.
# For general use, it's recommended to install reconlib (e.g., `pip install -e .` from root).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from reconlib.operators import NUFFTOperator
except ImportError:
    print("\n*****************************************************************")
    print("WARNING: NUFFTOperator not found in reconlib.operators.")
    print("         Using a MOCK NUFFTOperator for demonstration.")
    print("         RESULTS WILL BE RANDOM AND NOT MEANINGFUL.")
    print("*****************************************************************\n")
    # Define a Mock NUFFT operator if the real one is not available
    class MockNUFFTOperator:
        def __init__(self, k_trajectory, image_shape, device='cpu', oversamp_factor=(2.0,2.0),
                     kb_J=(6,6), kb_alpha=(13.8,13.8), Ld=(1024,1024), Kd=(128,128), kb_m=(0.0,0.0), **kwargs): # Standardized mock signature
            self.k_trajectory = k_trajectory
            self.image_shape = image_shape
            self.device = device
            self.oversamp_factor = oversamp_factor
            self.kb_J = kb_J
            self.kb_alpha = kb_alpha
            self.Ld_table_length = Ld
            self.Kd_oversampled_dims = Kd if Kd is not None else tuple(int(i*o) for i,o in zip(image_shape, oversamp_factor))
            self.kb_m = kb_m
            print(f"MockNUFFTOperator initialized for image shape {image_shape} on device {device}.")
            print(f"  K-traj shape: {k_trajectory.shape}, OS: {self.oversamp_factor}, Kernel J: {self.kb_J}, "
                  f"Alpha: {self.kb_alpha}, Table Ld: {self.Ld_table_length}, Grid Kd: {self.Kd_oversampled_dims}, KB_m: {self.kb_m}")

        def op(self, x: torch.Tensor) -> torch.Tensor: # Expects (H, W) or (B, H, W)
            if x.ndim == 2: # H, W
                # Simulate k-space data based on k_trajectory length
                num_k_points = self.k_trajectory.shape[0]
                return torch.randn(num_k_points, dtype=torch.complex64, device=self.device) * torch.mean(torch.abs(x))
            elif x.ndim == 3: # B, H, W (batch of images, e.g. coils)
                num_k_points = self.k_trajectory.shape[0]
                batch_size = x.shape[0]
                return torch.randn(batch_size, num_k_points, dtype=torch.complex64, device=self.device) * torch.mean(torch.abs(x))
            else:
                raise ValueError(f"MockNUFFTOperator.op expects 2D or 3D input, got {x.ndim}D")


        def op_adj(self, y: torch.Tensor) -> torch.Tensor: # Expects (K,) or (B, K)
            if y.ndim == 1: # K,
                return torch.randn(self.image_shape, dtype=torch.complex64, device=self.device) * torch.mean(torch.abs(y))
            elif y.ndim == 2: # B, K (batch of k-space data)
                batch_size = y.shape[0]
                return torch.randn(batch_size, *self.image_shape, dtype=torch.complex64, device=self.device) * torch.mean(torch.abs(y))
            else:
                raise ValueError(f"MockNUFFTOperator.op_adj expects 1D or 2D input, got {y.ndim}D")
    NUFFTOperator = MockNUFFTOperator # Use the mock if import failed


from reconlib.wavelets_scratch import (
    WaveletRegularizationTerm, 
    NUFFTWaveletRegularizedReconstructor
)

# --- Setup and Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Image parameters
image_size = (64, 64)  # H, W
num_coils = 1 # For simplicity in this example

# NUFFT parameters
num_spokes = 32
samples_per_spoke = image_size[0] * 2 # Oversampled radial
# Define NUFFT parameters more explicitly for clarity and consistency
oversamp_factor = (2.0, 2.0)
kb_J_param = (6, 6)  # Using _param to avoid potential name collisions
# kb_alpha: Kaiser-Bessel alpha parameter.
# Common alternatives include values around 2.34 * kb_J for os=2.0.
kb_alpha_param = tuple(k * os for k, os in zip(kb_J_param, oversamp_factor))
# Kd_param: Dimensions of the oversampled Cartesian grid for NUFFT.
Kd_param = tuple(int(im_s * os) for im_s, os in zip(image_size, oversamp_factor))
# Ld_param: Size of the Kaiser-Bessel interpolation lookup table.
Ld_param = (1024, 1024) # A common default for 2D
kb_m_param = (0.0,0.0)


# Reconstruction parameters
lambda_reg_val = 0.005 # Adjusted for potentially noisy mock NUFFT
n_iter_val = 20
step_size_val = 0.5 # May need tuning based on NUFFT op scaling

# Wavelet choices
wavelet_names_to_test = ['haar', 'db4'] # db4 might not be in ALL_WAVELET_FILTERS if pywt failed

# --- Helper Functions ---
def generate_simple_phantom(size, dev):
    H, W = size
    phantom = torch.zeros(H, W, device=dev)
    # Create a circle
    center_x, center_y = W // 2, H // 2
    radius = min(H, W) // 3
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing='ij')
    mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 < radius**2
    phantom[mask] = 1.0
    # Add another smaller circle with different intensity
    radius2 = radius // 2
    mask2 = (x_coords - center_x + radius//2)**2 + (y_coords - center_y)**2 < radius2**2
    phantom[mask2] = 0.5
    return phantom

def generate_radial_trajectory(num_spokes_traj, samples_per_spoke_traj, img_size, dev):
    angles = torch.linspace(0, np.pi, num_spokes_traj, endpoint=False, device=dev)
    radii = torch.linspace(-0.5, 0.5, samples_per_spoke_traj, device=dev)
    k_traj_polar = torch.stack(torch.meshgrid(radii, angles, indexing='ij'), dim=-1) # samples, spokes, 2
    k_traj_complex = k_traj_polar[..., 0] * torch.exp(1j * k_traj_polar[..., 1]) # r * e^(i*theta)
    k_x = k_traj_complex.real.flatten()
    k_y = k_traj_complex.imag.flatten()
    # Stack kx and ky: shape (num_total_k_points, 2)
    return torch.stack([k_x, k_y], dim=-1)

def generate_dummy_sensitivity_maps(coils, size, dev):
    H, W = size
    if coils == 1:
        return torch.ones(1, H, W, device=dev, dtype=torch.complex64)
    else:
        # Simple linear gradient maps for multi-coil (for future extension if needed)
        maps = torch.zeros(coils, H, W, device=dev, dtype=torch.complex64)
        for i in range(coils):
            # Create varied gradients
            grad_x = torch.linspace(0.1, 1.0 - 0.1*i, W, device=dev)
            grad_y = torch.linspace(0.1 + 0.1*i, 1.0, H, device=dev)
            map_2d = torch.sqrt(grad_y.unsqueeze(1) * grad_x.unsqueeze(0))
            maps[i] = map_2d * torch.exp(1j * np.pi * i / coils) # Add some phase variation
        # Normalize: sum_coils |map_i|^2 = 1 (approx)
        maps_sos = torch.sqrt(torch.sum(torch.abs(maps)**2, dim=0, keepdim=True))
        maps = maps / (maps_sos + 1e-8) # Avoid division by zero
        return maps


# --- Main Loop ---
for wavelet_name in wavelet_names_to_test:
    print(f"\n--- Running Reconstruction for Wavelet: {wavelet_name} ---")

    # Generate Data
    phantom = generate_simple_phantom(image_size, device).to(torch.complex64)
    k_trajectory = generate_radial_trajectory(num_spokes, samples_per_spoke, image_size, device)
    sensitivity_maps = generate_dummy_sensitivity_maps(num_coils, image_size, device)

    # Instantiate NUFFT Operator
    # Instantiate NUFFT Operator
    nufft_op_args = {
        'oversamp_factor': oversamp_factor, # Use the globally defined one
        'kb_J': kb_J_param,
        'kb_alpha': kb_alpha_param,
        'Ld': Ld_param,
        'Kd': Kd_param,
        'kb_m': kb_m_param
        # device is passed directly to constructor
    }
    # The TypeError fallback is removed as we are now providing the expected detailed parameters.
    nufft_op = NUFFTOperator(k_trajectory=k_trajectory, image_shape=image_size, device=device, **nufft_op_args)


    # Instantiate Wavelet Regularizer and Reconstructor
    try:
        wave_reg = WaveletRegularizationTerm(lambda_reg=lambda_reg_val, wavelet_name=wavelet_name, level=3, device=device)
    except KeyError:
        print(f"Wavelet '{wavelet_name}' not found in ALL_WAVELET_FILTERS (loaded in wavelets_scratch.py). Skipping.")
        continue
        
    reconstructor = NUFFTWaveletRegularizedReconstructor(
        nufft_op=nufft_op, 
        wavelet_regularizer=wave_reg, 
        n_iter=n_iter_val, 
        step_size=step_size_val
    )

    # Simulate K-Space Data
    # For single coil, NUFFTOperator typically expects (H,W) image
    image_for_nufft = phantom * sensitivity_maps.squeeze(0) # (H,W)
    true_kspace = nufft_op.op(image_for_nufft) # Should return (K,)

    # (Optional) Add noise
    noise_level = 0.01 * torch.mean(torch.abs(true_kspace)) * (torch.randn_like(true_kspace.real) + 1j * torch.randn_like(true_kspace.real))
    noisy_kspace = true_kspace + noise_level
    
    # Prepare kspace for reconstructor (expects coil dimension)
    kspace_input_recon = noisy_kspace.unsqueeze(0) # (1, K)

    # Initial Image Estimate
    # For single coil, op_adj takes (K,) returns (H,W)
    initial_estimate = nufft_op.op_adj(noisy_kspace) 
    # initial_estimate is (H,W), SENSE combination not strictly needed for num_coils=1 with ones smap
    # but the reconstructor expects it, so it will multiply by smap.conj() which is ones.

    print(f"Phantom shape: {phantom.shape}, dtype: {phantom.dtype}")
    print(f"K-space shape (simulated): {true_kspace.shape}, dtype: {true_kspace.dtype}")
    print(f"Sensitivity maps shape: {sensitivity_maps.shape}, dtype: {sensitivity_maps.dtype}")
    print(f"Initial estimate shape: {initial_estimate.shape}, dtype: {initial_estimate.dtype}")


    # Run Reconstruction
    print("Starting reconstruction...")
    reconstructed_image = reconstructor.forward(
        kspace_data=kspace_input_recon, 
        sensitivity_maps=sensitivity_maps, 
        initial_image_estimate=initial_estimate.clone() # Pass a clone
    )
    print("Reconstruction finished.")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(torch.abs(phantom).cpu().numpy(), cmap='gray')
    axes[0].set_title('Original Phantom')
    axes[0].axis('off')

    axes[1].imshow(torch.abs(reconstructed_image).cpu().numpy(), cmap='gray')
    axes[1].set_title(f'Reconstructed ({wavelet_name})')
    axes[1].axis('off')

    diff_image = torch.abs(torch.abs(reconstructed_image) - torch.abs(phantom))
    axes[2].imshow(diff_image.cpu().numpy(), cmap='magma')
    axes[2].set_title('Difference Image')
    axes[2].axis('off')
    
    plt.suptitle(f"Wavelet-Regularized NUFFT Recon Example ({wavelet_name})")
    plt.tight_layout()
    plt.show()

    # Assertions
    assert reconstructed_image.shape == phantom.shape, \
        f"Shape mismatch: Recon {reconstructed_image.shape}, Phantom {phantom.shape}"
    assert reconstructed_image.dtype == phantom.dtype, \
        f"Dtype mismatch: Recon {reconstructed_image.dtype}, Phantom {phantom.dtype}"
    assert not torch.isnan(reconstructed_image).any(), "NaNs found in reconstructed image."
    
    print(f"Assertions passed for {wavelet_name}.")

print("\nExample script finished.")

