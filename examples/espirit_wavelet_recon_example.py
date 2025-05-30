import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add reconlib to path - Adjust if your environment handles this differently
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from reconlib.operators import NUFFTOperator
except ImportError:
    print("NUFFTOperator not found in reconlib.operators. Using a MockNUFFTOperator for demonstration.")
    class MockNUFFTOperator: # Placeholder
        def __init__(self, k_trajectory, image_shape, device='cpu', oversamp_factor=(2.0,2.0), 
                     kb_J=(6,6), kb_alpha=(12,12), Ld=(128,128), **kwargs):
            self.k_trajectory = k_trajectory
            self.image_shape = image_shape
            self.device = device
            self.oversamp_factor = oversamp_factor
            self.kb_J = kb_J
            self.kb_alpha = kb_alpha
            self.Ld = Ld # Grid size
            print(f"MockNUFFTOperator initialized for image shape {image_shape} on device {device}.")
            print(f"  K-traj shape: {k_trajectory.shape}, OS: {oversamp_factor}, Kernel J: {kb_J}, Kernel Alpha: {kb_alpha}, Grid Ld: {Ld}")

        def op(self, x: torch.Tensor) -> torch.Tensor: # Expects (H, W)
            num_k_points = self.k_trajectory.shape[0]
            return torch.randn(num_k_points, dtype=torch.complex64, device=self.device) * torch.mean(torch.abs(x))

        def op_adj(self, y: torch.Tensor) -> torch.Tensor: # Expects (K,)
            return torch.randn(self.image_shape, dtype=torch.complex64, device=self.device) * torch.mean(torch.abs(y))
    NUFFTOperator = MockNUFFTOperator

try:
    from reconlib.nufft_multi_coil import MultiCoilNUFFTOperator
except ImportError:
    print("MultiCoilNUFFTOperator not found. Using a Mock version.")
    class MockOperatorBase: # Simplified base if reconlib.operators.Operator is also missing
        def __init__(self): self.device = torch.device('cpu')
        def op(self, x): raise NotImplementedError
        def op_adj(self, y): raise NotImplementedError

    class MockMultiCoilNUFFTOperator(MockOperatorBase):
        def __init__(self, single_coil_nufft_op):
            super().__init__()
            self.single_coil_nufft_op = single_coil_nufft_op
            self.device = single_coil_nufft_op.device
            self.image_shape = single_coil_nufft_op.image_shape
            self.k_trajectory = getattr(single_coil_nufft_op, 'k_trajectory', None)

        def op(self, multi_coil_image_data: torch.Tensor) -> torch.Tensor:
            num_coils = multi_coil_image_data.shape[0]
            num_kpoints = self.single_coil_nufft_op.k_trajectory.shape[0]
            return torch.randn(num_coils, num_kpoints, dtype=torch.complex64, device=self.device)

        def op_adj(self, multi_coil_kspace_data: torch.Tensor) -> torch.Tensor:
            num_coils = multi_coil_kspace_data.shape[0]
            return torch.randn(num_coils, *self.image_shape, dtype=torch.complex64, device=self.device)
    MultiCoilNUFFTOperator = MockMultiCoilNUFFTOperator


from reconlib.wavelets_scratch import (
    WaveletRegularizationTerm, 
    NUFFTWaveletRegularizedReconstructor
)

# --- Setup and Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Image parameters
image_size = (64, 64)  # H, W
num_coils = 4

# NUFFT parameters
num_spokes = image_size[0] // 2  # e.g., 32
samples_per_spoke = image_size[0] * 2 # e.g., 128
oversamp_factor = (2.0, 2.0)
kb_J = (6, 6) # Kernel width for Kaiser-Bessel
# kb_alpha calculation as specified, or common alternatives like 2.34 * J_val
# Using the formula provided: kb_alpha = tuple(k * os for k, os in zip(kb_J, oversamp_factor))
# However, a more standard approach for kb_alpha might be related to pi and J.
# Let's use a common approximation if the above is not standard for the NUFFTOperator implementation:
# kb_alpha = tuple(np.pi * j for j in kb_J) # Or another typical calculation
# For now, sticking to the prompt's specified calculation, assuming it aligns with the NUFFTOperator's expectation.
kb_alpha = tuple(k * os for k, os in zip(kb_J, oversamp_factor)) 
Ld_grid_size = tuple(int(im_s * os) for im_s, os in zip(image_size, oversamp_factor))


# Reconstruction parameters
lambda_reg = 0.005 
n_iter = 30        # Iterations for reconstruction
step_size = 0.5    # Step size for gradient descent

# Wavelet parameters
wavelet_name = 'db4' # Example: Daubechies 4
level = 3            # Decomposition level

# --- Helper Functions ---
def generate_simple_phantom(size, dev):
    H, W = size
    phantom = torch.zeros(H, W, device=dev, dtype=torch.complex64)
    center_x, center_y = W // 2, H // 2
    radius1 = min(H, W) // 3
    radius2 = min(H, W) // 5
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing='ij')
    
    mask1 = (x_coords - center_x)**2 + (y_coords - center_y)**2 < radius1**2
    phantom[mask1] = 1.0
    
    mask2 = (x_coords - (center_x + radius1//2))**2 + (y_coords - (center_y + radius1//2))**2 < radius2**2
    phantom[mask2] = 0.5
    
    mask3 = (x_coords - (center_x - radius1//2))**2 + (y_coords - (center_y - radius1//2))**2 < radius2**2
    phantom[mask3] = 0.75
    return phantom

def generate_radial_trajectory(num_spokes_traj, samples_per_spoke_traj, img_size, dev):
    angles = torch.linspace(0, np.pi, num_spokes_traj, endpoint=False, device=dev)
    radii = torch.linspace(-0.5, 0.5, samples_per_spoke_traj, device=dev) # K-space normalized to [-0.5, 0.5]
    
    # Create polar grid and convert to Cartesian
    k_radii, k_angles = torch.meshgrid(radii, angles, indexing='ij')
    k_x = k_radii * torch.cos(k_angles)
    k_y = k_radii * torch.sin(k_angles)
    
    # Stack kx and ky: shape (num_total_k_points, 2)
    return torch.stack([k_x.flatten(), k_y.flatten()], dim=-1)


def generate_analytical_espirit_maps(image_s, num_c, dev):
    H, W = image_s
    maps = torch.zeros(num_c, H, W, device=dev, dtype=torch.complex64)
    x_coords = torch.linspace(-np.pi, np.pi, W, device=dev)
    y_coords = torch.linspace(-np.pi, np.pi, H, device=dev)
    
    for c in range(num_c):
        # Create varying phase patterns for each coil
        phase_offset_x = (c - num_c / 2.0) * 0.5 # Adjust factor for phase variation speed
        phase_offset_y = (c - num_c / 2.0) * 0.3
        
        phase_x_map = x_coords.unsqueeze(0) * phase_offset_x
        phase_y_map = y_coords.unsqueeze(1) * phase_offset_y
        
        map_phase = phase_x_map + phase_y_map + (np.pi / num_c * c) # Add coil-specific phase shift
        
        # Create magnitude that varies slowly
        mag_x_variation = torch.cos(x_coords * 0.2 * (c - num_c / 2.0 + 1)) * 0.2 + 0.8
        mag_y_variation = torch.sin(y_coords * 0.3 * (c - num_c / 2.0 + 1)) * 0.2 + 0.8
        map_mag = mag_y_variation.unsqueeze(1) * mag_x_variation.unsqueeze(0)
        map_mag = (map_mag - map_mag.min()) / (map_mag.max() - map_mag.min() + 1e-8) # Normalize mag to [0,1]
        
        maps[c] = map_mag * torch.exp(1j * map_phase)

    # Optional: Normalize for SENSE Sum-of-Squares = 1 (approximately)
    # sos = torch.sqrt(torch.sum(torch.abs(maps)**2, dim=0, keepdim=True))
    # maps = maps / (sos + 1e-8) # Avoid division by zero
    return maps


# --- Main Script Logic ---
print(f"\n--- ESPIRiT-like Wavelet Recon Example ---")
print(f"Wavelet: {wavelet_name}, Level: {level}, Lambda: {lambda_reg}")
print(f"Iterations: {n_iter}, Step Size: {step_size}")

# Generate Data
phantom = generate_simple_phantom(image_size, device)
k_trajectory = generate_radial_trajectory(num_spokes, samples_per_spoke, image_size, device)
sensitivity_maps = generate_analytical_espirit_maps(image_size, num_coils, device)

# Instantiate NUFFT Operators
base_nufft_op_params = {
    'k_trajectory': k_trajectory, 
    'image_shape': image_size, 
    'oversamp_factor': oversamp_factor, 
    'kb_J': kb_J, 
    'kb_alpha': kb_alpha, 
    'Ld': Ld_grid_size, 
    'device': device
}
try:
    # These parameters are based on common NUFFT library interfaces (e.g., TorchKbNufft, SigPy)
    # Adjust if your NUFFTOperator has different parameter names or requirements
    single_coil_nufft = NUFFTOperator(**base_nufft_op_params)
except TypeError as e:
    print(f"NUFFTOperator instantiation failed with TypeError: {e}")
    print("This might be due to parameter name mismatch if using a non-mock NUFFTOperator.")
    print("Ensure base_nufft_op_params match the expected signature of your NUFFTOperator.")
    sys.exit(1)
    
multi_coil_nufft = MultiCoilNUFFTOperator(single_coil_nufft)

# Instantiate Reconstructor Components
try:
    wave_reg = WaveletRegularizationTerm(
        lambda_reg=lambda_reg, 
        wavelet_name=wavelet_name, 
        level=level, 
        device=device
    )
except KeyError:
    print(f"Wavelet '{wavelet_name}' not available. Check ALL_WAVELET_FILTERS in wavelets_scratch.py.")
    sys.exit(1)
    
reconstructor = NUFFTWaveletRegularizedReconstructor(
    nufft_op=multi_coil_nufft, 
    wavelet_regularizer=wave_reg, 
    n_iter=n_iter, 
    step_size=step_size
)

# Simulate Multi-Coil K-Space Data
coil_images = sensitivity_maps * phantom.unsqueeze(0)  # (num_coils, H, W)
y_coils = multi_coil_nufft.op(coil_images)             # Shape (num_coils, num_kpoints)

# Add complex Gaussian noise
noise_level_factor = 0.05 
noise_std_per_sample = torch.mean(torch.abs(y_coils)) * noise_level_factor
noise = (torch.randn_like(y_coils.real) + 1j * torch.randn_like(y_coils.real)) * noise_std_per_sample / np.sqrt(2) # Proper complex noise
y_coils_noisy = y_coils + noise

# Initial Image Estimate (SENSE-like combination)
img_adj_coils = multi_coil_nufft.op_adj(y_coils_noisy) # (num_coils, H, W)
initial_estimate = torch.sum(img_adj_coils * sensitivity_maps.conj(), dim=0) # Coil combination

print(f"Phantom shape: {phantom.shape}")
print(f"Sensitivity maps shape: {sensitivity_maps.shape}")
print(f"Coil images shape: {coil_images.shape}")
print(f"Multi-coil k-space (y_coils) shape: {y_coils.shape}")
print(f"Initial estimate shape: {initial_estimate.shape}")

# Run Reconstruction
print("Starting reconstruction...")
reconstructed_image = reconstructor.forward(
    kspace_data=y_coils_noisy, 
    sensitivity_maps=sensitivity_maps, 
    initial_image_estimate=initial_estimate.clone() # Pass a clone
)
print("Reconstruction finished.")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle(f"ESPIRiT-like Wavelet Recon ({wavelet_name}, $\lambda={lambda_reg}$)", fontsize=16)

# Original Phantom
axes[0, 0].imshow(torch.abs(phantom).cpu().numpy(), cmap='gray')
axes[0, 0].set_title('Original Phantom')
axes[0, 0].axis('off')

# Reconstructed Image
axes[0, 1].imshow(torch.abs(reconstructed_image).cpu().numpy(), cmap='gray')
axes[0, 1].set_title('Reconstructed Image')
axes[0, 1].axis('off')

# Sensitivity Map (Magnitude of Coil 0)
axes[1, 0].imshow(torch.abs(sensitivity_maps[0]).cpu().numpy(), cmap='viridis')
axes[1, 0].set_title('Sensitivity Map (Coil 0, Mag)')
axes[1, 0].axis('off')

# Difference Image
diff_image = torch.abs(torch.abs(phantom) - torch.abs(reconstructed_image))
im = axes[1, 1].imshow(diff_image.cpu().numpy(), cmap='magma')
axes[1, 1].set_title('Difference Image')
axes[1, 1].axis('off')
plt.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)


plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
plt.show()

# Assertions
assert reconstructed_image.shape == phantom.shape, \
    f"Shape mismatch: Recon {reconstructed_image.shape}, Phantom {phantom.shape}"
assert reconstructed_image.dtype == phantom.dtype, \
    f"Dtype mismatch: Recon {reconstructed_image.dtype}, Phantom {phantom.dtype}"
assert not torch.isnan(reconstructed_image).any(), "NaNs found in reconstructed image."

print(f"\nESPIRiT-like example with {wavelet_name} completed successfully.")
print("Note: If using MockNUFFTOperator, results are illustrative only.")
