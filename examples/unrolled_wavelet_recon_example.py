import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# This allows running the example directly from the 'examples' folder.
# For general use, it's recommended to install reconlib (e.g., `pip install -e .` from root).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try-except for NUFFT operators and other components for robustness in different environments
try:
    from reconlib.operators import NUFFTOperator # Corrected import
except ImportError:
    print("\n*****************************************************************")
    print("WARNING: NUFFTOperator not found in reconlib.operators.")
    print("         Using a MOCK NUFFTOperator for demonstration.")
    print("         RESULTS WILL BE RANDOM AND NOT MEANINGFUL.")
    print("*****************************************************************\n")
    class MockNUFFTOperator: # Placeholder
        def __init__(self, k_trajectory, image_shape, device='cpu', oversamp_factor=(2.0,2.0), 
                     kb_J=(6,6), kb_alpha=(12,12), Ld=(128,128), Kd=None, kb_m=(0.0,0.0), **kwargs): # Added kb_m
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

        def op(self, x: torch.Tensor) -> torch.Tensor: # Expects (H, W)
            num_k_points = self.k_trajectory.shape[0]
            return torch.randn(num_k_points, dtype=torch.complex64, device=self.device) * torch.mean(torch.abs(x))

        def op_adj(self, y: torch.Tensor) -> torch.Tensor: # Expects (K,)
            return torch.randn(self.image_shape, dtype=torch.complex64, device=self.device) * torch.mean(torch.abs(y))
    NUFFTOperator = MockNUFFTOperator

try:
    from reconlib.nufft_multi_coil import MultiCoilNUFFTOperator
except ImportError: # This was already corrected in the file, but ensure the warning style matches
    print("\n*****************************************************************")
    print("WARNING: MultiCoilNUFFTOperator not found in reconlib.nufft_multi_coil.")
    print("         Using a MOCK MultiCoilNUFFTOperator for demonstration.")
    print("         RESULTS WILL BE RANDOM AND NOT MEANINGFUL.")
    print("*****************************************************************\n")
    class MockOperatorBase: 
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

from reconlib.wavelets_scratch import WaveletTransform # Corrected import
from reconlib.deeplearning.denoisers import SimpleWaveletDenoiser # Corrected import
from reconlib.deeplearning.unrolled import LearnedRegularizationIteration # Corrected import


# --- Setup and Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Image parameters
image_size = (64, 64)  # H, W
num_coils = 4

# NUFFT parameters (same as espirit_wavelet_recon_example.py)
num_spokes = image_size[0] // 2
samples_per_spoke = image_size[0] * 2
oversamp_factor = (2.0, 2.0)
kb_J = (6, 6) 
# kb_alpha: Kaiser-Bessel alpha parameter. This example uses kb_J * oversamp_factor.
# Common alternatives include values around 2.34 * kb_J for os=2.0, or pi * kb_J.
# The optimal value depends on the specific NUFFT implementation details.
kb_alpha = tuple(k * os for k, os in zip(kb_J, oversamp_factor)) 

# Kd_oversampled_dims: Dimensions of the oversampled Cartesian grid for NUFFT.
Kd_oversampled_dims = tuple(int(im_s * os) for im_s, os in zip(image_size, oversamp_factor))
# Ld_table_length: Size of the Kaiser-Bessel interpolation lookup table.
Ld_table_length = (1024, 1024) # A common default for 2D

# Unrolled iteration parameters
eta_init = 0.1
num_unrolled_iterations = 5 # Fixed number for this example

# Wavelet settings
wavelet_name = 'db4'
wavelet_level = 3

# --- Helper Functions (copied/adapted from espirit_wavelet_recon_example.py) ---
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
    radii = torch.linspace(-0.5, 0.5, samples_per_spoke_traj, device=dev)
    k_radii, k_angles = torch.meshgrid(radii, angles, indexing='ij')
    k_x = k_radii * torch.cos(k_angles)
    k_y = k_radii * torch.sin(k_angles)
    return torch.stack([k_x.flatten(), k_y.flatten()], dim=-1)

def generate_analytical_espirit_maps(image_s, num_c, dev):
    H, W = image_s
    maps = torch.zeros(num_c, H, W, device=dev, dtype=torch.complex64)
    x_coords = torch.linspace(-np.pi, np.pi, W, device=dev)
    y_coords = torch.linspace(-np.pi, np.pi, H, device=dev)
    for c in range(num_c):
        phase_offset_x = (c - num_c / 2.0) * 0.5 
        phase_offset_y = (c - num_c / 2.0) * 0.3
        map_phase = x_coords.unsqueeze(0) * phase_offset_x + y_coords.unsqueeze(1) * phase_offset_y + (np.pi / num_c * c)
        mag_x_variation = torch.cos(x_coords * 0.2 * (c - num_c / 2.0 + 1)) * 0.2 + 0.8
        mag_y_variation = torch.sin(y_coords * 0.3 * (c - num_c / 2.0 + 1)) * 0.2 + 0.8
        map_mag = mag_y_variation.unsqueeze(1) * mag_x_variation.unsqueeze(0)
        map_mag = (map_mag - map_mag.min()) / (map_mag.max() - map_mag.min() + 1e-8)
        maps[c] = map_mag * torch.exp(1j * map_phase)
    return maps

# --- Main Script Logic ---
# Note: This example runs a fixed number of unrolled iterations.
# The 'LearnedRegularizationIteration' module's parameters (like eta or denoiser weights)
# are used with their initial/fixed values here and are not trained within this script.
# A separate training script would be needed to learn these parameters.
print(f"\n--- Unrolled Wavelet Recon Example ---")
print(f"Wavelet: {wavelet_name}, Level: {wavelet_level}, Eta_init: {eta_init}")
print(f"Unrolled Iterations: {num_unrolled_iterations}")

# Generate Data
phantom = generate_simple_phantom(image_size, device)
k_trajectory = generate_radial_trajectory(num_spokes, samples_per_spoke, image_size, device)
sensitivity_maps = generate_analytical_espirit_maps(image_size, num_coils, device)

# Instantiate Operators and Modules
base_nufft_op_params = {
    'k_trajectory': k_trajectory, 
    'image_shape': image_size, 
    'oversamp_factor': oversamp_factor, 
    'kb_J': kb_J, 
    'kb_alpha': kb_alpha, 
    'Ld': Ld_table_length,      # Use table length
    'Kd': Kd_oversampled_dims,  # Use oversampled grid dimensions
    'device': device
}
try:
    base_nufft_op = NUFFTOperator(**base_nufft_op_params)
except TypeError as e:
    print(f"NUFFTOperator instantiation failed: {e}. Ensure params match actual operator, including Kd and Ld.")
    sys.exit(1)
    
mc_nufft_op = MultiCoilNUFFTOperator(base_nufft_op)

try:
    wavelet_tf = WaveletTransform(wavelet_name=wavelet_name, level=wavelet_level, device=device)
    simple_denoiser = SimpleWaveletDenoiser(wavelet_transform_op=wavelet_tf)
    simple_denoiser.to(device) # Ensure denoiser is on device

    unrolled_iter_module = LearnedRegularizationIteration(
        nufft_op=mc_nufft_op,
        wavelet_transform_op=wavelet_tf,
        denoiser_module=simple_denoiser,
        eta_init=eta_init
    )
    unrolled_iter_module.to(device) # Move iteration module to device
except KeyError: # Handles if wavelet_name is not in ALL_WAVELET_FILTERS
    print(f"Wavelet '{wavelet_name}' not available. Check wavelets_scratch.py.")
    sys.exit(1)
except Exception as e:
    print(f"Error during module instantiation: {e}")
    sys.exit(1)

# Simulate Multi-Coil K-Space Data
coil_images = sensitivity_maps * phantom.unsqueeze(0)
y_true_kspace = mc_nufft_op.op(coil_images)

# Add complex Gaussian noise
noise_level_factor = 0.05 
noise_std_per_sample = torch.mean(torch.abs(y_true_kspace)) * noise_level_factor
noise = (torch.randn_like(y_true_kspace.real) + 1j * torch.randn_like(y_true_kspace.real)) * noise_std_per_sample / np.sqrt(2)
y_noisy_kspace = y_true_kspace + noise

# Initial Image Estimate
img_adj_coils = mc_nufft_op.op_adj(y_noisy_kspace)
x_current = torch.sum(img_adj_coils * sensitivity_maps.conj(), dim=0)

print(f"Phantom shape: {phantom.shape}")
print(f"Sensitivity maps shape: {sensitivity_maps.shape}")
print(f"Multi-coil k-space (noisy) shape: {y_noisy_kspace.shape}")
print(f"Initial estimate (x_current) shape: {x_current.shape}")

# Run Fixed Unrolled Iterations
intermediate_recons = [x_current.clone().cpu()] # Store CPU copies for plotting

print("Starting unrolled iterations...")
for iter_num in range(num_unrolled_iterations):
    x_next = unrolled_iter_module.forward(x_current, y_noisy_kspace, sensitivity_maps)
    x_current = x_next
    intermediate_recons.append(x_current.clone().cpu())
    print(f"Iteration {iter_num+1}/{num_unrolled_iterations} complete. Eta: {unrolled_iter_module.eta.item():.4f}")

print("Unrolled iterations finished.")

# Visualization
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle(f"Unrolled Wavelet Recon ({wavelet_name}, {num_unrolled_iterations} iters, $\eta_{{final}}={unrolled_iter_module.eta.item():.3f}$)", fontsize=16)

# Original Phantom
axes[0].imshow(torch.abs(phantom).cpu().numpy(), cmap='gray', vmin=0, vmax=torch.abs(phantom).max().item())
axes[0].set_title('Original Phantom')
axes[0].axis('off')

# Initial Estimate
axes[1].imshow(torch.abs(intermediate_recons[0]).cpu().numpy(), cmap='gray', vmin=0, vmax=torch.abs(phantom).max().item())
axes[1].set_title('Initial Estimate (Adjoint)')
axes[1].axis('off')

# Reconstruction after N iterations
final_recon = intermediate_recons[-1]
axes[2].imshow(torch.abs(final_recon).cpu().numpy(), cmap='gray', vmin=0, vmax=torch.abs(phantom).max().item())
axes[2].set_title(f'Reconstruction ({num_unrolled_iterations} iters)')
axes[2].axis('off')

# Difference Image
diff_image = torch.abs(torch.abs(phantom.cpu()) - torch.abs(final_recon))
im = axes[3].imshow(diff_image.numpy(), cmap='magma')
axes[3].set_title('Difference Image')
axes[3].axis('off')
plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Assertions
assert final_recon.shape == phantom.shape, \
    f"Shape mismatch: Final Recon {final_recon.shape}, Phantom {phantom.shape}"
assert not torch.isnan(final_recon).any(), "NaNs found in final reconstructed image."

print(f"\nUnrolled reconstruction example with {wavelet_name} completed successfully.")
print("Note: If using MockNUFFTOperator, results are illustrative only. Denoiser was not trained.")

```
