import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add reconlib to path - Adjust if your environment handles this differently
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try-except for NUFFT operators and other components for robustness
try:
    from reconlib import NUFFTOperator
except ImportError:
    print("NUFFTOperator not found. Using a MockNUFFTOperator for demonstration.")
    class MockNUFFTOperator: # Placeholder
        def __init__(self, k_trajectory, image_shape, device='cpu', oversamp_factor=(2.0,2.0), 
                     kb_J=(6,6), kb_alpha=(12,12), Ld=(128,128), **kwargs):
            self.k_trajectory = k_trajectory
            self.image_shape = image_shape
            self.device = device
            print(f"MockNUFFTOperator initialized for image shape {image_shape} on device {device}.")
        def op(self, x: torch.Tensor) -> torch.Tensor: 
            num_k_points = self.k_trajectory.shape[0]
            return torch.randn(num_k_points, dtype=torch.complex64, device=self.device) * torch.mean(torch.abs(x))
        def op_adj(self, y: torch.Tensor) -> torch.Tensor: 
            return torch.randn(self.image_shape, dtype=torch.complex64, device=self.device) * torch.mean(torch.abs(y))
    NUFFTOperator = MockNUFFTOperator

try:
    from reconlib import MultiCoilNUFFTOperator
except ImportError:
    print("MultiCoilNUFFTOperator not found. Using a Mock version.")
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
        def op(self, multi_coil_image_data: torch.Tensor) -> torch.Tensor:
            num_coils = multi_coil_image_data.shape[0]
            num_kpoints = self.single_coil_nufft_op.k_trajectory.shape[0]
            return torch.randn(num_coils, num_kpoints, dtype=torch.complex64, device=self.device)
        def op_adj(self, multi_coil_kspace_data: torch.Tensor) -> torch.Tensor:
            num_coils = multi_coil_kspace_data.shape[0]
            return torch.randn(num_coils, *self.image_shape, dtype=torch.complex64, device=self.device)
    MultiCoilNUFFTOperator = MockMultiCoilNUFFTOperator

from reconlib import WaveletTransform, WaveletRegularizationTerm
from reconlib.regularizers.common import TVRegularizer # Assuming this path is correct
from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor

# --- Setup and Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

image_size = (64, 64)
num_coils = 4
num_spokes = image_size[0] // 2
samples_per_spoke = image_size[0] * 2
oversamp_factor = (2.0, 2.0)
kb_J = (6, 6)
kb_alpha = tuple(k * os for k, os in zip(kb_J, oversamp_factor))
Ld_grid_size = tuple(int(im_s * os) for im_s, os in zip(image_size, oversamp_factor))

iterations_recon = 30
step_size_recon = 0.2 # Adjusted step size

# --- Helper Functions ---
def generate_simple_phantom(size, dev):
    H, W = size
    phantom = torch.zeros(H, W, device=dev, dtype=torch.complex64)
    center_x, center_y = W // 2, H // 2; radius1, radius2 = min(H, W)//3, min(H,W)//5
    y_coords, x_coords = torch.meshgrid(torch.arange(H,dev), torch.arange(W,dev), indexing='ij')
    phantom[(x_coords-center_x)**2 + (y_coords-center_y)**2 < radius1**2] = 1.0
    phantom[(x_coords-(center_x+radius1//2))**2 + (y_coords-(center_y+radius1//2))**2 < radius2**2] = 0.5
    return phantom

def generate_radial_trajectory(num_spokes_traj, samples_per_spoke_traj, img_size, dev):
    angles = torch.linspace(0,np.pi,num_spokes_traj,endpoint=False,device=dev)
    radii = torch.linspace(-0.5,0.5,samples_per_spoke_traj,device=dev)
    k_radii,k_angles = torch.meshgrid(radii,angles,indexing='ij')
    k_x,k_y = k_radii*torch.cos(k_angles),k_radii*torch.sin(k_angles)
    return torch.stack([k_x.flatten(),k_y.flatten()],dim=-1)

def generate_analytical_espirit_maps(image_s, num_c, dev):
    H,W=image_s; maps=torch.zeros(num_c,H,W,device=dev,dtype=torch.complex64)
    x,y=torch.linspace(-np.pi,np.pi,W,device=dev),torch.linspace(-np.pi,np.pi,H,device=dev)
    for c in range(num_c):
        pxo,pyo=(c-num_c/2.)*0.5,(c-num_c/2.)*0.3
        phase=x.unsqueeze(0)*pxo+y.unsqueeze(1)*pyo+(np.pi/num_c*c)
        mag_x,mag_y=torch.cos(x*0.2*(c-num_c/2.+1))*.2+.8,torch.sin(y*0.3*(c-num_c/2.+1))*.2+.8
        mag=(mag_y.unsqueeze(1)*mag_x.unsqueeze(0)); mag=(mag-mag.min())/(mag.max()-mag.min()+1e-8)
        maps[c]=mag*torch.exp(1j*phase)
    return maps

# --- Main Script Logic ---
print("\n--- Generic Reconstructor Demo with TV and Wavelet Regularizers ---")

# Generate Common Data
phantom = generate_simple_phantom(image_size, device)
k_trajectory = generate_radial_trajectory(num_spokes, samples_per_spoke, image_size, device)
sensitivity_maps = generate_analytical_espirit_maps(image_size, num_coils, device)

base_nufft_op_params = {'k_trajectory':k_trajectory,'image_shape':image_size,'oversamp_factor':oversamp_factor,
                        'kb_J':kb_J,'kb_alpha':kb_alpha,'Ld':Ld_grid_size,'device':device}
try:
    base_nufft_op = NUFFTOperator(**base_nufft_op_params)
except TypeError: # Fallback if some params are not in mock/actual
    base_nufft_op = NUFFTOperator(k_trajectory, image_size, device=device)
mc_nufft_op = MultiCoilNUFFTOperator(base_nufft_op)

true_kspace = mc_nufft_op.op(sensitivity_maps * phantom.unsqueeze(0))
noise = (torch.randn_like(true_kspace.real)+1j*torch.randn_like(true_kspace.real)) * 0.05 * torch.mean(torch.abs(true_kspace)) / np.sqrt(2)
noisy_kspace = true_kspace + noise

# Define SENSE Forward and Adjoint Functions
def forward_op_sense(image_2d, smaps_3d): return mc_nufft_op.op(smaps_3d * image_2d.unsqueeze(0))
def adjoint_op_sense(kspace_coils_2d, smaps_3d): return torch.sum(mc_nufft_op.op_adj(kspace_coils_2d) * smaps_3d.conj(), dim=0)

# Initial Estimate
initial_estimate = adjoint_op_sense(noisy_kspace, sensitivity_maps)

# Instantiate Reconstructor
pg_reconstructor = ProximalGradientReconstructor(iterations=iterations_recon, step_size=step_size_recon, verbose=True)

# --- TV Regularization ---
print("\n--- Reconstruction with TV Regularizer ---")
tv_lambda = 0.01 
tv_regularizer = TVRegularizer(lambda_param=tv_lambda, device=device, n_iter=5) # Pass lambda here
recon_tv = pg_reconstructor.reconstruct(
    kspace_data=noisy_kspace,
    sensitivity_maps=sensitivity_maps,
    forward_op_fn=forward_op_sense,
    adjoint_op_fn=adjoint_op_sense,
    regularizer_prox_fn=tv_regularizer.proximal_operator, # Pass the method itself
    x_init=initial_estimate.clone()
)
print("TV Reconstruction finished.")

# --- Wavelet Regularization ---
print("\n--- Reconstruction with Wavelet Regularizer ---")
wavelet_lambda = 0.005 
wavelet_tf = WaveletTransform(wavelet_name='db4', level=3, device=device)
wavelet_regularizer = WaveletRegularizationTerm(lambda_reg=wavelet_lambda, wavelet_transform_op=wavelet_tf) # Pass lambda here
recon_wavelet = pg_reconstructor.reconstruct(
    kspace_data=noisy_kspace,
    sensitivity_maps=sensitivity_maps,
    forward_op_fn=forward_op_sense,
    adjoint_op_fn=adjoint_op_sense,
    regularizer_prox_fn=wavelet_regularizer.proximal_operator, # Pass the method itself
    x_init=initial_estimate.clone()
)
print("Wavelet Reconstruction finished.")

# Visualization
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Generic Reconstructor Demo: TV vs. Wavelet", fontsize=16)
vmax_plot = torch.abs(phantom).max().item()

axes[0].imshow(torch.abs(phantom).cpu().numpy(), cmap='gray', vmin=0, vmax=vmax_plot)
axes[0].set_title('Original Phantom'); axes[0].axis('off')
axes[1].imshow(torch.abs(initial_estimate).cpu().numpy(), cmap='gray', vmin=0, vmax=vmax_plot)
axes[1].set_title('Initial Estimate'); axes[1].axis('off')
axes[2].imshow(torch.abs(recon_tv).cpu().numpy(), cmap='gray', vmin=0, vmax=vmax_plot)
axes[2].set_title(f'TV Recon (λ={tv_lambda})'); axes[2].axis('off')
axes[3].imshow(torch.abs(recon_wavelet).cpu().numpy(), cmap='gray', vmin=0, vmax=vmax_plot)
axes[3].set_title(f'Wavelet Recon (λ={wavelet_lambda}, db4)'); axes[3].axis('off')

plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust for suptitle and potential colorbars
plt.show()

# Assertions
assert recon_tv.shape == phantom.shape
assert not torch.isnan(recon_tv).any(), "NaNs in TV reconstruction."
assert recon_wavelet.shape == phantom.shape
assert not torch.isnan(recon_wavelet).any(), "NaNs in Wavelet reconstruction."

print("\nGeneric reconstructor demo script finished successfully.")
print("Note: If using Mock operators, results are illustrative only.")

```
