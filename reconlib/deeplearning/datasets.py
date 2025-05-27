""" PyTorch Dataset classes for MRI reconstruction deep learning models. """
import torch
from torch.utils.data import Dataset
import numpy as np
import math
# Adjust path for reconlib components
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from reconlib.operators import NUFFTOperator
# Assuming iternufft.py is at the root or accessible in PYTHONPATH
# For robust import, iternufft's functions might need to be part of reconlib proper
try:
    from iternufft import generate_phantom_2d, generate_phantom_3d, generate_radial_trajectory_2d, generate_radial_trajectory_3d
except ImportError:
    print("ERROR: iternufft.py not found. Ensure it's in the PYTHONPATH or its functions are part of reconlib.")
    # Define dummy functions if iternufft is not found, so module can load for inspection
    def generate_phantom_2d(size, device): return torch.zeros((size,size), device=device)
    def generate_phantom_3d(shape, device): return torch.zeros(shape, device=device)
    def generate_radial_trajectory_2d(num_spokes, samples_per_spoke, device): return torch.zeros((num_spokes*samples_per_spoke, 2), device=device)
    def generate_radial_trajectory_3d(num_profiles_z, num_spokes_per_profile, samples_per_spoke, shape, device): return torch.zeros((num_profiles_z*num_spokes_per_profile*samples_per_spoke, 3), device=device)


class MoDLDataset(Dataset):
    """
    A PyTorch Dataset for training MoDL or other unrolled networks.
    Generates data on-the-fly:
    - Ground truth image (phantom)
    - Undersampled k-space data (using NUFFTOperator)
    - Initial reconstruction (zero-filled via NUFFTOperator.op_adj)
    """
    def __init__(self,
                 dataset_size: int,
                 image_shape: tuple[int, ...],
                 # NUFFTOperator parameters (passed directly or used to create one)
                 k_trajectory_func, # Function to generate k-space trajectory, e.g., generate_radial_trajectory_2d
                 k_trajectory_params: dict, # Params for k_trajectory_func
                 nufft_op_params: dict,     # Params for NUFFTOperator constructor (oversamp_factor, kb_J, etc.)
                 phantom_func, # Function to generate phantom, e.g., generate_phantom_2d
                 phantom_params: dict, # Params for phantom_func
                 noise_level_kspace: float = 0.00, # Relative noise level to add to k-space
                 device: str | torch.device = 'cpu'
                ):
        self.dataset_size = dataset_size
        self.image_shape = image_shape
        self.dim = len(image_shape)
        
        self.k_trajectory_func = k_trajectory_func
        self.k_trajectory_params = k_trajectory_params
        self.nufft_op_params = nufft_op_params
        self.phantom_func = phantom_func
        self.phantom_params = phantom_params
        self.noise_level_kspace = noise_level_kspace
        self.device = torch.device(device)

        # Generate a representative k-trajectory once for operator instantiation
        # This assumes the trajectory structure is fixed for all samples, 
        # though specific points could vary if k_trajectory_func had randomness without fixed seed.
        # For simplicity, we generate one trajectory here.
        self.k_traj = self.k_trajectory_func(device=self.device, **self.k_trajectory_params)
        
        self.nufft_op = NUFFTOperator(
            k_trajectory=self.k_traj,
            image_shape=self.image_shape,
            device=self.device,
            **self.nufft_op_params
        )
        print(f"MoDLDataset initialized with {self.dim}D NUFFT operator on {self.device}.")
        if self.dim == 2 and self.k_traj.shape[-1] != 2:
             raise ValueError("2D dataset, but k_traj last dim is not 2")
        if self.dim == 3 and self.k_traj.shape[-1] != 3:
             raise ValueError("3D dataset, but k_traj last dim is not 3")


    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            initial_recon_x0 (torch.Tensor): Initial reconstruction (e.g., A^H y). Shape: image_shape.
            k_space_observed_y (torch.Tensor): Undersampled k-space data. Shape: (num_k_points,).
            ground_truth_image_x_true (torch.Tensor): Fully sampled reference image. Shape: image_shape.
        """
        # 1. Generate ground truth image (phantom)
        # Add seed if phantom_func supports it for reproducibility, or handle randomness if desired
        current_phantom_params = self.phantom_params.copy()
        if 'size' in current_phantom_params and self.dim == 2: # For generate_phantom_2d
            current_phantom_params['size'] = self.image_shape[0] # Assuming square for 2D
        elif 'shape' in current_phantom_params and self.dim == 3: # For generate_phantom_3d
            current_phantom_params['shape'] = self.image_shape
            
        x_true = self.phantom_func(device=self.device, **current_phantom_params).to(torch.complex64)

        # 2. Simulate undersampled k-space data: y = A(x_true) + noise
        y_clean = self.nufft_op.op(x_true)
        
        if self.noise_level_kspace > 0:
            # Add complex Gaussian noise
            noise_std_val = self.noise_level_kspace * torch.mean(torch.abs(y_clean))
            noise = noise_std_val * (torch.randn_like(y_clean.real) + 1j * torch.randn_like(y_clean.real))
            y_observed = y_clean + noise
        else:
            y_observed = y_clean
            
        # 3. Compute initial reconstruction: x0 = A^H(y_observed)
        # Note: NUFFTOperator.op_adj already includes scaling factor for adjoint.
        # DCF should be applied to y_observed if NUFFT2D/3D adjoint expects it (current NUFFT2D does, NUFFT3D user applies)
        # For MoDL, A^H y is part of the DC step, so x0 can be simpler.
        # Often, x0 is just A^H y without explicit DCF for the *initial* input to network.
        # Let's provide A^H y. If NUFFT2D applies DCF internally in its op_adj, it will be applied.
        x0 = self.nufft_op.op_adj(y_observed)
        
        return x0, y_observed, x_true


if __name__ == '__main__':
    print("Testing MoDLDataset...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 2D Dataset Test ---
    print("\n--- Testing 2D MoDLDataset ---")
    image_shape_2d = (64, 64)
    k_traj_params_2d = {'num_spokes': 32, 'samples_per_spoke': 64}
    
    nufft_op_params_2d = {
        'oversamp_factor': (2.0, 2.0),
        'kb_J': (4, 4),
        'kb_alpha': tuple(2.34 * J for J in (4,4)),
        'Ld': (2**8, 2**8)
    }
    phantom_params_2d = {'size': image_shape_2d[0]} # For generate_phantom_2d

    dataset_2d = MoDLDataset(
        dataset_size=4,
        image_shape=image_shape_2d,
        k_trajectory_func=generate_radial_trajectory_2d,
        k_trajectory_params=k_traj_params_2d,
        nufft_op_params=nufft_op_params_2d,
        phantom_func=generate_phantom_2d,
        phantom_params=phantom_params_2d,
        noise_level_kspace=0.01,
        device=device
    )
    
    x0_2d, y_2d, xt_2d = dataset_2d[0]
    print(f"2D Example shapes: x0: {x0_2d.shape}, y: {y_2d.shape}, x_true: {xt_2d.shape}")
    assert x0_2d.shape == image_shape_2d
    assert xt_2d.shape == image_shape_2d
    assert y_2d.ndim == 1

    # --- 3D Dataset Test ---
    print("\n--- Testing 3D MoDLDataset ---")
    image_shape_3d = (16, 32, 32) # Small for test
    k_traj_params_3d = {
        'num_profiles_z': 16, 
        'num_spokes_per_profile': 16, 
        'samples_per_spoke': 32,
        'shape': image_shape_3d 
    }
    nufft_op_params_3d = {
        'oversamp_factor': (1.5, 1.5, 1.5),
        'kb_J': (4, 4, 4),
        'kb_alpha': tuple(2.34 * J for J in (4,4,4)),
        'Ld': (2**6, 2**6, 2**6), # Smaller table for faster test
        'nufft_type_3d': 'table' # Explicitly use table for 3D
    }
    phantom_params_3d = {'shape': image_shape_3d}

    dataset_3d = MoDLDataset(
        dataset_size=2,
        image_shape=image_shape_3d,
        k_trajectory_func=generate_radial_trajectory_3d,
        k_trajectory_params=k_traj_params_3d,
        nufft_op_params=nufft_op_params_3d,
        phantom_func=generate_phantom_3d,
        phantom_params=phantom_params_3d,
        noise_level_kspace=0.01,
        device=device
    )

    x0_3d, y_3d, xt_3d = dataset_3d[0]
    print(f"3D Example shapes: x0: {x0_3d.shape}, y: {y_3d.shape}, x_true: {xt_3d.shape}")
    assert x0_3d.shape == image_shape_3d
    assert xt_3d.shape == image_shape_3d
    assert y_3d.ndim == 1
    
    print("\nMoDLDataset basic tests completed.")
