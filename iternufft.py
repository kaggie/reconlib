# -*- coding: utf-8 -*-
"""
Created on Sat May 17 10:19:55 2025

@author: Josh
"""

import torch
import numpy as np
import math # Added for kb_alpha calculation
import matplotlib.pyplot as plt
from reconlib.operators import NUFFTOperator

# Iterative reconstruction using conjugate gradient (CG)
def iterative_recon(kspace_data: torch.Tensor, nufft_op: NUFFTOperator, num_iters=10):
    device = nufft_op.device
    image_shape = nufft_op.image_shape

    # Initial guess (zero image)
    x = torch.zeros(image_shape, dtype=torch.complex64, device=device)

    # Helper lambda for forward NUFFT
    A = lambda img: nufft_op.op(img)

    # Helper lambda for adjoint NUFFT
    At = lambda data: nufft_op.op_adj(data)

    # Precompute A^H y
    # kspace_data is assumed to be appropriately shaped for nufft_op.op_adj
    # For current NUFFT2D/3D, it expects 1D (num_k_points,)
    if kspace_data.ndim > 1 : # Basic check, might need adjustment based on op_adj needs
        if kspace_data.shape[0] == 1 or kspace_data.shape[1] ==1: # (1, N) or (N,1)
             kspace_data = kspace_data.reshape(-1)
        else:
            # This case might be an error depending on how op_adj is used in more complex scenarios
            # For now, assuming if it's not 1D, it's an error for current simple recon.
            raise ValueError(f"iterative_recon expects 1D kspace_data for At, got shape {kspace_data.shape}")


    AHy = At(kspace_data)

    r = AHy.clone()  # residual
    p = r.clone()
    rsold = torch.sum(torch.conj(r) * r).real

    for i in range(num_iters):
        Ap = At(A(p))
        alpha_num = rsold
        alpha_den = torch.sum(torch.conj(p) * Ap).real
        if torch.abs(alpha_den) < 1e-12: # Avoid division by zero or very small denominator
            print(f"Iter {i+1}, Denominator too small, stopping.")
            break
        alpha = alpha_num / alpha_den
        
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(torch.conj(r) * r).real
        if torch.sqrt(rsnew) < 1e-6:
            print(f"Iter {i+1}, Residual below tolerance, stopping.")
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        print(f"Iter {i+1}, Residual: {torch.sqrt(rsnew):.6e}")

    return x

# -------- Example usage --------

def _simple_phantom_2d_pattern(size=128, device='cpu'): # Renamed
    Y, X = torch.meshgrid(torch.linspace(-1, 1, size, device=device),
                          torch.linspace(-1, 1, size, device=device), indexing='ij')
    phantom = torch.zeros_like(X)
    phantom += 1.0 * (((X)**2 + (Y/1.5)**2) <= 0.9**2).float()
    phantom -= 0.8 * (((X+0.3)**2 + (Y/1.5)**2) <= 0.4**2).float()
    phantom += 0.5 * (((X-0.2)**2 + (Y-0.2)**2) <= 0.2**2).float()
    return phantom

def generate_phantom_2d(size=128, device='cpu'): # Renamed wrapper
    phantom_resized = _simple_phantom_2d_pattern(size, device=device)
    return torch.tensor(phantom_resized, dtype=torch.float32, device=device)

def generate_phantom_3d(shape=(64,64,64), device='cpu'):
    Nz, Ny, Nx = shape
    Z, Y, X = torch.meshgrid(torch.linspace(-1, 1, Nz, device=device),
                             torch.linspace(-1, 1, Ny, device=device),
                             torch.linspace(-1, 1, Nx, device=device), indexing='ij')
    phantom = torch.zeros_like(X)
    # Large sphere
    phantom += 1.0 * ((X**2 + Y**2 + Z**2) <= 0.8**2).float()
    # Smaller sphere (subtraction)
    phantom -= 0.5 * (((X-0.3)**2 + (Y-0.3)**2 + (Z-0.3)**2) <= 0.3**2).float()
    return phantom


def generate_radial_trajectory_2d(num_spokes=64, samples_per_spoke=256, device='cpu'):
    angles = torch.linspace(0, np.pi, num_spokes, device=device)
    radii = torch.linspace(-0.5, 0.5, samples_per_spoke, device=device)
    radii_grid, angles_grid = torch.meshgrid(radii, angles, indexing='ij')
    kx = (radii_grid * torch.cos(angles_grid)).reshape(-1)
    ky = (radii_grid * torch.sin(angles_grid)).reshape(-1)
    return torch.stack((kx, ky), dim=-1)


def generate_radial_trajectory_3d(num_profiles_z=32, num_spokes_per_profile=32, samples_per_spoke=64, shape=(64,64,64), device='cpu'):
    """Generates a 3D stack-of-stars trajectory."""
    Nz, Ny, Nx = shape # Unused for now, but good for context
    
    k_stack = []
    # z locations for stack-of-stars
    kz_positions = torch.linspace(-0.5, 0.5, num_profiles_z, device=device)

    for kz_val in kz_positions:
        angles = torch.linspace(0, np.pi, num_spokes_per_profile, endpoint=False, device=device)
        radii = torch.linspace(-0.5, 0.5, samples_per_spoke, device=device)
        
        radii_grid, angles_grid = torch.meshgrid(radii, angles, indexing='ij')
        
        kx_slice = (radii_grid * torch.cos(angles_grid)).reshape(-1)
        ky_slice = (radii_grid * torch.sin(angles_grid)).reshape(-1)
        kz_slice = torch.full_like(kx_slice, kz_val)
        
        k_stack.append(torch.stack((kx_slice, ky_slice, kz_slice), dim=-1))
        
    return torch.cat(k_stack, dim=0)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 2D Example ---
    print("\n--- Running 2D NUFFT Example ---")
    Nx_2d, Ny_2d = 128, 128
    image_shape_2d = (Nx_2d, Ny_2d)

    phantom_2d = generate_phantom_2d(Nx_2d, device=device)
    phantom_2d_complex = phantom_2d.to(torch.complex64)

    num_spokes_2d = 128
    samples_per_spoke_2d = 256
    k_traj_2d = generate_radial_trajectory_2d(num_spokes_2d, samples_per_spoke_2d, device=device)

    # Instantiate NUFFTOperator for 2D
    oversamp_factor_2d = (2.0, 2.0)
    kb_J_2d = (4, 4) 
    kb_alpha_2d = (2.34 * kb_J_2d[0], 2.34 * kb_J_2d[1])
    Ld_2d = (2**10, 2**10)
    Kd_2d = (int(Nx_2d * oversamp_factor_2d[0]), int(Ny_2d * oversamp_factor_2d[1]))
    kb_m_2d = (0.0, 0.0)
    n_shift_2d = (0.0, 0.0)

    nufft_op_2d = NUFFTOperator(k_trajectory=k_traj_2d, 
                                image_shape=image_shape_2d, 
                                oversamp_factor=oversamp_factor_2d, 
                                kb_J=kb_J_2d, 
                                kb_alpha=kb_alpha_2d, 
                                Ld=Ld_2d, 
                                kb_m=kb_m_2d, 
                                Kd=Kd_2d, 
                                n_shift=n_shift_2d, 
                                device=device)

    print("Simulating 2D k-space data...")
    kspace_data_2d = nufft_op_2d.op(phantom_2d_complex)
    # Add some noise
    noise_level = 0.01 * torch.mean(torch.abs(kspace_data_2d)) * (torch.randn_like(kspace_data_2d.real) + 1j * torch.randn_like(kspace_data_2d.real))
    kspace_data_2d += noise_level


    print("Running 2D iterative reconstruction...")
    recon_img_2d = iterative_recon(kspace_data=kspace_data_2d, 
                                   nufft_op=nufft_op_2d, 
                                   num_iters=10)

    fig_2d, axs_2d = plt.subplots(1, 2, figsize=(10, 5))
    axs_2d[0].imshow(phantom_2d.cpu().numpy(), cmap='gray')
    axs_2d[0].set_title("Original 2D Phantom")
    axs_2d[0].axis('off')
    axs_2d[1].imshow(recon_img_2d.abs().cpu().numpy(), cmap='gray')
    axs_2d[1].set_title("Reconstructed 2D Image")
    axs_2d[1].axis('off')
    plt.tight_layout()
    plt.show()

    # --- 3D Example ---
    print("\n--- Running 3D NUFFT Example ---")
    Nz_3d, Ny_3d, Nx_3d = 32, 32, 32 # Smaller for quicker test
    image_shape_3d = (Nz_3d, Ny_3d, Nx_3d)

    phantom_3d = generate_phantom_3d(shape=image_shape_3d, device=device)
    phantom_3d_complex = phantom_3d.to(torch.complex64)

    num_profiles_z_3d = 32
    num_spokes_per_profile_3d = 32
    samples_per_spoke_3d = Nx_3d # Match one dimension for simplicity
    k_traj_3d = generate_radial_trajectory_3d(num_profiles_z_3d, num_spokes_per_profile_3d, samples_per_spoke_3d, shape=image_shape_3d, device=device)

    # Instantiate NUFFTOperator for 3D (table-based)
    oversamp_factor_3d = (1.5, 1.5, 1.5) # Reduced oversampling for speed
    kb_J_3d = (4, 4, 4)
    kb_alpha_3d = (2.34 * kb_J_3d[0], 2.34 * kb_J_3d[1], 2.34 * kb_J_3d[2])
    Ld_3d = (2**8, 2**8, 2**8) # Reduced table oversampling
    Kd_3d = (int(Nz_3d * oversamp_factor_3d[0]), int(Ny_3d * oversamp_factor_3d[1]), int(Nx_3d * oversamp_factor_3d[2]))
    kb_m_3d = (0.0, 0.0, 0.0)
    n_shift_3d = (0.0, 0.0, 0.0)

    nufft_op_3d = NUFFTOperator(k_trajectory=k_traj_3d,
                                image_shape=image_shape_3d,
                                oversamp_factor=oversamp_factor_3d,
                                kb_J=kb_J_3d,
                                kb_alpha=kb_alpha_3d,
                                Ld=Ld_3d,
                                kb_m=kb_m_3d,
                                Kd=Kd_3d,
                                n_shift=n_shift_3d,
                                device=device,
                                nufft_type_3d='table')
    
    print("Simulating 3D k-space data...")
    kspace_data_3d = nufft_op_3d.op(phantom_3d_complex)
    # Add some noise
    noise_level_3d = 0.01 * torch.mean(torch.abs(kspace_data_3d)) * (torch.randn_like(kspace_data_3d.real) + 1j * torch.randn_like(kspace_data_3d.real))
    kspace_data_3d += noise_level_3d


    print("Running 3D iterative reconstruction...")
    recon_img_3d = iterative_recon(kspace_data=kspace_data_3d,
                                   nufft_op=nufft_op_3d,
                                   num_iters=5) # Reduced iterations for speed

    fig_3d, axs_3d = plt.subplots(1, 2, figsize=(10, 5))
    center_slice_z = Nz_3d // 2
    axs_3d[0].imshow(phantom_3d[center_slice_z, :, :].cpu().numpy(), cmap='gray')
    axs_3d[0].set_title(f"Original 3D Phantom (Slice {center_slice_z})")
    axs_3d[0].axis('off')
    
    axs_3d[1].imshow(recon_img_3d[center_slice_z, :, :].abs().cpu().numpy(), cmap='gray')
    axs_3d[1].set_title(f"Reconstructed 3D Image (Slice {center_slice_z})")
    axs_3d[1].axis('off')
    plt.tight_layout()
    plt.show()
