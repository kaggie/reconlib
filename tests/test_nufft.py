import torch
import numpy as np
import math # For math.pi if used in kb_alpha calculation
# It's good practice to be able to run tests without full reconlib install,
# so adjust path if operators/nufft are not directly importable.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reconlib.operators import NUFFTOperator

# Helper function to create a 2D NUFFTOperator for testing
def create_nufft_op_2d_for_test(device='cpu'):
    Nx_2d, Ny_2d = 64, 64 # Smaller for faster tests
    image_shape_2d = (Nx_2d, Ny_2d)
    
    # Simple radial trajectory
    num_spokes = 32
    samples_per_spoke = Nx_2d
    angles = torch.linspace(0, np.pi, num_spokes, device=device, dtype=torch.float32)
    radii = torch.linspace(-0.5, 0.5, samples_per_spoke, device=device, dtype=torch.float32)
    kx = torch.cat([r * torch.cos(theta) for theta in angles])
    ky = torch.cat([r * torch.sin(theta) for theta in angles])
    k_traj_2d = torch.stack((kx, ky), dim=-1)

    oversamp_factor_2d = (2.0, 2.0)
    kb_J_2d = (4, 4)
    kb_alpha_2d = tuple(2.34 * J for J in kb_J_2d)
    Ld_2d = (2**8, 2**8) # Smaller table for faster test setup
    Kd_2d = tuple(int(N * os) for N, os in zip(image_shape_2d, oversamp_factor_2d))

    return NUFFTOperator(
        k_trajectory=k_traj_2d,
        image_shape=image_shape_2d,
        oversamp_factor=oversamp_factor_2d,
        kb_J=kb_J_2d,
        kb_alpha=kb_alpha_2d,
        Ld=Ld_2d,
        Kd=Kd_2d,
        device=device
    )

# Helper function to create a 3D NUFFTOperator for testing
def create_nufft_op_3d_for_test(device='cpu'):
    Nz_3d, Ny_3d, Nx_3d = 16, 16, 16 # Much smaller for 3D tests
    image_shape_3d = (Nz_3d, Ny_3d, Nx_3d)

    # Simple 3D radial trajectory (stack-of-stars)
    num_profiles_z = 8
    num_spokes_per_profile = 8
    samples_per_spoke = Nx_3d
    
    k_stack = []
    kz_positions = torch.linspace(-0.5, 0.5, num_profiles_z, device=device, dtype=torch.float32)
    for kz_val in kz_positions:
        angles = torch.linspace(0, np.pi, num_spokes_per_profile, endpoint=False, device=device, dtype=torch.float32)
        radii = torch.linspace(-0.5, 0.5, samples_per_spoke, device=device, dtype=torch.float32)
        radii_grid, angles_grid = torch.meshgrid(radii, angles, indexing='ij')
        kx_slice = (radii_grid * torch.cos(angles_grid)).reshape(-1)
        ky_slice = (radii_grid * torch.sin(angles_grid)).reshape(-1)
        kz_slice = torch.full_like(kx_slice, kz_val)
        k_stack.append(torch.stack((kx_slice, ky_slice, kz_slice), dim=-1))
    k_traj_3d = torch.cat(k_stack, dim=0)

    oversamp_factor_3d = (1.5, 1.5, 1.5)
    kb_J_3d = (4, 4, 4)
    kb_alpha_3d = tuple(2.34 * J for J in kb_J_3d)
    Ld_3d = (2**6, 2**6, 2**6) # Smaller table for tests
    Kd_3d = tuple(int(N * os) for N, os in zip(image_shape_3d, oversamp_factor_3d))
    n_shift_3d = (0.0, 0.0, 0.0)

    return NUFFTOperator(
        k_trajectory=k_traj_3d,
        image_shape=image_shape_3d,
        oversamp_factor=oversamp_factor_3d,
        kb_J=kb_J_3d,
        kb_alpha=kb_alpha_3d,
        Ld=Ld_3d,
        Kd=Kd_3d,
        n_shift=n_shift_3d,
        device=device,
        nufft_type_3d='table'
    )

def adjoint_test(nufft_op, device='cpu', dtype=torch.complex64, tol=1e-5):
    print(f"Running adjoint test for operator with image_shape {nufft_op.image_shape} on {device}")
    # Image domain vector
    x_shape = nufft_op.image_shape
    x = torch.randn(x_shape, dtype=dtype, device=device)
    
    # K-space domain vector
    y_shape = (nufft_op.k_trajectory.shape[0],)
    y = torch.randn(y_shape, dtype=dtype, device=device)
    
    # Compute Ax and A*y
    Ax = nufft_op.op(x)
    Aty = nufft_op.op_adj(y)
    
    # Compute dot products
    # <Ax, y>
    lhs = torch.sum(Ax * torch.conj(y))
    # <x, A*y>
    rhs = torch.sum(x * torch.conj(Aty))
    
    abs_diff = torch.abs(lhs - rhs).item()
    rel_diff = abs_diff / torch.abs(lhs).item() if torch.abs(lhs) > 1e-9 else 0.0
    
    print(f"  LHS (<Ax, y>): {lhs.item()}")
    print(f"  RHS (<x, A*y>): {rhs.item()}")
    print(f"  Absolute Difference: {abs_diff:.6e}")
    print(f"  Relative Difference: {rel_diff:.6e}")
    
    assert abs_diff < tol, f"Adjoint test failed: abs_diff={abs_diff:.6e} >= tol={tol}"
    # Relative diff can be an issue if lhs is very small
    if torch.abs(lhs) > 1e-4 : # Only check relative if lhs is not too small
        assert rel_diff < tol, f"Adjoint test failed: rel_diff={rel_diff:.6e} >= tol={tol}"
    print(f"  Adjoint test passed (tol={tol}).")


def test_adjointness_2d(device='cpu'):
    print("\n--- Testing 2D NUFFTOperator Adjointness ---")
    nufft_op_2d = create_nufft_op_2d_for_test(device=device)
    adjoint_test(nufft_op_2d, device=device)

def test_adjointness_3d(device='cpu'):
    print("\n--- Testing 3D NUFFTOperator Adjointness ---")
    nufft_op_3d = create_nufft_op_3d_for_test(device=device)
    adjoint_test(nufft_op_3d, device=device, tol=1e-4) # 3D might have slightly higher tolerance due to more ops

# Placeholder for accuracy tests
def test_accuracy_2d(device='cpu'):
    print("\n--- Placeholder: Accuracy Test for 2D NUFFTOperator ---")
    # TODO: Implement an accuracy test, e.g., NUFFT of a known phantom
    # (like a centered Gaussian) with a simple trajectory, and compare
    # against an analytical or reference FT.
    assert True # Placeholder

def test_accuracy_3d(device='cpu'):
    print("\n--- Placeholder: Accuracy Test for 3D NUFFTOperator ---")
    # TODO: Implement for 3D
    assert True # Placeholder

if __name__ == '__main__':
    test_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running NUFFT tests on device: {test_device}")
    
    test_adjointness_2d(device=test_device)
    test_accuracy_2d(device=test_device) # Will just pass
    
    test_adjointness_3d(device=test_device)
    test_accuracy_3d(device=test_device) # Will just pass
    
    print("\nAll tests completed.")
