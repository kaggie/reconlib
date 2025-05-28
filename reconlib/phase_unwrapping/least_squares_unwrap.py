# reconlib/phase_unwrapping/least_squares_unwrap.py
"""3D Least-Squares Phase Unwrapping using FFT-based Poisson Solver in PyTorch."""

import torch
import numpy as np

def _wrap_phase(phase: torch.Tensor) -> torch.Tensor:
    """Wraps phase values to the interval [-pi, pi) using PyTorch operations."""
    return (phase + np.pi) % (2 * np.pi) - np.pi

def _compute_wrapped_gradients_3d(phase_wrapped: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes wrapped phase gradients along z, y, x using torch.roll (periodic boundary).
    """
    if phase_wrapped.ndim != 3:
        raise ValueError(f"phase_wrapped must be a 3D tensor, got shape {phase_wrapped.shape}")

    gz = _wrap_phase(torch.roll(phase_wrapped, shifts=-1, dims=0) - phase_wrapped)
    gy = _wrap_phase(torch.roll(phase_wrapped, shifts=-1, dims=1) - phase_wrapped)
    gx = _wrap_phase(torch.roll(phase_wrapped, shifts=-1, dims=2) - phase_wrapped)
    return gz, gy, gx

def _compute_divergence_3d(gz: torch.Tensor, gy: torch.Tensor, gx: torch.Tensor) -> torch.Tensor:
    """
    Computes divergence of 3D vector field (gradients) using torch.roll (periodic boundary).
    """
    if not (gz.ndim == 3 and gy.ndim == 3 and gx.ndim == 3):
        raise ValueError("Input gradients must be 3D tensors.")
    if not (gz.shape == gy.shape == gx.shape):
        raise ValueError("Input gradient tensors must have the same shape.")

    # Divergence: d(gz)/dz + d(gy)/dy + d(gx)/dx
    # Using backward difference for divergence to be consistent with forward difference for gradient
    # (or vice-versa, as long as they are adjoints up to a sign)
    # For periodic, (gz[i] - gz[i-1])
    dz = gz - torch.roll(gz, shifts=1, dims=0)
    dy = gy - torch.roll(gy, shifts=1, dims=1)
    dx = gx - torch.roll(gx, shifts=1, dims=2)
    
    return dz + dy + dx

def _solve_poisson_fft_3d(rhs: torch.Tensor) -> torch.Tensor:
    """
    Solves the 3D Poisson equation (Laplacian(phi) = rhs) using FFTs.
    Internal helper function.
    """
    if rhs.ndim != 3:
        raise ValueError(f"rhs must be a 3D tensor, got shape {rhs.shape}")
    
    device = rhs.device
    D, H, W = rhs.shape

    # 1. FFT of RHS
    rhs_fft = torch.fft.fftn(rhs, dim=(-3, -2, -1))

    # 2. Laplacian Kernel in Fourier Domain
    # Create frequency coordinates (normalized to [0, 1))
    # These represent k/N, so values range from 0 up to (N-1)/N.
    # For cosine term, 2*pi*k/N.
    kz_freq = torch.fft.fftfreq(D, d=1.0, device=device).reshape(-1, 1, 1)
    ky_freq = torch.fft.fftfreq(H, d=1.0, device=device).reshape(1, -1, 1)
    kx_freq = torch.fft.fftfreq(W, d=1.0, device=device).reshape(1, 1, -1)
    
    # Laplacian operator in Fourier domain: L_fft = 2*(cos(2*pi*kz) - 1) + 2*(cos(2*pi*ky) - 1) + 2*(cos(2*pi*kx) - 1)
    # which simplifies to 2*(cos(2*pi*kz) + cos(2*pi*ky) + cos(2*pi*kx) - 3)
    laplacian_kernel_fft = 2.0 * (
        torch.cos(2 * np.pi * kz_freq) + 
        torch.cos(2 * np.pi * ky_freq) + 
        torch.cos(2 * np.pi * kx_freq) - 3.0
    )
    
    # 3. Solve for Phi_fft
    # Avoid division by zero at DC component (0,0,0 frequency)
    # The mean of the unwrapped phase is undefined; setting DC of phi_fft to zero is a common convention.
    # Store original DC component of rhs_fft if needed for other conventions, but typically not for phase.
    
    # Create a copy to modify for safe division
    laplacian_kernel_fft_safe_div = laplacian_kernel_fft.clone()
    laplacian_kernel_fft_safe_div[0, 0, 0] = 1.0 # Avoid division by zero, result for DC will be handled next.

    phi_fft = rhs_fft / laplacian_kernel_fft_safe_div
    
    # Explicitly set DC component of phi_fft to 0
    phi_fft[0, 0, 0] = 0.0 

    # 4. IFFT to get Phi
    phi_unwrapped = torch.fft.ifftn(phi_fft, dim=(-3, -2, -1)).real
    
    return phi_unwrapped

def unwrap_phase_3d_least_squares(wrapped_phase: torch.Tensor) -> torch.Tensor:
    """
    Performs 3D phase unwrapping using a least-squares algorithm with an FFT-based Poisson solver.

    This method assumes that the true, unwrapped phase field `phi` is related to the
    wrapped phase `psi` by `psi = wrap(phi)`. The goal is to find `phi`.

    The core idea is that the wrapped gradients of `phi` can be estimated from `psi`.
    Let `gx, gy, gz` be the wrapped differences (gradients) of `psi`.
    The divergence of this estimated gradient field, `rho = div(g)`, is computed.
    The Poisson equation, `Laplacian(phi_unwrapped) = rho`, is then solved for `phi_unwrapped`.
    This solution provides the unwrapped phase up to an additive constant.

    The FFT-based Poisson solver inherently assumes periodic boundary conditions.
    The solution is unique up to an additive constant (DC component), which is set to zero here.

    Args:
        wrapped_phase (torch.Tensor): 3D tensor of wrapped phase values (in radians).
                                      Shape (D, H, W). Must be a PyTorch tensor.
                                      Values should ideally be in the range [-pi, pi).

    Returns:
        torch.Tensor: 3D tensor of unwrapped phase values, on the same device as input.
    """
    if not isinstance(wrapped_phase, torch.Tensor):
        raise TypeError("wrapped_phase must be a PyTorch tensor.")
    if wrapped_phase.ndim != 3:
        raise ValueError(f"Input wrapped_phase must be a 3D tensor, got shape {wrapped_phase.shape}")
    if not torch.is_floating_point(wrapped_phase):
        # Ensure it's float for FFT and other operations
        wrapped_phase = wrapped_phase.float() 

    # 1. Compute wrapped phase gradients
    gz, gy, gx = _compute_wrapped_gradients_3d(wrapped_phase)

    # 2. Compute divergence of the wrapped gradients (this is the RHS of Poisson equation)
    rhs = _compute_divergence_3d(gz, gy, gx)

    # 3. Solve Poisson equation using FFTs
    unwrapped_phase = _solve_poisson_fft_3d(rhs)
    
    return unwrapped_phase
