# reconlib/phase_unwrapping/goldstein_unwrap.py
"""3D Goldstein-style Phase Unwrapping using FFT-based filtering in PyTorch."""

import torch
import numpy as np

def unwrap_phase_3d_goldstein(wrapped_phase: torch.Tensor, k_filter_strength: float = 1.0) -> torch.Tensor:
    """
    Performs 3D phase unwrapping using a simplified Goldstein-style algorithm,
    which relies on filtering in the k-space (Fourier domain).

    The method involves transforming the wrapped phase into complex phasors,
    performing an FFT, filtering the k-space representation, performing an IFFT,
    and then extracting the phase angle. The filter is based on the magnitude
    of the k-space components, raised to a power `k_filter_strength`.

    This is a simplified interpretation of Goldstein's ideas, focusing on spectral
    filtering rather than explicit branch cut placement or residue handling.
    The effectiveness of this method can depend on the nature of the phase
    and the choice of `k_filter_strength`.

    Args:
        wrapped_phase (torch.Tensor): 3D tensor of wrapped phase values (in radians).
                                      Shape (D, H, W). Must be a PyTorch tensor.
                                      Values should ideally be in the range [-pi, pi).
        k_filter_strength (float, optional): Controls the strength of the k-space filter.
                                             Typically >= 0.
                                             - If 0, the filter is all ones (no effective filtering of
                                               relative magnitudes in k-space, though FFT/IFFT still occurs).
                                             - Higher values mean stronger filtering, which tends to
                                               emphasize dominant (lower frequency, higher magnitude)
                                               spectral components more.
                                             Defaults to 1.0.

    Returns:
        torch.Tensor: 3D tensor of unwrapped phase values, on the same device as input.
    """
    if not isinstance(wrapped_phase, torch.Tensor):
        raise TypeError("wrapped_phase must be a PyTorch tensor.")
    if wrapped_phase.ndim != 3:
        raise ValueError(f"Input wrapped_phase must be a 3D tensor, got shape {wrapped_phase.shape}")
    if not torch.is_floating_point(wrapped_phase):
        # Ensure it's float for exp and other operations
        wrapped_phase = wrapped_phase.float()
    if k_filter_strength < 0:
        raise ValueError("k_filter_strength must be non-negative.")

    device = wrapped_phase.device

    # 1. Convert wrapped phase to complex phasors
    complex_phasors = torch.exp(1j * wrapped_phase)

    # 2. Forward FFT
    phasors_fft = torch.fft.fftn(complex_phasors, dim=(-3, -2, -1))

    # 3. K-space Filtering
    if k_filter_strength == 0:
        # No filtering needed if strength is 0
        filtered_phasors_fft = phasors_fft
    else:
        magnitude_fft = torch.abs(phasors_fft)
        
        # Normalize magnitude (optional, but good for consistent k_filter_strength behavior)
        # Max value could be 0 if input is all zeros, hence epsilon.
        max_mag = torch.max(magnitude_fft)
        if max_mag > 1e-9: # Avoid division by zero or issues with near-zero max
            magnitude_fft_norm = magnitude_fft / max_mag
        else: # Handle all-zero or near-zero input magnitude
            magnitude_fft_norm = torch.zeros_like(magnitude_fft) # or ones_like, but zeros makes sense if input is zero

        # Create the filter
        # filter_k = magnitude_fft_norm ** k_filter_strength
        filter_k = torch.pow(magnitude_fft_norm, k_filter_strength)


        # Apply the filter
        filtered_phasors_fft = phasors_fft * filter_k

    # 4. Inverse FFT
    filtered_phasors = torch.fft.ifftn(filtered_phasors_fft, dim=(-3, -2, -1))

    # 5. Extract Unwrapped Phase
    unwrapped_phase = torch.angle(filtered_phasors)
    
    return unwrapped_phase
