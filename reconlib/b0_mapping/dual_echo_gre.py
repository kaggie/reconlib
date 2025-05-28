# reconlib/b0_mapping/dual_echo_gre.py
"""Dual-echo (and multi-echo) Gradient Echo B0 mapping using PyTorch."""

import torch
import numpy as np # Retained for np.pi if torch.pi is not available
from typing import Callable

def calculate_b0_map_dual_echo(
    phase_images: torch.Tensor,
    echo_times: torch.Tensor,
    mask: torch.Tensor = None,
    unwrap_method_fn: Callable[[torch.Tensor], torch.Tensor] = None
) -> torch.Tensor:
    """
    Calculates B0 map using the phase difference method for two echoes using PyTorch.

    The B0 map is calculated as: B0_map = phase_difference / (2 * pi * delta_TE).
    The `phase_difference` (phase_echo2 - phase_echo1) can be optionally unwrapped
    using a user-provided function via `unwrap_method_fn`.

    Args:
        phase_images (torch.Tensor): PyTorch tensor of phase images.
                                     Shape: (num_echoes, ...), where `...` can be (D, H, W) or (H, W).
                                     `num_echoes` must be at least 2. Phase values are expected in radians.
        echo_times (torch.Tensor): PyTorch tensor of echo times in seconds.
                                   Shape: (num_echoes,).
        mask (torch.Tensor, optional): Boolean PyTorch tensor of the same spatial dimensions
                                       as a single phase image (e.g., (D, H, W) or (H, W)).
                                       Voxels where mask is False are set to 0 in the output B0 map.
                                       Defaults to None (no masking).
        unwrap_method_fn (typing.Callable, optional): A function to unwrap the calculated
                                                      phase difference map (phase_echo2 - phase_echo1).
                                                      The function should accept a PyTorch tensor
                                                      (the phase difference map) and return an unwrapped
                                                      PyTorch tensor of the same shape.
                                                      Example: `from reconlib.phase_unwrapping import unwrap_phase_3d_quality_guided`.
                                                      If None (default), the raw (potentially wrapped)
                                                      phase difference is used, which may lead to B0 aliasing
                                                      if the true phase difference exceeds pi radians.
                                                      Defaults to None.

    Returns:
        torch.Tensor: Calculated B0 map in Hz, with the same spatial dimensions as a single input phase image
                      and on the same device.
    """
    if not isinstance(phase_images, torch.Tensor):
        raise TypeError("phase_images must be a PyTorch tensor.")
    if not isinstance(echo_times, torch.Tensor):
        raise TypeError("echo_times must be a PyTorch tensor.")
    if mask is not None and not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a PyTorch tensor if provided.")
    if unwrap_method_fn is not None and not callable(unwrap_method_fn):
        raise TypeError("unwrap_method_fn must be a callable function or None.")

    device = phase_images.device
    echo_times = echo_times.to(device) # Ensure echo_times is on the same device

    if phase_images.shape[0] < 2:
        raise ValueError("At least two echo images are required.")
    if echo_times.shape[0] < 2:
        raise ValueError("At least two echo times are required.")
    if echo_times.shape[0] != phase_images.shape[0]:
        raise ValueError("Number of echo times must match the number of phase images.")

    delta_te = echo_times[1].item() - echo_times[0].item()
    if delta_te == 0:
        raise ValueError("Echo times for the first two echoes must be different.")

    pi_val = getattr(torch, 'pi', np.pi) # Keep for pi constant

    # Calculate raw phase difference (phase_TE2 - phase_TE1)
    phase_diff = phase_images[1, ...] - phase_images[0, ...]

    # Conditionally unwrap the phase difference
    if unwrap_method_fn is not None:
        # Assuming unwrap_method_fn handles the input shape as is (e.g. D,H,W or H,W)
        phase_to_use_for_b0 = unwrap_method_fn(phase_diff)
    else:
        # If no unwrapping function is provided, use the raw (potentially wrapped) phase difference.
        # Note: This might lead to B0 aliasing if abs(true phase difference) > pi.
        # For direct use without unwrapping, the phase difference should be within [-pi, pi).
        # If it's known to be outside this range and no unwrap_method_fn is given,
        # the user might expect this function to handle it.
        # However, the original logic was a simple re-wrap.
        # The new logic is: use unwrapper if given, else use raw difference.
        # If the raw difference is used, it should be the true difference,
        # which might exceed [-pi, pi]. If it needs to be in [-pi, pi] for some reason
        # without a full unwrap, that would be a simple wrap:
        # phase_to_use_for_b0 = (phase_diff + pi_val) % (2 * pi_val) - pi_val
        # But the instruction is to REMOVE the re-wrapping line and use phase_diff directly.
        phase_to_use_for_b0 = phase_diff

    b0_map = phase_to_use_for_b0 / (2 * pi_val * delta_te) # B0 in Hz

    if mask is not None:
        # Ensure mask has compatible dimensions
        if mask.shape != phase_images.shape[1:]:
             raise ValueError("Mask dimensions must match spatial dimensions of phase images.")
        if mask.device != b0_map.device:
            mask = mask.to(b0_map.device)
        b0_map[~mask] = 0

    return b0_map

def calculate_b0_map_multi_echo_linear_fit(
    phase_images: torch.Tensor, 
    echo_times: torch.Tensor, 
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Calculates B0 map by linear fitting of phase vs. echo times for multiple echoes using PyTorch.

    This function performs a voxel-wise linear regression of phase = slope * TE + intercept.
    The B0 map is then calculated as: B0 = slope / (2 * pi).
    The implementation uses `torch.linalg.lstsq` for efficient vectorized computation.
    It is recommended that input `phase_images` are unwrapped for accurate fitting,
    though the function will proceed with any input phase data.

    Args:
        phase_images (torch.Tensor): PyTorch tensor of phase images.
                                     Shape: (num_echoes, D, H, W) or (num_echoes, H, W).
                                     Phase values are expected in radians.
        echo_times (torch.Tensor): PyTorch tensor of echo times in seconds.
                                   Shape: (num_echoes,).
        mask (torch.Tensor, optional): Boolean PyTorch tensor for ROI.
                                       Shape should match spatial dimensions of `phase_images`.
                                       Voxels where mask is False are set to 0 in the output B0 map.
                                       Defaults to None.

    Returns:
        torch.Tensor: Calculated B0 map in Hz, with the same spatial dimensions as input phase images
                      and on the same device. If `torch.linalg.lstsq` fails, a zero map is returned
                      with a printed warning.
    """
    if not isinstance(phase_images, torch.Tensor):
        raise TypeError("phase_images must be a PyTorch tensor.")
    if not isinstance(echo_times, torch.Tensor):
        raise TypeError("echo_times must be a PyTorch tensor.")
    if mask is not None and not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a PyTorch tensor if provided.")

    device = phase_images.device
    dtype = phase_images.dtype
    echo_times = echo_times.to(device=device, dtype=dtype) # Ensure echo_times is on the same device and dtype

    num_echoes = phase_images.shape[0]
    if num_echoes < 2:
        raise ValueError("At least two echo images are required for linear fitting.")
    if echo_times.shape[0] != num_echoes:
        raise ValueError("Number of echo times must match the number of phase images.")

    spatial_dims_shape = phase_images.shape[1:]
    num_spatial_locations = int(torch.prod(torch.tensor(spatial_dims_shape)).item())

    # Reshape phase_images for vectorized processing: (num_echoes, N_spatial_locations)
    phase_data_reshaped = phase_images.reshape(num_echoes, -1) # (num_echoes, N)
    
    # Prepare design matrix A for linear regression (y = A * [slope; intercept])
    # A = [TEs, 1s]
    A = torch.stack([echo_times, torch.ones_like(echo_times)], dim=1) # Shape: (num_echoes, 2)
    
    # Transpose phase_data for lstsq: (N_spatial_locations, num_echoes)
    y = phase_data_reshaped.T # Shape: (N, num_echoes)
    
    # Solve Ax = y for x = [slope, intercept].T using torch.linalg.lstsq
    # lstsq expects b to be (N, num_echoes, k) if A is (num_echoes, 2)
    # Here, y is (N, num_echoes), so we need y.unsqueeze(-1) to make it (N, num_echoes, 1)
    # The design matrix A is (num_echoes, 2)
    try:
        solution = torch.linalg.lstsq(A, y.unsqueeze(-1))
        # solution.solution is (N, 2, 1) where N is num_spatial_locations
        slopes = solution.solution[:, 0, 0] # Slopes for each spatial location
    except torch.linalg.LinAlgError as e:
        # Fallback or error handling if lstsq fails globally (e.g., due to A matrix issues)
        # This is a global failure, so pixel-wise might not help unless the issue is specific to data.
        print(f"torch.linalg.lstsq failed: {e}. Returning zero map.")
        return torch.zeros(spatial_dims_shape, device=device, dtype=dtype)
        
    # B0 = slope / (2 * pi)
    pi_val = getattr(torch, 'pi', np.pi)
    b0_map_flat = slopes / (2 * pi_val)
    
    # Reshape b0_map back to original spatial dimensions
    b0_map = b0_map_flat.reshape(spatial_dims_shape)

    if mask is not None:
        if mask.shape != spatial_dims_shape:
             raise ValueError("Mask dimensions must match spatial dimensions of phase images.")
        if mask.device != b0_map.device:
            mask = mask.to(b0_map.device)
        b0_map[~mask] = 0
            
    return b0_map
