# reconlib/b0_mapping/phase_based_b0_field_maps.py
"""Phase-based B0 field map estimation methods using PyTorch."""

import torch
import numpy as np # Retained for np.pi if torch.pi is not available
import typing # For Optional

def calculate_b0_map_dual_echo(
    processed_phase_images: torch.Tensor,
    echo_times: torch.Tensor,
    mask: typing.Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Calculates B0 map using the phase difference method from two pre-processed echoes using PyTorch.

    This function assumes the input `processed_phase_images` are phase-only and have been
    appropriately pre-processed (e.g., coil-combined, spatially unwrapped for each echo).
    The B0 map is calculated as: B0_map = (phase_echo2 - phase_echo1) / (2 * pi * delta_TE).
    No further unwrapping of the phase difference is performed by this function.

    For preparing inputs from raw multi-coil data, consider using utilities like
    `reconlib.pipeline_utils.preprocess_multi_coil_multi_echo_data`.

    Args:
        processed_phase_images (torch.Tensor): PyTorch tensor of pre-processed phase images
                                               for two echoes.
                                               Shape: (2, ...), where `...` can be (D, H, W) or (H, W).
                                               Phase values are expected in radians and should be spatially unwrapped.
        echo_times (torch.Tensor): PyTorch tensor of echo times in seconds for the two echoes.
                                   Shape: (2,).
        mask (typing.Optional[torch.Tensor], optional): Boolean PyTorch tensor of the same spatial dimensions
                                                        as a single phase image (e.g., (D, H, W) or (H, W)).
                                                        Voxels where mask is False are set to 0 in the output B0 map.
                                                        Defaults to None (no masking).

    Returns:
        torch.Tensor: Calculated B0 map in Hz, with the same spatial dimensions as a single input phase image
                      and on the same device.
    """
    if not isinstance(processed_phase_images, torch.Tensor):
        raise TypeError("processed_phase_images must be a PyTorch tensor.")
    if not isinstance(echo_times, torch.Tensor):
        raise TypeError("echo_times must be a PyTorch tensor.")
    if mask is not None and not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a PyTorch tensor if provided.")
    if processed_phase_images.is_complex():
        raise ValueError("processed_phase_images should be real-valued (phase-only), not complex.")

    device = processed_phase_images.device
    echo_times = echo_times.to(device)

    if processed_phase_images.shape[0] != 2: # Strict check for dual-echo
        raise ValueError("processed_phase_images must contain exactly two echoes.")
    if echo_times.shape[0] != 2:
        raise ValueError("echo_times must contain exactly two echo times.")

    delta_te = echo_times[1].item() - echo_times[0].item()
    if delta_te == 0:
        raise ValueError("Echo times for the two echoes must be different.")

    pi_val = getattr(torch, 'pi', np.pi)

    # Calculate phase difference using the pre-processed (spatially unwrapped) phase images
    phase_diff = processed_phase_images[1, ...] - processed_phase_images[0, ...]
    # No further unwrapping of phase_diff is done here. User must ensure inputs are suitable.

    b0_map = phase_diff / (2 * pi_val * delta_te) # B0 in Hz

    if mask is not None:
        if mask.shape != processed_phase_images.shape[1:]:
             raise ValueError("Mask dimensions must match spatial dimensions of phase images.")
        if mask.device != b0_map.device:
            mask = mask.to(b0_map.device)
        b0_map[~mask] = 0

    return b0_map

def calculate_b0_map_multi_echo_linear_fit(
    processed_phase_images: torch.Tensor, 
    echo_times: torch.Tensor, 
    mask: typing.Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Calculates B0 map by linear fitting of phase vs. echo times for multiple pre-processed echoes using PyTorch.

    This function assumes the input `processed_phase_images` are phase-only and that each echo's
    phase image has been appropriately pre-processed (e.g., coil-combined and spatially unwrapped).
    It performs a voxel-wise linear regression of phase = slope * TE + intercept.
    The B0 map is then calculated as: B0 = slope / (2 * pi).
    The implementation uses `torch.linalg.lstsq` for efficient vectorized computation.

    For preparing inputs from raw multi-coil data, consider using utilities like
    `reconlib.pipeline_utils.preprocess_multi_coil_multi_echo_data`.

    Args:
        processed_phase_images (torch.Tensor): PyTorch tensor of pre-processed phase images.
                                               Shape: (num_echoes, D, H, W) or (num_echoes, H, W).
                                               Phase values are expected in radians and should be spatially unwrapped for each echo.
        echo_times (torch.Tensor): PyTorch tensor of echo times in seconds.
                                   Shape: (num_echoes,).
        mask (typing.Optional[torch.Tensor], optional): Boolean PyTorch tensor for ROI.
                                                        Shape should match spatial dimensions of a single phase image.
                                                        Voxels where mask is False are set to 0 in the output B0 map.
                                                        Defaults to None.

    Returns:
        torch.Tensor: Calculated B0 map in Hz, with the same spatial dimensions as a single input phase image
                      and on the same device. If `torch.linalg.lstsq` fails, a zero map is returned
                      with a printed warning.
    """
    if not isinstance(processed_phase_images, torch.Tensor):
        raise TypeError("processed_phase_images must be a PyTorch tensor.")
    if not isinstance(echo_times, torch.Tensor):
        raise TypeError("echo_times must be a PyTorch tensor.")
    if mask is not None and not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a PyTorch tensor if provided.")
    if processed_phase_images.is_complex():
        raise ValueError("processed_phase_images should be real-valued (phase-only), not complex.")

    device = processed_phase_images.device
    dtype = processed_phase_images.dtype # Use dtype of input phase images
    echo_times = echo_times.to(device=device, dtype=dtype) 

    num_echoes = processed_phase_images.shape[0]
    if num_echoes < 2:
        raise ValueError("At least two echo images are required for linear fitting.")
    if echo_times.shape[0] != num_echoes:
        raise ValueError("Number of echo times must match the number of phase images.")

    spatial_dims_shape = processed_phase_images.shape[1:]
    
    # Input `processed_phase_images` are used directly.
    # Reshape for vectorized processing: (num_echoes, N_spatial_locations)
    phase_data_reshaped = processed_phase_images.reshape(num_echoes, -1) # (num_echoes, N)
    
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
