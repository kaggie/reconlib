# reconlib/phase_unwrapping/reference_echo_unwrap.py
"""Multi-echo phase unwrapping using a reference echo and spatial unwrapping."""

import torch
import numpy as np # For np.pi fallback
import typing # For Callable

# It's assumed the user will pass a compatible spatial_unwrap_fn, 
# e.g., from reconlib.phase_unwrapping import unwrap_phase_3d_quality_guided

def _wrap_to_pi(phase_diff: torch.Tensor) -> torch.Tensor:
    """
    Wraps phase values to the interval [-pi, pi) using PyTorch operations.
    """
    pi = getattr(torch, 'pi', np.pi)
    return (phase_diff + pi) % (2 * pi) - pi

def unwrap_multi_echo_masked_reference(
    magnitude_images: torch.Tensor, 
    wrapped_phase_images: torch.Tensor, 
    snr_threshold: float, 
    spatial_unwrap_fn: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unwraps multi-echo, coil-combined phase images using a reference echo strategy with masking.

    This function assumes that the input `magnitude_images` and `wrapped_phase_images`
    have already been coil-combined (e.g., using `reconlib.utils.combine_coils_complex_sum`).
    For a full pipeline starting from raw multi-coil, multi-echo data, consider using
    `reconlib.pipeline_utils.preprocess_multi_coil_multi_echo_data`.

    The method involves:
    1.  Creating a binary mask from the magnitude of the coil-combined first echo 
        (`magnitude_images[0, ...]`) using `snr_threshold`.
    2.  Spatially unwrapping the coil-combined phase of the first echo 
        (`wrapped_phase_images[0, ...]`) using the provided `spatial_unwrap_fn` and the 
        generated mask. This becomes the reference unwrapped phase.
    3.  For each subsequent echo `e > 0`:
        a.  Calculate the phase difference: 
            `wrapped_phase_images[e, ...] - wrapped_phase_images[0, ...]`.
        b.  Wrap this difference to `[-pi, pi)` using `_wrap_to_pi`.
        c.  Spatially unwrap this wrapped difference using `spatial_unwrap_fn` and the 
            generated mask.
        d.  Add the unwrapped difference to the unwrapped first echo's phase to get the 
            unwrapped phase for echo `e`.

    Args:
        magnitude_images (torch.Tensor): Tensor of coil-combined magnitude images for each echo.
                                         Shape: (num_echoes, D, H, W) or (num_echoes, H, W).
        wrapped_phase_images (torch.Tensor): Tensor of coil-combined wrapped phase images for each echo.
                                             Shape: (num_echoes, D, H, W) or (num_echoes, H, W).
                                             Phase values are expected in radians.
        snr_threshold (float): Threshold to apply to the first echo's combined magnitude image
                               (`magnitude_images[0, ...]`) to generate a binary mask. 
                               Values above the threshold are considered True.
        spatial_unwrap_fn (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
                               A function for spatial phase unwrapping. It should accept two arguments:
                               1. A phase tensor (e.g., a single echo's coil-combined phase or a 
                                  phase difference map) with spatial dimensions (D, H, W) or (H, W).
                               2. A boolean mask tensor with the same spatial dimensions.
                               It should return an unwrapped phase tensor of the same shape.
                               Example: `from reconlib.phase_unwrapping import unwrap_phase_3d_quality_guided`.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - unwrapped_phases_all_echoes (torch.Tensor): Tensor containing the unwrapped phase
              for all echoes. Same shape as `wrapped_phase_images`.
            - generated_mask (torch.Tensor): Boolean tensor mask derived from the first echo's
              magnitude. Same spatial dimensions as a single echo image.
    """
    # Input Validation (Basic)
    if not isinstance(magnitude_images, torch.Tensor) or not isinstance(wrapped_phase_images, torch.Tensor):
        raise TypeError("magnitude_images and wrapped_phase_images must be PyTorch tensors.")
    if magnitude_images.shape != wrapped_phase_images.shape:
        raise ValueError("magnitude_images and wrapped_phase_images must have the same shape.")
    if wrapped_phase_images.ndim < 3: # e.g., (num_echoes, H, W) or (num_echoes, D, H, W)
        raise ValueError("Input images must have at least 3 dimensions (num_echoes, spatial_dims...).")
    if wrapped_phase_images.shape[0] < 2:
        raise ValueError("At least two echoes are required for multi-echo unwrapping.")
    if not callable(spatial_unwrap_fn):
        raise TypeError("spatial_unwrap_fn must be a callable function.")

    device = wrapped_phase_images.device
    
    # 1. Generate Mask from First Echo Magnitude
    first_echo_magnitude = magnitude_images[0, ...]
    # Ensure threshold is a tensor for comparison, or that magnitude is float for comparison with float threshold
    generated_mask = first_echo_magnitude > torch.tensor(snr_threshold, device=device, dtype=first_echo_magnitude.dtype)

    # 2. Spatially Unwrap First Echo
    # spatial_unwrap_fn expects (phase_volume, mask_volume)
    unwrapped_first_echo = spatial_unwrap_fn(wrapped_phase_images[0, ...], generated_mask)

    # 3. Initialize Output Array
    num_echoes = wrapped_phase_images.shape[0]
    unwrapped_phases_all_echoes = torch.zeros_like(wrapped_phase_images, device=device)
    unwrapped_phases_all_echoes[0, ...] = unwrapped_first_echo

    # 4. Unwrap Subsequent Echoes
    for e in range(1, num_echoes):
        current_wrapped_phase_echo = wrapped_phase_images[e, ...]
        reference_wrapped_phase_echo = wrapped_phase_images[0, ...] # Using the original wrapped first echo as reference for diff
        
        # Calculate wrapped difference: (phi_e - phi_ref_echo0)
        wrapped_diff = _wrap_to_pi(current_wrapped_phase_echo - reference_wrapped_phase_echo)
        
        # Spatially unwrap the difference map
        unwrapped_diff = spatial_unwrap_fn(wrapped_diff, generated_mask)
        
        # Combine with the unwrapped reference echo
        unwrapped_phases_all_echoes[e, ...] = unwrapped_first_echo + unwrapped_diff
            
    return unwrapped_phases_all_echoes, generated_mask
