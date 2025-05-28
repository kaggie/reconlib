# reconlib/pipeline_utils.py
"""Higher-level utilities for orchestrating multi-step MRI data processing pipelines."""

import torch
import typing # For Optional, Callable

# Assuming these modules and functions are accessible in the reconlib package
from .utils import combine_coils_complex_sum 
from .phase_unwrapping.reference_echo_unwrap import unwrap_multi_echo_masked_reference
# Note: The user of `preprocess_multi_coil_multi_echo_data` will pass a specific
# `spatial_unwrap_fn` like `unwrap_phase_3d_quality_guided`.

def preprocess_multi_coil_multi_echo_data(
    multi_coil_multi_echo_complex_images: torch.Tensor, 
    snr_threshold_for_mask: float, 
    spatial_unwrap_fn: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Processes raw multi-coil, multi-echo complex image data to produce coil-combined, 
    spatially unwrapped phase images for each echo, along with the generated mask and
    combined magnitudes.

    The pipeline involves:
    1. Coil-combining each echo's multi-coil complex data using complex sum.
    2. Using the combined magnitudes and (initially wrapped) combined phases as input to
       `unwrap_multi_echo_masked_reference`, which internally:
       a. Generates a mask from the first echo's combined magnitude.
       b. Spatially unwraps the first combined echo phase using this mask.
       c. Unwraps subsequent echoes by spatially unwrapping the (wrapped) phase difference
          relative to the first echo and adding it to the unwrapped first echo phase.

    Args:
        multi_coil_multi_echo_complex_images (torch.Tensor): 
            PyTorch tensor of complex-valued images.
            Shape: (num_echoes, num_coils, D, H, W) or (num_echoes, num_coils, H, W).
        snr_threshold_for_mask (float): 
            Scalar threshold applied to the magnitude of the coil-combined first echo
            to generate a binary mask. Used by `unwrap_multi_echo_masked_reference`.
        spatial_unwrap_fn (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
            The spatial unwrapping function to be used internally by 
            `unwrap_multi_echo_masked_reference`. This function should accept two arguments:
            a phase tensor (e.g., (D,H,W) or (H,W)) and a boolean mask tensor of the
            same spatial shape, and return an unwrapped phase tensor.
            Example: `from reconlib.phase_unwrapping import unwrap_phase_3d_quality_guided`.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - final_unwrapped_phases (torch.Tensor): 
              Tensor of unwrapped phase for each echo, coil-combined.
              Shape: (num_echoes, D, H, W) or (num_echoes, H, W).
            - final_mask (torch.Tensor): 
              Boolean tensor mask generated from the first echo's combined magnitude.
              Shape: (D, H, W) or (H, W).
            - stacked_combined_magnitudes (torch.Tensor): 
              Tensor of coil-combined magnitude for each echo.
              Shape: (num_echoes, D, H, W) or (num_echoes, H, W).
    """
    # Input Validation (Basic)
    if not isinstance(multi_coil_multi_echo_complex_images, torch.Tensor):
        raise TypeError("multi_coil_multi_echo_complex_images must be a PyTorch tensor.")
    if not multi_coil_multi_echo_complex_images.is_complex():
        raise ValueError("multi_coil_multi_echo_complex_images must be a complex-valued tensor.")
    if not (multi_coil_multi_echo_complex_images.ndim == 4 or multi_coil_multi_echo_complex_images.ndim == 5):
        raise ValueError("multi_coil_multi_echo_complex_images must have 4 (echoes, coils, H, W) "
                         "or 5 (echoes, coils, D, H, W) dimensions.")
    if not callable(spatial_unwrap_fn):
        raise TypeError("spatial_unwrap_fn must be a callable function.")

    device = multi_coil_multi_echo_complex_images.device
    num_echoes = multi_coil_multi_echo_complex_images.shape[0]

    # Initialize lists to store per-echo combined data
    list_combined_phases = []
    list_combined_magnitudes = []

    # 1. Coil-combine each echo
    for e in range(num_echoes):
        current_multi_coil_echo_data = multi_coil_multi_echo_complex_images[e, ...]
        
        # Perform complex sum for the current echo. No mask is applied at this stage of coil combination.
        combined_phase_echo_e, combined_magnitude_echo_e = combine_coils_complex_sum(
            current_multi_coil_echo_data, 
            mask=None 
        )
        
        list_combined_phases.append(combined_phase_echo_e)
        list_combined_magnitudes.append(combined_magnitude_echo_e)

    # 2. Stack the combined phases and magnitudes
    # These are now (num_echoes, D, H, W) or (num_echoes, H, W)
    stacked_combined_phases = torch.stack(list_combined_phases, dim=0)
    stacked_combined_magnitudes = torch.stack(list_combined_magnitudes, dim=0)

    # 3. Perform multi-echo reference unwrapping
    # `unwrap_multi_echo_masked_reference` expects:
    #   magnitude_images: (num_echoes, ...) - used for mask generation from TE1
    #   wrapped_phase_images: (num_echoes, ...) - these are the coil-combined phases
    #   snr_threshold: float
    #   spatial_unwrap_fn: Callable that takes (phase_to_unwrap, mask_for_unwrapper)
    
    final_unwrapped_phases, final_mask = unwrap_multi_echo_masked_reference(
        magnitude_images=stacked_combined_magnitudes, # Used for mask generation from its 1st echo
        wrapped_phase_images=stacked_combined_phases, # These are the coil-combined phases to be unwrapped
        snr_threshold=snr_threshold_for_mask,
        spatial_unwrap_fn=spatial_unwrap_fn
    )

    return final_unwrapped_phases, final_mask, stacked_combined_magnitudes
