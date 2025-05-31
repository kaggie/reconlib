"""Module for utility functions."""

import numpy as np
import torch
from .data import MRIData # For type hinting and usage in get_echo_data
from reconlib.voronoi.density_weights_pytorch import compute_voronoi_density_weights_pytorch


def extract_phase(complex_data: np.ndarray) -> np.ndarray:
    """
    Extracts the phase from complex-valued data.

    Args:
        complex_data (np.ndarray): Input complex data.

    Returns:
        np.ndarray: Phase of the complex data in radians.
    """
    return np.angle(complex_data)


def extract_magnitude(complex_data: np.ndarray) -> np.ndarray:
    """
    Extracts the magnitude from complex-valued data.

    Args:
        complex_data (np.ndarray): Input complex data.

    Returns:
        np.ndarray: Magnitude of the complex data.
    """
    return np.abs(complex_data)


def get_echo_data(mri_data: MRIData, echo_index: int) -> np.ndarray:
    """
    Extracts k-space data for a specific echo from an MRIData object.

    Assumes that if multi-echo data is present, the first dimension of 
    mri_data.k_space_data is the echo dimension.

    Args:
        mri_data (MRIData): The MRIData object containing k-space data.
        echo_index (int): The index of the echo to extract.

    Returns:
        np.ndarray: The k-space data for the specified echo.

    Raises:
        ValueError: If the echo_index is out of bounds or if the data
                    does not appear to be multi-echo.
    """
    k_data = mri_data.k_space_data
    
    # Check if data appears to be multi-echo. 
    # A simple check is if it has more dimensions than typical for single-echo (e.g., > 3 for coils, kx, ky)
    # Or more robustly, by checking the length of the first dimension.
    # For this implementation, we'll assume if echo_times is present and has length > 1,
    # then the first dimension of k_space_data is echoes.
    # Or, if k_space_data has at least 2 dimensions (echoes, coils/data_points).
    
    is_multi_echo_shape = k_data.ndim > 2 # (echoes, coils, samples) or (echoes, samples_x, samples_y)
    
    if not is_multi_echo_shape and echo_index == 0:
        # If it doesn't look like multi-echo data but echo_index is 0, assume it's single echo.
        # Or, if k_data.shape[0] is not consistent with num_echoes if mri_data.echo_times exists.
        # For simplicity, let's be strict: if echo_index > 0, it must look multi-echo.
        pass # Allow if echo_index is 0 and not clearly multi-echo
    elif not is_multi_echo_shape and echo_index > 0:
         raise ValueError(
            f"k_space_data does not appear to be multi-echo (shape {k_data.shape}), "
            f"but echo_index {echo_index} was requested."
        )

    if echo_index < 0:
        raise ValueError("echo_index must be a non-negative integer.")

    if k_data.shape[0] <= echo_index:
        raise ValueError(
            f"echo_index {echo_index} is out of bounds for k_space_data with "
            f"{k_data.shape[0]} echoes (first dimension)."
        )
        
    return k_data[echo_index, ...]


def calculate_density_compensation(k_trajectory, image_shape, method='radial_simple', device='cpu', **kwargs):
    """
    Calculates density compensation weights for k-space trajectories.

    Args:
        k_trajectory (torch.Tensor): The k-space trajectory, normalized to [-0.5, 0.5].
                                     Shape: (num_samples, Ndims).
        image_shape (tuple): Shape of the image (e.g., (H, W) or (D, H, W)).
                             Not directly used by 'radial_simple' but kept for API consistency.
        method (str, optional): The method to use for calculation.
                                Defaults to 'radial_simple'.
                                Supported: 'radial_simple', 'voronoi' (placeholder).
        device (str, optional): PyTorch device ('cpu' or 'cuda'). Defaults to 'cpu'.
        **kwargs: Additional keyword arguments for specific methods.

    Returns:
        torch.Tensor: Density compensation weights. Shape: (num_samples,).
    """
    k_trajectory = torch.as_tensor(k_trajectory, device=device, dtype=torch.float32)

    if method == 'radial_simple':
        # Calculate k-space radius for each sample
        # k_trajectory is (num_samples, Ndims)
        radius = torch.sqrt(torch.sum(k_trajectory**2, dim=1))
        
        # Add epsilon to avoid issues at k-space center if radius is zero
        dcf = radius + 1e-6 
        
        # Normalize dcf
        max_val = torch.max(dcf)
        if max_val > 1e-9: # Avoid division by zero or very small numbers if all radii are near zero
            dcf = dcf / max_val
        else: # If max_val is very small, implies all radii are near zero (e.g. single point at center)
            dcf = torch.ones_like(dcf) # Uniform weighting
            
        return dcf.to(device)
        
    elif method == 'voronoi':
        # New logic:
        # Extract 'bounds' from kwargs if present
        bounds = kwargs.get('bounds', None) 
        if bounds is not None:
            # Ensure bounds is a tensor on the same device and dtype as k_trajectory for consistency,
            # although compute_voronoi_density_weights will handle CPU conversion for SciPy.
            bounds = torch.as_tensor(bounds, device=k_trajectory.device, dtype=k_trajectory.dtype)

        # compute_voronoi_density_weights_pytorch handles device of k_trajectory (points)
        # and returns weights on the original device of its input 'points'.
        # k_trajectory is already on the 'device' specified at the start of calculate_density_compensation.
        return compute_voronoi_density_weights_pytorch(points=k_trajectory, bounds=bounds)
        
    else:
        raise NotImplementedError(f"Density compensation method '{method}' is not implemented.")


# --- Start of new function: combine_coils_complex_sum ---
import typing # For Optional

def combine_coils_complex_sum(
    multi_coil_complex_data_one_echo: torch.Tensor, 
    mask: typing.Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Combines multi-coil complex data for a single echo using a complex sum.

    Args:
        multi_coil_complex_data_one_echo (torch.Tensor): PyTorch tensor of shape 
                                                         (num_coils, D, H, W) or (num_coils, H, W) 
                                                         containing complex data for one echo.
        mask (typing.Optional[torch.Tensor], optional): Boolean PyTorch tensor of spatial shape 
                                                        (D, H, W) or (H, W). If provided, 
                                                        calculations are performed only within the mask 
                                                        (e.g., complex sum is zeroed outside mask 
                                                        before taking angle/abs). Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - combined_phase (torch.Tensor): PyTorch tensor of spatial shape (D, H, W) or (H, W), 
                                             with phase values in [-pi, pi).
            - combined_magnitude (torch.Tensor): PyTorch tensor of spatial shape (D, H, W) or (H, W).
    """
    # Input Validation
    if not isinstance(multi_coil_complex_data_one_echo, torch.Tensor):
        raise TypeError("multi_coil_complex_data_one_echo must be a PyTorch tensor.")
    if not multi_coil_complex_data_one_echo.is_complex():
        raise ValueError("multi_coil_complex_data_one_echo must be a complex-valued tensor.")
    if not (multi_coil_complex_data_one_echo.ndim == 3 or multi_coil_complex_data_one_echo.ndim == 4):
        raise ValueError("multi_coil_complex_data_one_echo must have 3 (coils, H, W) or 4 (coils, D, H, W) dimensions.")

    device = multi_coil_complex_data_one_echo.device
    spatial_shape = multi_coil_complex_data_one_echo.shape[1:]

    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            raise TypeError("mask must be a PyTorch tensor if provided.")
        if mask.dtype != torch.bool:
            raise TypeError("mask must be a boolean tensor.")
        if mask.shape != spatial_shape:
            raise ValueError(f"Mask shape {mask.shape} must match input data spatial shape {spatial_shape}.")
        if mask.device != device:
            mask = mask.to(device)

    # 1. Complex sum over the coil dimension
    combined_complex_sum = torch.sum(multi_coil_complex_data_one_echo, dim=0)

    # 2. Apply mask if provided
    if mask is not None:
        # Ensure mask is broadcastable for element-wise multiplication if needed,
        # but here direct indexing is fine.
        combined_complex_sum[~mask] = 0 + 0j # Zero out regions outside the mask

    # 3. Calculate magnitude
    combined_magnitude = torch.abs(combined_complex_sum)

    # 4. Calculate phase
    combined_phase = torch.angle(combined_complex_sum)
    
    # Ensure phase is zero where magnitude is zero (or very small) to avoid noisy phase values
    # This is often implicitly handled by torch.angle for 0+0j input, but can be made explicit.
    # For example, if magnitude is very close to zero, phase might be ill-defined.
    # A common practice is to set phase to 0 where magnitude is below a small threshold.
    # However, torch.angle(0+0j) already returns 0.0.
    # If magnitude is non-zero but very small, phase might be noisy.
    # The current implementation relies on torch.angle's behavior.

    return combined_phase, combined_magnitude
# --- End of new function: combine_coils_complex_sum ---
