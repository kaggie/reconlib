"""Module for utility functions."""

import numpy as np
import torch

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
        print("Warning: Voronoi DCF (method='voronoi') not yet implemented. "
              "Returning uniform weights (ones).")
        return torch.ones(k_trajectory.shape[0], device=device, dtype=torch.float32)
        
    else:
        raise NotImplementedError(f"Density compensation method '{method}' is not implemented.")
