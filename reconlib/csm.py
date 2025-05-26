"""Module for Coil Sensitivity Map (CSM) estimation utilities."""

import torch
import torchvision.transforms.functional as TF
from reconlib.operators import NUFFTOperator # Assuming NUFFTOperator is in __init__ of operators

def estimate_csm_from_central_kspace(
    multi_coil_kspace_data,  # PyTorch tensor (num_coils, num_k_samples)
    k_trajectory,            # PyTorch tensor (num_k_samples, Ndims), normalized to [-0.5, 0.5]
    image_shape,             # tuple, e.g., (H, W) or (D, H, W)
    central_k_region_ratio=0.08, # Ratio of k-space center to use (e.g., 0.08 means +/- 4% from center)
    smoothing_sigma=5.0,       # Sigma for Gaussian blur; if 0 or None, no smoothing
    device='cpu'
):
    """
    Estimates coil sensitivity maps (CSMs) from the central region of multi-coil k-space data.

    Args:
        multi_coil_kspace_data (torch.Tensor): Multi-coil k-space data.
            Shape: (num_coils, num_k_samples).
        k_trajectory (torch.Tensor): K-space trajectory, coordinates normalized to [-0.5, 0.5].
            Shape: (num_k_samples, Ndims), where Ndims is 2 for 2D, 3 for 3D.
        image_shape (tuple): Desired image shape, e.g., (H, W) or (D, H, W).
        central_k_region_ratio (float, optional): Fraction of the k-space center to use.
            Defaults to 0.08 (i.e., +/- 4% of the FOV from k-space center).
        smoothing_sigma (float, optional): Standard deviation for Gaussian blur smoothing of
            low-resolution coil images. If 0 or None, no smoothing is applied. Defaults to 5.0.
            For 3D data, smoothing is currently skipped with a warning.
        device (str, optional): PyTorch device ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        torch.Tensor: Estimated coil sensitivity maps. Shape: (num_coils, *image_shape).
    """
    
    # Ensure inputs are PyTorch tensors and on the specified device
    multi_coil_kspace_data = torch.as_tensor(multi_coil_kspace_data, device=device, dtype=torch.complex64)
    k_trajectory = torch.as_tensor(k_trajectory, device=device, dtype=torch.float32)
    
    num_coils = multi_coil_kspace_data.shape[0]
    Ndims = k_trajectory.shape[1]

    if len(image_shape) != Ndims:
        raise ValueError(f"image_shape dimension {len(image_shape)} does not match "
                         f"k_trajectory dimension {Ndims}.")

    # Select Central K-space Points
    # Mask for k-space points where each coordinate k_d satisfies
    # -central_k_region_ratio/2 <= k_d <= central_k_region_ratio/2
    half_ratio = central_k_region_ratio / 2.0
    central_k_indices = torch.all(
        (k_trajectory >= -half_ratio) & (k_trajectory <= half_ratio), 
        dim=1
    )

    if not torch.any(central_k_indices):
        raise ValueError(
            f"No k-space points found in the central region defined by ratio {central_k_region_ratio}. "
            "Consider increasing the ratio or checking k_trajectory normalization."
        )
            
    central_k_traj = k_trajectory[central_k_indices]
    central_k_space_data_mc = multi_coil_kspace_data[:, central_k_indices]

    # Create NUFFT Operator for reconstructing low-resolution images from central k-space
    # This uses the selected central k-space points but reconstructs onto the full image_shape.
    # NUFFTOperator's internal 2D kwargs (oversamp, width, beta) will use defaults if not specified.
    low_res_nufft_op = NUFFTOperator(
        k_trajectory=central_k_traj, 
        image_shape=image_shape, 
        device=device
    )

    low_res_coil_images = torch.zeros((num_coils, *image_shape), dtype=torch.complex64, device=device)

    for c in range(num_coils):
        coil_k_space_central = central_k_space_data_mc[c, :]
        img_c = low_res_nufft_op.op_adj(coil_k_space_central) # Adjoint NUFFT

        if smoothing_sigma is not None and smoothing_sigma > 0:
            if Ndims == 2:
                # Kernel size should be odd and typically > 2*sigma.
                # Example: kernel_radius = int(2*smoothing_sigma), kernel_size = 2*kernel_radius + 1
                # The prompt used int(4*sigma+1)|1, which is also fine.
                # Let's use a common method: kernel_size = 2 * int(2*sigma) + 1, ensure odd.
                # Or simpler: int(round(3.5*sigma)) * 2 + 1
                # Let's use the int(4*sigma+1)|1 from prompt as it's specified.
                kernel_s = int(4 * smoothing_sigma + 1) | 1 # Ensure odd kernel size

                # Smooth real and imaginary parts separately
                # TF.gaussian_blur expects (..., H, W)
                img_c_real_smooth = TF.gaussian_blur(
                    img_c.real.unsqueeze(0), # Add channel dim: (1, H, W)
                    kernel_size=[kernel_s, kernel_s], 
                    sigma=smoothing_sigma
                ).squeeze(0) # Remove channel dim
                
                img_c_imag_smooth = TF.gaussian_blur(
                    img_c.imag.unsqueeze(0), # Add channel dim: (1, H, W)
                    kernel_size=[kernel_s, kernel_s],
                    sigma=smoothing_sigma
                ).squeeze(0) # Remove channel dim
                
                img_c = torch.complex(img_c_real_smooth, img_c_imag_smooth)
            elif Ndims == 3:
                print("Warning: estimate_csm_from_central_kspace - 3D smoothing with torchvision.transforms.functional "
                      "is not directly supported. Returning unsmoothed low-resolution images for CSM estimation.")
            # else: Ndims not 2 or 3, should have been caught earlier or by NUFFTOperator.
            
        low_res_coil_images[c, ...] = img_c

    # Normalize by Root-Sum-of-Squares (RSS)
    rss_image = torch.sqrt(torch.sum(torch.abs(low_res_coil_images)**2, dim=0)) + 1e-9 # Epsilon for stability
    csm_maps = low_res_coil_images / rss_image.unsqueeze(0) # Expand rss_image for broadcasting

    return csm_maps
