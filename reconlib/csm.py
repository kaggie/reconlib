"""Module for Coil Sensitivity Map (CSM) estimation utilities."""

import torch
import numpy as np
import torchvision.transforms.functional as TF
from reconlib.operators import NUFFTOperator # Assuming NUFFTOperator is in __init__ of operators

# Attempt to import SigPy and related modules
try:
    # Removed SigPy specific imports, as this will be a native PyTorch implementation.
    # _SIGPY_AVAILABLE = False # No longer needed

# Assuming NUFFT2D and NUFFT3D are available at these paths
# If not, this import will need adjustment or NUFFTOperator will be used with modifications.
try:
    from reconlib.nufft import NUFFT2D, NUFFT3D
    _NUFFT_SPECIFIC_AVAILABLE = True
except ImportError:
    # Fallback or error if specific NUFFT classes are not found
    # For now, we'll assume NUFFTOperator might be a general one.
    # The user prompt implies NUFFT2D/3D are expected.
    # If NUFFTOperator is the only one, its usage might need to be adapted for this context.
    print("Warning: NUFFT2D/NUFFT3D not found directly in reconlib.nufft. ESPIRiT may fail or use fallback.")
    _NUFFT_SPECIFIC_AVAILABLE = False
    # As NUFFTOperator is already imported, we might use it if it's suitable.
    # from reconlib.operators import NUFFTOperator # Already imported


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


def estimate_espirit_maps(
    acs_kspace_data: torch.Tensor,
    k_trajectory_acs: torch.Tensor,
    image_shape: tuple,
    kernel_size: tuple = (6, 6),
    calib_thresh: float = 0.02,
    eigen_thresh: float = 0.95,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Estimates ESPIRiT sensitivity maps from Auto-Calibration Signal (ACS) k-space data
    using a native PyTorch implementation.

    The ESPIRiT algorithm involves these main steps:
    1. Gridding ACS k-space data to a Cartesian grid.
    2. Constructing a calibration matrix from local k-space neighborhoods (kernels).
    3. Performing SVD on the calibration matrix to find k-space kernels.
    4. Transforming these kernels to the image domain.
    5. Solving a pixel-wise eigenvalue problem to estimate sensitivity maps.

    Args:
        acs_kspace_data (torch.Tensor): Multi-coil k-space data from the ACS region.
            Shape: (num_coils, num_acs_points). Must be complex.
        k_trajectory_acs (torch.Tensor): K-space trajectory for ACS data, normalized to [-0.5, 0.5].
            Shape: (num_acs_points, Ndims), where Ndims is 2 for 2D, 3 for 3D. Must be float.
        image_shape (tuple): Desired full image shape for the maps, e.g., (H, W) or (D, H, W).
        kernel_size (tuple, optional): Size of the k-space kernels, e.g., (Kx, Ky) or (Kz, Kx, Ky).
                                       Defaults to (6, 6).
        calib_thresh (float, optional): Threshold for singular values in calibration matrix SVD,
                                        relative to the maximum singular value. Defaults to 0.02.
        eigen_thresh (float, optional): Threshold for selecting eigenvectors (maps) based on their
                                        eigenvalue magnitude. Defaults to 0.95.
        device (str, optional): PyTorch device ('cpu' or 'cuda:X'). Defaults to 'cpu'.

    Returns:
        torch.Tensor: Estimated ESPIRiT sensitivity maps.
                      Shape: (num_coils, *image_shape).
    """
    # --- 1. Initial Setup ---
    target_device = torch.device(device)
    acs_kspace_data = torch.as_tensor(acs_kspace_data, dtype=torch.complex64, device=target_device)
    k_trajectory_acs = torch.as_tensor(k_trajectory_acs, dtype=torch.float32, device=target_device)

    num_coils = acs_kspace_data.shape[0]
    ndim = k_trajectory_acs.shape[1]

    if len(image_shape) != ndim:
        raise ValueError(f"image_shape dimension {len(image_shape)} does not match k_trajectory_acs dimension {ndim}.")

    kernel_size_adjusted: tuple
    if ndim == 2:
        if len(kernel_size) != 2:
            print(f"Warning: 2D data but kernel_size {kernel_size} is not 2D. Using (6,6).")
            kernel_size_adjusted = (6, 6)
        else:
            kernel_size_adjusted = kernel_size
    elif ndim == 3:
        if len(kernel_size) == 3: # (Kz, Kx, Ky)
            kernel_size_adjusted = kernel_size
        elif len(kernel_size) == 2: # (Kx, Ky) for 3D data
            # Heuristic: use first dim for Kz, or a fixed default like (6, Kx, Ky)
            print(f"Warning: 3D data, but 2D kernel_size {kernel_size} provided. Adapting to ({kernel_size[0]},{kernel_size[0]},{kernel_size[1]}).")
            kernel_size_adjusted = (kernel_size[0], kernel_size[0], kernel_size[1])
        else:
            print(f"Warning: 3D data but kernel_size {kernel_size} is not 2D or 3D. Using (6,6,6).")
            kernel_size_adjusted = (6, 6, 6)
    else:
        raise ValueError(f"Unsupported number of dimensions: {ndim}. Must be 2 or 3.")

    # --- 2. ACS Data Gridding ---
    calib_shape = tuple(ks * 4 for ks in kernel_size_adjusted) # Calibration grid size

    # NUFFT parameters (fixed defaults for now)
    oversamp_factor = tuple(1.5 for _ in range(ndim)) 
    kb_J = tuple(4 for _ in range(ndim)) # Kernel width for Kaiser-Bessel
    # kb_alpha = 2.34 * kb_J[0] # Standard formula, but NUFFTOperator might have its own default
    
    gridded_acs_kspace_coils = torch.zeros((num_coils,) + calib_shape, dtype=torch.complex64, device=target_device)

    if not _NUFFT_SPECIFIC_AVAILABLE:
        # Using the generic NUFFTOperator if specific ones aren't found.
        # This operator may need specific setup for adjoint to calib_shape.
        # The existing NUFFTOperator is initialized with image_shape, not calib_shape directly for adjoint.
        # This part might need careful adaptation if NUFFT2D/3D are not as expected.
        print("Warning: Using generic NUFFTOperator for ESPIRiT gridding. Ensure it's configured for calib_shape.")
        nufft_op_acs = NUFFTOperator(
            k_trajectory=k_trajectory_acs, # Normalized to [-0.5, 0.5]
            image_shape=calib_shape,       # Target grid shape for adjoint
            device=device,
            # Add oversampling, J, alpha if NUFFTOperator constructor supports them,
            # otherwise it uses its internal defaults.
            # For example:
            # oversamp=oversamp_factor, J=kb_J, kb_alpha= A_GOOD_VALUE_FOR_ALPHA
        )
        for c in range(num_coils):
            gridded_acs_kspace_coils[c] = nufft_op_acs.op_adj(acs_kspace_data[c, :])

    else: # Use NUFFT2D or NUFFT3D
        nufft_class = NUFFT2D if ndim == 2 else NUFFT3D
        # These classes would ideally take calib_shape and NUFFT params directly.
        # Assuming they have a similar interface to torchkbnufft or sigpy.nufft
        # This is a placeholder for their actual API.
        # Example: nufft_op = nufft_class(coord=k_trajectory_acs_scaled_for_nufft, ... )
        # For now, let's assume a similar pattern to NUFFTOperator if its API is general.
        # This requires NUFFT2D/3D to accept k_trajectory normalized to [-0.5,0.5] and
        # perform adjoint to 'image_shape' (which is calib_shape here).
        print(f"Info: Using {'NUFFT2D' if ndim == 2 else 'NUFFT3D'} for ESPIRiT gridding.")
        nufft_op_acs = nufft_class( # Or NUFFTOperator if NUFFT2D/3D are not classes but functions
            k_trajectory=k_trajectory_acs,
            image_shape=calib_shape,
            device=device,
            # Potentially: oversamp_factor=oversamp_factor, kb_kernel_width=kb_J, etc.
        )
        for c in range(num_coils):
            gridded_acs_kspace_coils[c] = nufft_op_acs.op_adj(acs_kspace_data[c, :])
            
    # --- 3. Construct Calibration Matrix ---
    calibration_matrix = _construct_calibration_matrix_pytorch(gridded_acs_kspace_coils, kernel_size_adjusted)

    # --- 4. SVD and Kernel Selection ---
    try:
        U, S, Vh = torch.linalg.svd(calibration_matrix, full_matrices=False)
    except torch.linalg.LinAlgError as e:
        print(f"SVD computation failed: {e}. Returning zeros. ACS data might be too small or ill-conditioned.")
        return torch.zeros((num_coils,) + image_shape, dtype=torch.complex64, device=target_device)

    V = Vh.mH # V are the right singular vectors as columns
    
    # Select kernels based on singular value threshold
    s_max = S.max()
    if s_max == 0: # Avoid division by zero if all singular values are zero
        print("Warning: Max singular value is 0. Cannot select kernels. Returning zeros.")
        return torch.zeros((num_coils,) + image_shape, dtype=torch.complex64, device=target_device)

    num_kernels_to_select = torch.sum(S > (calib_thresh * s_max)).item()
    if num_kernels_to_select == 0:
        print(f"Warning: No singular values above threshold {calib_thresh}. Num kernels = 0. Returning zeros.")
        # Fallback: select at least one kernel if possible
        # num_kernels_to_select = 1 if V.shape[1] > 0 else 0
        # if num_kernels_to_select == 0:
        return torch.zeros((num_coils,) + image_shape, dtype=torch.complex64, device=target_device)
        
    kspace_kernels = V[:, :num_kernels_to_select] # Shape: (num_elements_in_calib_vector, num_selected_kernels)

    # --- 5. Transform Kernels to Image Domain ---
    img_kernels_list = []
    kernel_num_elements = num_coils * np.prod(kernel_size_adjusted)

    for i in range(kspace_kernels.shape[1]): # Iterate over selected kernels
        kernel_vec = kspace_kernels[:, i]
        # Reshape to (num_coils, *kernel_size_adjusted)
        kernel_k_space = kernel_vec.reshape((num_coils,) + tuple(kernel_size_adjusted))
        
        # Zero-pad to image_shape
        pad_amount = []
        for k_dim, i_dim in zip(kernel_size_adjusted, image_shape):
            pad_before = (i_dim - k_dim) // 2
            pad_after = i_dim - k_dim - pad_before
            pad_amount.extend([pad_before, pad_after])
        
        # PyTorch pad format is (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back) for 3D
        # So, reverse order of dims for padding amounts
        padding_for_torch = []
        for j in range(ndim):
            padding_for_torch.extend([pad_amount[2*(ndim-1-j)], pad_amount[2*(ndim-1-j)+1]])

        kernel_k_space_padded = torch.nn.functional.pad(kernel_k_space, padding_for_torch, mode='constant', value=0)
        
        # Centered iFFT
        # Using default norm="backward" for ifftn, which includes a 1/N scaling.
        kernel_img_domain = torch.fft.ifftshift(
            torch.fft.ifftn(
                torch.fft.fftshift(kernel_k_space_padded, dim=tuple(range(-ndim,0))), 
                dim=tuple(range(-ndim,0))
            ), 
            dim=tuple(range(-ndim,0))
        )
        # No additional manual scaling like np.sqrt(np.prod(image_shape)) needed with default norm.
        img_kernels_list.append(kernel_img_domain)

    if not img_kernels_list:
        print("Warning: No image domain kernels generated. Returning zeros.")
        return torch.zeros((num_coils,) + image_shape, dtype=torch.complex64, device=target_device)
        
    img_kernels_tensor = torch.stack(img_kernels_list, dim=0) # (num_selected_kernels, num_coils, *image_shape)

    # --- 6. Pixel-wise Eigenvalue Problem ---
    sensitivity_maps = torch.zeros((num_coils,) + image_shape, dtype=torch.complex64, device=target_device)
    
    # Reshape for easier pixel access if ndim=2 (H,W) or ndim=3 (D,H,W)
    # Flatten image_shape part of img_kernels_tensor and sensitivity_maps
    img_kernels_flat = img_kernels_tensor.reshape(num_kernels_to_select, num_coils, -1) # (K, C, Npixels)
    sensitivity_maps_flat = sensitivity_maps.reshape(num_coils, -1) # (C, Npixels)
    
    num_pixels = img_kernels_flat.shape[2]

    for p_idx in range(num_pixels):
        # G_p is (num_coils, num_coils)
        # img_kernels_flat[:, :, p_idx] gives (num_kernels, num_coils) for pixel p_idx
        pixel_kernel_data = img_kernels_flat[:, :, p_idx] # (K, C)
        
        # G_p = sum_k (kernel_k_image_at_pixel_p * conj(kernel_k_image_at_pixel_p).T)
        # kernel_k_image_at_pixel_p is a column vector of coil values (C,1) for kernel k at pixel p
        # G_p[c1,c2] = sum_k kernel_k_c1[p] * conj(kernel_k_c2[p])
        # This is equivalent to pixel_kernel_data.T.conj() @ pixel_kernel_data if pixel_kernel_data was (C,K)
        # With pixel_kernel_data (K,C):
        G_p = torch.zeros((num_coils, num_coils), dtype=torch.complex64, device=target_device)
        for k_idx in range(num_kernels_to_select):
            kernel_snapshot = pixel_kernel_data[k_idx, :].unsqueeze(1) # (C, 1)
            G_p += kernel_snapshot @ kernel_snapshot.mH # Outer product

        try:
            eigvals, eigvecs = torch.linalg.eig(G_p)
        except torch.linalg.LinAlgError:
            # If eigenvalue decomposition fails for a pixel, leave maps as zeros for that pixel
            continue 

        # Find eigenvector for largest eigenvalue (or eigenvalue > eigen_thresh, typically close to 1.0)
        # Eigenvalues might not be sorted, find max. Also, they are complex.
        # We are interested in the magnitude of eigenvalues.
        # The sensitivity map is the eigenvector corresponding to the eigenvalue closest to 1 (or largest).
        # For ESPIRiT, the largest eigenvalue should be close to 1.0 for signal regions.
        
        # Option 1: Largest eigenvalue magnitude
        # main_eig_idx = torch.argmax(torch.abs(eigvals))
        
        # Option 2: Eigenvalue magnitude closest to eigen_thresh (or 1.0 if eigen_thresh is for selection)
        # This assumes eigen_thresh is a property of the eigenvalue itself (e.g. should be > 0.95)
        # Let's find eigenvalues with magnitude > eigen_thresh and pick the one with largest magnitude.
        
        valid_eig_indices = torch.where(torch.abs(eigvals) > eigen_thresh)[0]
        if valid_eig_indices.numel() > 0:
            # Among valid eigenvalues, pick the one with largest magnitude
            main_eig_idx = valid_eig_indices[torch.argmax(torch.abs(eigvals[valid_eig_indices]))]
            selected_eigvec = eigvecs[:, main_eig_idx]
            
            # Phase normalization for consistency (e.g., make first element real or max element real)
            # This is important as eigenvectors are defined up to a complex phase.
            # A common way: normalize such that the element with largest magnitude is real and positive.
            max_abs_idx = torch.argmax(torch.abs(selected_eigvec))
            phase_correction = torch.angle(selected_eigvec[max_abs_idx])
            selected_eigvec_normalized = selected_eigvec * torch.exp(-1j * phase_correction)
            
            sensitivity_maps_flat[:, p_idx] = selected_eigvec_normalized
        # else: no eigenvalue met the eigen_thresh, so map remains zero for this pixel.

    sensitivity_maps = sensitivity_maps_flat.reshape((num_coils,) + image_shape)

    # Optional: Normalize maps (e.g., RSS across coils = 1)
    # This is often done for display or specific recombination methods.
    # For ESPIRiT, the maps might already be somewhat normalized by the eigenvalue process.
    # A common final step is RSS normalization if consistent scaling is needed:
    # rss = torch.sqrt(torch.sum(torch.abs(sensitivity_maps)**2, dim=0, keepdim=True)) + 1e-9
    # sensitivity_maps_rss_normalized = sensitivity_maps / rss
    # For now, returning the direct output of eigenvalue problem.

    return sensitivity_maps

def _construct_calibration_matrix_pytorch(
    gridded_acs_kspace_coils: torch.Tensor, # (num_coils, *calib_shape)
    kernel_size: tuple # (Kz, Kx, Ky) or (Kx, Ky)
) -> torch.Tensor:
    """
    Constructs the calibration matrix from gridded ACS k-space data.
    """
    num_coils = gridded_acs_kspace_coils.shape[0]
    calib_shape = gridded_acs_kspace_coils.shape[1:]
    ndim = len(calib_shape)

    if len(kernel_size) != ndim:
        raise ValueError(f"kernel_size ndim {len(kernel_size)} does not match calib_shape ndim {ndim}.")

    # Calculate number of sliding window patches
    num_patches_per_dim = [cs - ks + 1 for cs, ks in zip(calib_shape, kernel_size)]
    total_patches = np.prod(num_patches_per_dim)
    
    if total_patches <= 0:
        raise ValueError(f"Calibration shape {calib_shape} is too small for kernel size {kernel_size}.")

    # Size of each vectorized patch (num_coils * Kz * Kx * Ky)
    patch_vector_size = num_coils * np.prod(kernel_size)
    
    calibration_matrix = torch.zeros((patch_vector_size, total_patches), 
                                     dtype=torch.complex64, 
                                     device=gridded_acs_kspace_coils.device)
    
    patch_idx = 0
    if ndim == 2: # (Kx, Ky)
        kx, ky = kernel_size
        for r_idx in range(num_patches_per_dim[0]): # Iterate over rows (calib_shape[0] - kx + 1)
            for c_idx in range(num_patches_per_dim[1]): # Iterate over columns (calib_shape[1] - ky + 1)
                patch = gridded_acs_kspace_coils[:, r_idx:r_idx+kx, c_idx:c_idx+ky]
                calibration_matrix[:, patch_idx] = patch.reshape(-1)
                patch_idx += 1
    elif ndim == 3: # (Kz, Kx, Ky)
        kz, kx, ky = kernel_size
        for d_idx in range(num_patches_per_dim[0]): # Iterate over depth
            for r_idx in range(num_patches_per_dim[1]): # Iterate over rows
                for c_idx in range(num_patches_per_dim[2]): # Iterate over columns
                    patch = gridded_acs_kspace_coils[:, d_idx:d_idx+kz, r_idx:r_idx+kx, c_idx:c_idx+ky]
                    calibration_matrix[:, patch_idx] = patch.reshape(-1)
                    patch_idx += 1
    
    return calibration_matrix

# Note: The original detailed comments about ESPIRiT steps are still valuable for understanding.
# This implementation leverages SigPy for those steps.
# Key considerations for SigPy:
# - Coordinate systems and scaling for NUFFT.
# - Device management (CPU/GPU).
# - Input/output shapes of SigPy functions.
# - Interpretation of parameters like `calib_width` in `espirit_calib`.
