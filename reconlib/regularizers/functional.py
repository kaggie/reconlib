import torch
import torch.nn.functional as F # For F.pad in Total Variation
import math

def l1_norm(image: torch.Tensor) -> torch.Tensor:
    """Computes the L1 norm of an image (sum of absolute values)."""
    return torch.sum(torch.abs(image))

def l2_norm_squared(image: torch.Tensor) -> torch.Tensor:
    """Computes the squared L2 norm of an image (sum of squared values)."""
    return torch.sum(image * torch.conj(image)).real # Ensure real output for complex case

def total_variation(image: torch.Tensor, isotropic: bool = True) -> torch.Tensor:
    """
    Computes the Total Variation (TV) of an image (2D or 3D).
    Assumes image is (D, H, W) or (H, W) or (N, C, D, H, W) etc.
    TV is computed for the spatial dimensions (last 2 or 3 dims).
    """
    ndim = image.ndim
    if ndim < 2:
        raise ValueError("Image must be at least 2D to compute Total Variation.")

    # Determine number of spatial dimensions to consider (typically last 2 or 3)
    # Heuristic: if ndim is 2 (H,W) or 3 (C,H,W or D,H,W where C/D is small), spatial_dims=2.
    # If ndim is 4 (N,C,H,W or N,D,H,W) or 5 (N,C,D,H,W), spatial_dims could be 2 or 3.
    # Let's assume for N,C,H,W -> spatial_dims=2 (H,W)
    # For N,C,D,H,W -> spatial_dims=3 (D,H,W)
    # This means we look at the trailing dimensions.
    
    if ndim == 2: # H, W
        spatial_dims_to_process = [0, 1]
    elif ndim == 3: # C, H, W or D, H, W
        # If first dim is small (e.g. <=4 for channels), assume spatial are last 2.
        # Otherwise, assume all 3 are spatial (D,H,W).
        spatial_dims_to_process = [1, 2] if image.shape[0] <= 4 and ndim >2 else [0, 1, 2]
    elif ndim == 4: # N, C, H, W or N, D, H, W
        # Assume last 2 for (N,C,H,W) or last 3 if first dim is batch (N,D,H,W)
        spatial_dims_to_process = [2, 3] if image.shape[1] <=4 else [1,2,3]
    elif ndim == 5: # N, C, D, H, W
        spatial_dims_to_process = [2, 3, 4] # D, H, W
    else: # Default to last 2 for higher dims, or raise error
        raise ValueError(f"TV for ndim={ndim} not explicitly handled by this heuristic. Please check spatial_dims_to_process.")

    if not isotropic:
        tv_sum_aniso = torch.tensor(0.0, device=image.device, dtype=image.real.dtype if image.is_complex() else image.dtype)
        for d_axis_idx in spatial_dims_to_process:
            # Ensure d_axis_idx is within bounds of actual image dimensions
            if d_axis_idx >= ndim: continue

            slice_all = slice(None)
            slicers_start = [slice_all] * ndim
            slicers_end = [slice_all] * ndim
            
            slicers_start[d_axis_idx] = slice(None, -1)
            slicers_end[d_axis_idx] = slice(1, None)
            
            grad_d = image[tuple(slicers_end)] - image[tuple(slicers_start)]
            tv_sum_aniso = tv_sum_aniso + torch.sum(torch.abs(grad_d))
        return tv_sum_aniso
    else: # Isotropic TV
        # Sum of squared gradients for each spatial dimension
        # Gradients need to be padded to match original image shape before summing squares
        
        # Determine the shape of the spatial part of the image for gradient calculation
        spatial_shape_for_grad_sum = list(image.shape)
        for i in range(ndim - len(spatial_dims_to_process)): # Zero out non-spatial leading dims
            spatial_shape_for_grad_sum[i] = 1 
        # Make it compatible for broadcasting with gradients from each spatial dim
        # Example: if image is (N,C,H,W) and spatial_dims_to_process are [2,3] (H,W)
        # gradients_sq_sum should be like (N,C,H,W) to accumulate gx^2, gy^2
        # If image is (H,W), then (H,W)
        
        # The gradients will be computed on slices that are one smaller along the diff dim.
        # We need a reference shape for the sum of squares. Max spatial extent for padding.
        # It's simpler to pad each gradient back to original image size, then sum.
        
        sum_of_sq_gradients = torch.zeros_like(image, dtype=image.real.dtype if image.is_complex() else image.dtype)

        for d_axis_idx in spatial_dims_to_process:
            if d_axis_idx >= ndim: continue

            slice_all = slice(None)
            s_start = [slice_all] * ndim
            s_end = [slice_all] * ndim
            s_start[d_axis_idx] = slice(None, -1)
            s_end[d_axis_idx] = slice(1, None)
            
            grad_d = image[tuple(s_end)] - image[tuple(s_start)]
            
            # Pad grad_d back to original image size along the current dimension d_axis_idx
            # Padding format for F.pad: (pad_left, pad_right, pad_top, pad_bottom, ...)
            # It applies from the last dim to the first.
            padding_config = [0] * (2 * ndim) # (pad_dimN_left, pad_dimN_right, pad_dimN-1_left, ...)
            
            # Calculate which pair in padding_config corresponds to d_axis_idx
            # Torch pad expects (..., pad_dim1_start, pad_dim1_end, pad_dim0_start, pad_dim0_end)
            # If d_axis_idx is 0 for (H,W,D), it's the H dim.
            # If ndim=3, d_axis_idx=0 (H): corresponds to 2*(3-1-0)+1 = 5th element for right pad (end of dim)
            # If ndim=3, d_axis_idx=1 (W): corresponds to 2*(3-1-1)+1 = 3rd element
            # If ndim=3, d_axis_idx=2 (D): corresponds to 2*(3-1-2)+1 = 1st element
            
            # Axis 'd_axis_idx' needs padding (0,1) at its end to restore size.
            # The dimension in F.pad ordering is (ndim - 1 - d_axis_idx)
            dim_pair_idx_in_pad_config = 2 * (ndim - 1 - d_axis_idx)
            padding_config[dim_pair_idx_in_pad_config] = 0 # pad_left/start for this dim
            padding_config[dim_pair_idx_in_pad_config + 1] = 1 # pad_right/end for this dim
            
            grad_d_padded = F.pad(grad_d, tuple(padding_config))
            sum_of_sq_gradients = sum_of_sq_gradients + (grad_d_padded * torch.conj(grad_d_padded)).real
            
        return torch.sum(torch.sqrt(sum_of_sq_gradients))


def charbonnier_penalty(image: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    """Computes the Charbonnier penalty sum(sqrt(x_i^2 + epsilon^2))."""
    # Ensure epsilon is on the same device and type for robustness
    epsilon_sq = torch.tensor(epsilon**2, device=image.device, dtype=image.real.dtype if image.is_complex() else image.dtype)
    return torch.sum(torch.sqrt(image * torch.conj(image) + epsilon_sq).real)


def huber_penalty(image: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """
    Computes the Huber penalty.
    L_delta(a) = 0.5 * a^2 if |a| <= delta
                 delta * (|a| - 0.5 * delta) if |a| > delta
    """
    abs_image = torch.abs(image)
    loss_sq = 0.5 * (image * torch.conj(image)).real # For complex, use |image|^2
    loss_lin = delta * (abs_image - 0.5 * delta)
    
    use_linear_loss = abs_image > delta
    
    return torch.sum(torch.where(use_linear_loss, loss_lin, loss_sq))

if __name__ == '__main__':
    print("Testing regularizer value functions...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a sample 2D image (e.g., batch_size=1, channels=1, H, W)
    img_2d_real = torch.randn(1, 1, 32, 32, device=device)
    img_2d_cplx = torch.randn(1, 1, 32, 32, dtype=torch.complex64, device=device)

    # Create a sample 3D image (N,C,D,H,W)
    img_3d_real_ncdhw = torch.randn(1, 1, 16, 32, 32, device=device)
    # Create a sample 3D image (D,H,W)
    img_3d_real_dhw = torch.randn(16, 32, 32, device=device)


    print(f"L1(2D_real): {l1_norm(img_2d_real).item()}")
    print(f"L2sq(2D_real): {l2_norm_squared(img_2d_real).item()}")
    print(f"L1(2D_cplx): {l1_norm(img_2d_cplx).item()}") # abs is used
    print(f"L2sq(2D_cplx): {l2_norm_squared(img_2d_cplx).item()}")

    print(f"TV_iso(2D_real NCHW): {total_variation(img_2d_real, isotropic=True).item()}")
    print(f"TV_aniso(2D_real NCHW): {total_variation(img_2d_real, isotropic=False).item()}")
    
    print(f"TV_iso(3D_real NCDHW): {total_variation(img_3d_real_ncdhw, isotropic=True).item()}")
    print(f"TV_aniso(3D_real NCDHW): {total_variation(img_3d_real_ncdhw, isotropic=False).item()}")

    print(f"TV_iso(3D_real DHW): {total_variation(img_3d_real_dhw, isotropic=True).item()}")
    print(f"TV_aniso(3D_real DHW): {total_variation(img_3d_real_dhw, isotropic=False).item()}")

    
    # Test with (H,W) image for TV
    img_hw = torch.randn(32,32, device=device)
    print(f"TV_iso(HW): {total_variation(img_hw, isotropic=True).item()}")


    print(f"Charbonnier(2D_real, eps=1e-3): {charbonnier_penalty(img_2d_real, epsilon=1e-3).item()}")
    print(f"Charbonnier(2D_cplx, eps=1e-3): {charbonnier_penalty(img_2d_cplx, epsilon=1e-3).item()}")

    print(f"Huber(2D_real, delta=0.5): {huber_penalty(img_2d_real, delta=0.5).item()}")
    print(f"Huber(2D_cplx, delta=0.5): {huber_penalty(img_2d_cplx, delta=0.5).item()}")
    
    print("Regularizer value function tests completed.")
