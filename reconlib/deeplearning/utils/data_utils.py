import torch
import numpy as np
# Adjust path if NUFFTOperator is not directly importable
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))) # Go up three levels for reconlib
from reconlib.operators import NUFFTOperator # For zero-filled recon

def generate_cartesian_undersampling_mask(
    image_shape: tuple[int, ...], 
    acceleration_factor: float, 
    num_center_lines: int = 16,
    seed: int | None = None
) -> torch.Tensor:
    """
    Generates a simple Cartesian undersampling mask for 2D or 3D.
    For 2D (H, W), undersamples along the width dimension (phase-encoding).
    For 3D (D, H, W), undersamples along the height (phase-encoding dim 1) 
    and width (phase-encoding dim 2) dimensions. Keeps center lines fully sampled.

    Args:
        image_shape: Tuple of image dimensions (e.g., (H, W) or (D, H, W)).
        acceleration_factor: Target acceleration (e.g., 4 for R=4).
        num_center_lines: Number of lines in the k-space center to keep fully sampled.
                          For 3D, this applies to both phase-encoding dimensions.
        seed: Optional random seed for reproducible mask generation.

    Returns:
        A boolean or float tensor mask of the same shape as k-space (image_shape).
        True or 1.0 where k-space is sampled, False or 0.0 otherwise.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed) # For numpy random operations if any

    num_dims = len(image_shape)
    mask = torch.zeros(image_shape, dtype=torch.float32)

    if num_dims == 2: # (H, W), undersample W (phase-encoding)
        num_lines = image_shape[1]
        lines_to_keep = int(num_lines / acceleration_factor)
        
        center_start = (num_lines - num_center_lines) // 2
        center_end = center_start + num_center_lines
        
        mask[:, center_start:center_end] = 1.0
        
        remaining_lines_to_sample = lines_to_keep - num_center_lines
        if remaining_lines_to_sample < 0: remaining_lines_to_sample = 0
            
        outer_region_indices = [i for i in range(num_lines) if not (center_start <= i < center_end)]
        
        if remaining_lines_to_sample > 0 and len(outer_region_indices) > 0:
            if remaining_lines_to_sample >= len(outer_region_indices):
                 # Keep all outer lines if remaining_lines_to_sample is too high (low acceleration)
                for i in outer_region_indices:
                    mask[:, i] = 1.0
            else:
                chosen_outer_indices = np.random.choice(outer_region_indices, remaining_lines_to_sample, replace=False)
                for i in chosen_outer_indices:
                    mask[:, i] = 1.0
                    
    elif num_dims == 3: # (D, H, W), undersample H (PE1) and W (PE2)
        # This creates a 2D mask and then tiles it across the D dimension (frequency encoding)
        num_lines_h = image_shape[1] # PE1
        num_lines_w = image_shape[2] # PE2

        lines_to_keep_h = int(num_lines_h / np.sqrt(acceleration_factor)) # Approx for 2D undersampling
        lines_to_keep_w = int(num_lines_w / np.sqrt(acceleration_factor))

        center_start_h = (num_lines_h - num_center_lines) // 2
        center_end_h = center_start_h + num_center_lines
        center_start_w = (num_lines_w - num_center_lines) // 2
        center_end_w = center_start_w + num_center_lines

        mask_2d_slice = torch.zeros((num_lines_h, num_lines_w), dtype=torch.float32)
        mask_2d_slice[center_start_h:center_end_h, :] = 1.0 # Fully sample center H lines first
        mask_2d_slice[:, center_start_w:center_end_w] = 1.0 # Then fully sample center W lines (union)

        # Count actually sampled lines in center (avoid double counting intersection)
        center_sampled_count = torch.sum(mask_2d_slice).item()
        
        # For outer region, sample randomly based on remaining points needed
        # This is a simplification; true variable density random sampling is more complex.
        # Here, we just ensure the total number of points is roughly met.
        total_points_to_keep = int((num_lines_h * num_lines_w) / acceleration_factor)
        remaining_points_to_sample = total_points_to_keep - center_sampled_count
        if remaining_points_to_sample < 0: remaining_points_to_sample = 0

        outer_region_indices_flat = []
        for r in range(num_lines_h):
            for c in range(num_lines_w):
                if mask_2d_slice[r, c] == 0: # If not already in center
                    outer_region_indices_flat.append(r * num_lines_w + c)
        
        if remaining_lines_to_sample > 0 and len(outer_region_indices_flat) > 0:
            if remaining_points_to_sample >= len(outer_region_indices_flat):
                for flat_idx in outer_region_indices_flat:
                    r, c = flat_idx // num_lines_w, flat_idx % num_lines_w
                    mask_2d_slice[r,c] = 1.0
            else:
                chosen_flat_indices = np.random.choice(outer_region_indices_flat, remaining_lines_to_sample, replace=False)
                for flat_idx in chosen_flat_indices:
                    r, c = flat_idx // num_lines_w, flat_idx % num_lines_w
                    mask_2d_slice[r,c] = 1.0
        
        mask[:, :, :] = mask_2d_slice.unsqueeze(0) # Tile across the D dimension

    else:
        raise ValueError(f"Unsupported image dimensionality: {num_dims}. Only 2D or 3D.")
        
    return mask


def apply_cartesian_mask(k_space_full: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Applies a Cartesian undersampling mask to fully sampled k-space data.
    Assumes k_space_full is already FFT-shifted (center at center of matrix).
    """
    if k_space_full.shape != mask.shape:
        raise ValueError(f"k_space_full shape {k_space_full.shape} must match mask shape {mask.shape}")
    return k_space_full * mask


def get_zero_filled_reconstruction(
    k_space_undersampled_cartesian: torch.Tensor, 
    image_shape: tuple[int,...], 
    device: str | torch.device = 'cpu'
) -> torch.Tensor:
    """
    Performs a zero-filled reconstruction from Cartesian undersampled k-space.
    This is a simple IFFT.
    k_space_undersampled_cartesian: FFT-shifted k-space data.
    """
    if not isinstance(device, torch.device): # Ensure device is a torch.device object
        device = torch.device(device)
        
    if k_space_undersampled_cartesian.device != device:
        k_space_undersampled_cartesian = k_space_undersampled_cartesian.to(device)
    
    # Ensure complex type
    if not k_space_undersampled_cartesian.is_complex():
        k_space_undersampled_cartesian = k_space_undersampled_cartesian.to(torch.complex64)

    # IFFT
    img_zero_filled = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(k_space_undersampled_cartesian), s=image_shape))
    # Scaling to roughly match input data magnitude (optional, but often done)
    # img_zero_filled = img_zero_filled * float(torch.prod(torch.tensor(image_shape)))
    return img_zero_filled


if __name__ == '__main__':
    print("Testing data_utils...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 2D Mask Test ---
    ishape2d = (128, 128)
    accel2d = 4
    center2d = 24
    mask2d = generate_cartesian_undersampling_mask(ishape2d, accel2d, center2d, seed=0)
    print(f"2D Mask shape: {mask2d.shape}, Sum: {mask2d.sum()}, Expected sum approx: {ishape2d[0]*ishape2d[1]/accel2d}")
    assert mask2d.shape == ishape2d
    # Basic check for center lines
    assert torch.all(mask2d[:, (ishape2d[1]-center2d)//2 : (ishape2d[1]+center2d)//2] == 1.0)

    # --- 3D Mask Test ---
    ishape3d = (32, 64, 64)
    accel3d = 4
    center3d = 12
    mask3d = generate_cartesian_undersampling_mask(ishape3d, accel3d, center3d, seed=0)
    print(f"3D Mask shape: {mask3d.shape}, Sum: {mask3d.sum()}, Expected sum approx: {ishape3d[0]*ishape3d[1]*ishape3d[2]/accel3d}")
    assert mask3d.shape == ishape3d
    # Basic check for center slice (assuming it's tiled)
    center_slice_mask_2d = mask3d[0,:,:]
    assert torch.all(center_slice_mask_2d[(ishape3d[1]-center3d)//2 : (ishape3d[1]+center3d)//2, : ] == 1.0)
    assert torch.all(center_slice_mask_2d[:, (ishape3d[2]-center3d)//2 : (ishape3d[2]+center3d)//2 ] == 1.0)


    # --- Apply Mask and Zero-fill Test (2D) ---
    phantom_2d_test = torch.randn(ishape2d, dtype=torch.complex64, device=device)
    kspace_full_2d = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(phantom_2d_test)))
    
    kspace_undersampled_2d = apply_cartesian_mask(kspace_full_2d, mask2d.to(device))
    img_zf_2d = get_zero_filled_reconstruction(kspace_undersampled_2d, ishape2d, device=device)
    print(f"Zero-filled 2D image shape: {img_zf_2d.shape}")
    assert img_zf_2d.shape == ishape2d

    print("data_utils basic tests completed.")
