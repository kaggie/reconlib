# reconlib/phase_unwrapping/quality_guided_unwrap.py
"""3D Quality-Guided Phase Unwrapping in PyTorch."""

import torch
import numpy as np
from queue import PriorityQueue
# torchvision.transforms.functional might be used for blurring if Conv3D is not preferred
# import torchvision.transforms.functional as TF

def wrap_phase(phase_tensor: torch.Tensor) -> torch.Tensor:
    """Wraps phase values to the interval [-pi, pi) using PyTorch operations."""
    return (phase_tensor + np.pi) % (2 * np.pi) - np.pi

def _get_gradient_magnitude_3d(phase_tensor_3d: torch.Tensor) -> torch.Tensor:
    """
    Computes the 3D gradient magnitude using torch.diff.
    Prepends to keep original dimensions.
    """
    dz = torch.diff(phase_tensor_3d, dim=0, prepend=phase_tensor_3d[:1,...])
    dy = torch.diff(phase_tensor_3d, dim=1, prepend=phase_tensor_3d[:,:1,...])
    dx = torch.diff(phase_tensor_3d, dim=2, prepend=phase_tensor_3d[:,:,:1])
    return torch.sqrt(dz**2 + dy**2 + dx**2)

def _gaussian_blur_3d_conv(
    tensor_3d: torch.Tensor, 
    sigma_blur: float, 
    device: torch.device
) -> torch.Tensor:
    """
    Applies 3D Gaussian blur using torch.nn.Conv3d.
    Internal helper function.
    """
    if sigma_blur <= 0:
        return tensor_3d

    # Determine kernel size (odd integer)
    kernel_size = int(2 * round(2.5 * sigma_blur) + 1) 
    if kernel_size < 3: kernel_size = 3

    # Create 1D Gaussian kernel
    gauss_1d = torch.exp(-torch.arange(-(kernel_size//2), kernel_size//2 + 1, 
                                     dtype=torch.float32, device=device)**2 / (2*sigma_blur**2))
    gauss_1d = gauss_1d / gauss_1d.sum()

    # Create 3D kernel from outer products of 1D kernel
    kernel_3d = gauss_1d.unsqueeze(0).unsqueeze(0) * \
                gauss_1d.unsqueeze(0).unsqueeze(1) * \
                gauss_1d.unsqueeze(1).unsqueeze(0) 
    kernel_3d = kernel_3d.unsqueeze(0).unsqueeze(0) # Shape: (1, 1, k, k, k) for Conv3d

    # Pad the input tensor
    pad_size = kernel_size // 2
    tensor_padded = torch.nn.functional.pad(
        tensor_3d.unsqueeze(0).unsqueeze(0), # Shape: (1, 1, D, H, W)
        (pad_size, pad_size, pad_size, pad_size, pad_size, pad_size), 
        mode='replicate' # 'replicate' or 'reflect' are common choices
    )
    
    # Apply 3D convolution
    blurred_tensor = torch.nn.functional.conv3d(tensor_padded, kernel_3d).squeeze(0).squeeze(0) # Back to (D,H,W)
    return blurred_tensor

def unwrap_phase_3d_quality_guided(
    wrapped_phase: torch.Tensor, 
    quality_metric: str = 'gradient_magnitude', 
    sigma_blur: float = 1.0
) -> torch.Tensor:
    """
    Performs 3D quality-guided phase unwrapping on a PyTorch tensor.

    This algorithm unwraps phase by starting from a high-quality seed point and
    iteratively unwrapping adjacent voxels in order of their quality. The quality
    is typically derived from phase gradients; regions with lower phase gradients
    (smoother phase) are considered higher quality.

    The process involves:
    1.  Calculating a quality map (e.g., based on inverse gradient magnitude of the wrapped phase).
    2.  Optionally blurring the quality map to reduce noise.
    3.  Initializing a priority queue with a seed voxel (highest quality).
    4.  Iteratively extracting voxels from the queue and unwrapping their unvisited neighbors,
        adding them to the queue based on their quality.

    Args:
        wrapped_phase (torch.Tensor): 3D tensor of wrapped phase values (in radians).
                                      Shape (D, H, W). Must be a PyTorch tensor.
        quality_metric (str, optional): Method to compute the quality map.
                                        Currently, only 'gradient_magnitude' is supported, where
                                        higher gradient magnitude implies lower quality.
                                        Defaults to 'gradient_magnitude'.
        sigma_blur (float, optional): Standard deviation for Gaussian blur applied to the
                                      quality map. Blurring can help in noisy data.
                                      If 0 or negative, no blurring is applied. Defaults to 1.0.

    Returns:
        torch.Tensor: 3D tensor of unwrapped phase values, on the same device as input.
    """
    if not isinstance(wrapped_phase, torch.Tensor):
        raise TypeError("wrapped_phase must be a PyTorch tensor.")
    if wrapped_phase.ndim != 3:
        raise ValueError(f"wrapped_phase must be a 3D tensor, got shape {wrapped_phase.shape}")

    device = wrapped_phase.device

    # 1. Calculate Quality Map
    if quality_metric == 'gradient_magnitude':
        grad_mag = _get_gradient_magnitude_3d(wrapped_phase)
        if sigma_blur > 0:
            quality_map_blurred = _gaussian_blur_3d_conv(grad_mag, sigma_blur, device)
        else:
            quality_map_blurred = grad_mag
        # Higher gradient magnitude means lower quality (more phase jumps).
        # PriorityQueue gets minimum first, so we use -quality or 1/(quality+eps).
        # Here, we use -grad_mag, so higher original grad_mag (lower quality) gets lower priority (less negative).
        # Or, if quality is good (low grad_mag), -grad_mag is higher (closer to 0).
        # We want to process high quality first. So, highest quality = smallest grad_mag.
        # PriorityQueue extracts minimum. So, store -grad_mag for "max-heap" like behavior or use 1/grad_mag.
        # Let's make quality positive: higher is better. So small grad_mag -> high quality.
        # quality = 1.0 / (quality_map_blurred + 1e-9) # Add epsilon to avoid division by zero
        # Maximize quality (argmax). For PriorityQueue (min-heap), use -quality.
        quality = -quality_map_blurred # Higher blur_grad_mag is lower quality, so this is more negative = lower priority
    else:
        raise NotImplementedError(f"Quality metric '{quality_metric}' not implemented.")

    # 2. Initialization
    unwrapped_phase = torch.zeros_like(wrapped_phase, device=device)
    visited = torch.zeros_like(wrapped_phase, dtype=torch.bool, device=device)
    
    pq = PriorityQueue()

    # Find starting index (highest quality = minimum blurred gradient magnitude = maximum of `quality` because it's negated)
    start_idx_flat = torch.argmax(quality).item() # item() to get scalar for unravel_index
    start_idx = np.unravel_index(start_idx_flat, quality.shape) # (z, y, x) tuple

    unwrapped_phase[start_idx] = wrapped_phase[start_idx]
    visited[start_idx] = True
    # PriorityQueue stores (priority_value, z_idx, y_idx, x_idx)
    # We use quality[start_idx].item() as priority. Higher quality (less negative) should be processed first.
    pq.put((quality[start_idx].item(), start_idx[0], start_idx[1], start_idx[2]))

    neighbors = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
    dims = wrapped_phase.shape # (D, H, W)

    # 3. Main Loop (Quality-Guided Flood Fill)
    while not pq.empty():
        _, z, y, x = pq.get() # Current pixel to process

        current_phase_scalar = unwrapped_phase[z, y, x].item() # Scalar for calculations

        for dz, dy, dx in neighbors:
            nz, ny, nx = z + dz, y + dy, x + dx

            # Boundary checks
            if not (0 <= nz < dims[0] and 0 <= ny < dims[1] and 0 <= nx < dims[2]):
                continue

            if not visited[nz, ny, nx]:
                visited[nz, ny, nx] = True
                
                # Phase difference between neighbor and current voxel
                # wrapped_phase[nz,ny,nx] - unwrapped_phase[z,y,x] is WRONG.
                # It should be wrapped_phase[nz,ny,nx] - wrapped_phase[z,y,x]
                # then this wrapped difference is added to the unwrapped_phase[z,y,x]
                
                diff_wrapped = wrap_phase(wrapped_phase[nz, ny, nx] - wrapped_phase[z, y, x])
                unwrapped_phase[nz, ny, nx] = current_phase_scalar + diff_wrapped
                
                # Add neighbor to queue with its quality as priority
                pq.put((quality[nz, ny, nx].item(), nz, ny, nx))
                
    return unwrapped_phase
