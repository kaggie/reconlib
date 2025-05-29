# reconlib/phase_unwrapping/romeo.py
"""
Native PyTorch implementation of a region-growing phase unwrapping algorithm
inspired by ROMEO (Rapid Open-source Minimum spanning treE algOrithm) principles.
"""

import torch
import collections
from typing import Optional, Tuple, List, Deque

def _get_neighbors_romeo(
    voxel_idx: Tuple[int, ...],
    shape: Tuple[int, ...],
    mask: Optional[torch.Tensor],
    connectivity: int
) -> List[Tuple[int, ...]]:
    """
    Gets valid neighbors of a voxel given connectivity.

    Args:
        voxel_idx: Tuple of integer coordinates for the current voxel (e.g., (d,h,w) or (h,w)).
        shape: Tuple representing the shape of the image volume.
        mask: Optional boolean tensor. If provided, neighbors outside the mask are excluded.
        connectivity: Defines neighborhood:
            - 1: Face neighbors (6 in 3D, 4 in 2D).
            - 2: Face and edge neighbors (18 in 3D). Not implemented for 2D.
            - 3: Face, edge, and corner neighbors (26 in 3D). Not implemented for 2D.

    Returns:
        List of valid neighbor coordinate tuples.
    """
    neighbors: List[Tuple[int, ...]] = []
    ndim = len(shape)
    
    if ndim == 2: # (H, W)
        h, w = voxel_idx
        offsets_conn1 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        all_offsets = offsets_conn1 # For 2D, only connectivity 1 is typically used/simple
        if connectivity > 1:
            # Add diagonal for connectivity 2 (8-connectivity)
            offsets_conn2_diag = [(1,1), (1,-1), (-1,1), (-1,-1)]
            all_offsets.extend(offsets_conn2_diag)

    elif ndim == 3: # (D, H, W)
        d, h, w = voxel_idx
        offsets_conn1 = [
            (0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)
        ]
        all_offsets = offsets_conn1
        if connectivity >= 2: # Add edge neighbors
            offsets_conn2_edges = [
                (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
            ]
            all_offsets.extend(offsets_conn2_edges)
        if connectivity >= 3: # Add corner neighbors
            offsets_conn3_corners = [
                (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
            ]
            all_offsets.extend(offsets_conn3_corners)
    else:
        raise ValueError(f"Unsupported number of dimensions: {ndim}. Expected 2 or 3.")

    for offset in all_offsets:
        neighbor_coords_list = [voxel_idx[i] + offset[i] for i in range(ndim)]
        
        valid = True
        for i in range(ndim):
            if not (0 <= neighbor_coords_list[i] < shape[i]):
                valid = False
                break
        if not valid:
            continue

        neighbor_coords_tuple = tuple(neighbor_coords_list)
        if mask is not None and not mask[neighbor_coords_tuple].item():
            continue
        
        neighbors.append(neighbor_coords_tuple)
        
    return neighbors

def _compute_romeo_quality_map_spatial(
    phase_data_for_quality: torch.Tensor,
    mask: Optional[torch.Tensor],
    voxel_size: Tuple[float, ...], # e.g. (vz, vy, vx) or (vy, vx)
    is_magnitude_provided: bool
) -> torch.Tensor:
    """
    Computes a spatial quality map.
    If magnitude is provided, it's used as quality. Otherwise, inverse phase gradient magnitude.
    """
    device = phase_data_for_quality.device
    ndim = phase_data_for_quality.ndim

    if is_magnitude_provided:
        quality_map = phase_data_for_quality.clone().float()
    else: # Use inverse squared phase gradient magnitude
        gradient_sq_sum = torch.zeros_like(phase_data_for_quality, dtype=torch.float32, device=device)
        
        for i in range(ndim):
            current_voxel_size = voxel_size[i] if len(voxel_size) == ndim else 1.0
            
            # Difference with next neighbor
            diff_next = phase_data_for_quality - torch.roll(phase_data_for_quality, shifts=-1, dims=i)
            diff_next_wrapped = diff_next - 2 * torch.pi * torch.round(diff_next / (2 * torch.pi))
            grad_next_sq = (diff_next_wrapped / current_voxel_size)**2
            
            # Difference with previous neighbor
            diff_prev = phase_data_for_quality - torch.roll(phase_data_for_quality, shifts=1, dims=i)
            diff_prev_wrapped = diff_prev - 2 * torch.pi * torch.round(diff_prev / (2 * torch.pi))
            grad_prev_sq = (diff_prev_wrapped / current_voxel_size)**2

            # For boundary voxels, one of these differences is across the boundary.
            # Summing them up effectively gives a central-difference like behavior for interior,
            # and forward/backward difference at boundaries, but it's doubled.
            # A common way for quality is sum of squared differences to *all* neighbors.
            # Here, let's sum up contributions from both directions for each dimension.
            # To avoid double counting edges for the sum, we can average or just sum.
            # Summing captures more local variation.
            
            # At boundaries, rolled diff wraps around. We want to zero out this wrapped gradient.
            # Let's set boundary gradient contribution to be based on one-sided diff only.
            # Or, simpler: use the sum, and it will be higher at boundaries if phase wraps sharply.
            
            # Create a boundary mask for the current dimension at the "end"
            boundary_mask_dim_end = torch.zeros_like(phase_data_for_quality, dtype=torch.bool)
            slicer = [slice(None)] * ndim; slicer[i] = -1
            boundary_mask_dim_end[tuple(slicer)] = True
            
            # Create a boundary mask for the current dimension at the "start"
            boundary_mask_dim_start = torch.zeros_like(phase_data_for_quality, dtype=torch.bool)
            slicer_start = [slice(None)] * ndim; slicer_start[i] = 0
            boundary_mask_dim_start[tuple(slicer_start)] = True

            # Accumulate, effectively using |grad_fwd|^2 + |grad_bwd|^2 for each point's contribution to its neighbors' variance
            # This is related to the sum of squared differences (SSD) quality metrics.
            gradient_sq_sum += grad_next_sq 
            gradient_sq_sum += grad_prev_sq # Doing this sums up all connections twice, essentially.
                                            # So, divide by 2 later or use one direction, e.g. only grad_next_sq
                                            # and ensure it's symmetric by also adding grad_prev_sq.
                                            # The current sum is fine, higher values just mean lower quality.

        # The above loop calculates sum of squared differences to neighbors in each dim (forward and backward)
        # For a voxel v, GSS_v = sum_{n in N(v)} (phi_v - phi_n)^2 / voxel_size^2
        # The loop calculates sum_{d in dims} [ ( (phi_v - phi_{v+d}) / vs_d )^2 + ( (phi_v - phi_{v-d}) / vs_d )^2 ]
        # This is a valid measure of local phase variation.

        quality_map = 1.0 / (1.0 + gradient_sq_sum) # Add 1 to avoid division by zero
        # Areas with zero gradient (flat phase) will have quality 1.
        # Areas with high gradient will have quality close to 0.

    # Normalize quality map to [0, 1]
    min_q = quality_map.min()
    max_q = quality_map.max()
    if max_q > min_q:
        quality_map = (quality_map - min_q) / (max_q - min_q)
    else: # Uniform quality
        quality_map.fill_(0.5) # Or 1.0 if all truly identical

    if mask is not None:
        quality_map.masked_fill_(~mask, 0) # Zero quality outside mask

    return quality_map.to(device)


def _find_highest_quality_seed_voxel_romeo(
    quality_map: torch.Tensor,
    mask: Optional[torch.Tensor],
    quality_threshold: float
) -> Optional[Tuple[int, ...]]:
    """
    Finds the coordinates of the highest quality voxel above a threshold.
    """
    temp_quality_map = quality_map.clone()
    if mask is not None:
        temp_quality_map.masked_fill_(~mask, -1) # Ensure masked out voxels are not chosen

    # Ensure only voxels above threshold are considered
    temp_quality_map.masked_fill_(temp_quality_map < quality_threshold, -1)

    if temp_quality_map.max() < 0: # No voxel meets criteria
        return None

    # Find the index of the maximum quality voxel
    # flat_idx = torch.argmax(temp_quality_map) # Gets flattened index
    # seed_voxel_indices = np.unravel_index(flat_idx.cpu().numpy(), temp_quality_map.shape)
    # Using where to get all max locations, then pick first one.
    max_qual_val = temp_quality_map.max() # This can be -1 if all voxels are below threshold or masked
    if max_qual_val < 0: # Handles case where no voxel is valid
        return None

    candidates = torch.where(temp_quality_map == max_qual_val)
    
    # Check if any candidates were found (e.g. if temp_quality_map was all -1)
    if not candidates[0].numel():
         # This case should ideally be caught by max_qual_val < 0, but as a safeguard:
        return None
    
    # Return the first candidate as a tuple of ints
    seed_voxel_indices = tuple(c[0].item() for c in candidates)
    return seed_voxel_indices


def unwrap_phase_romeo(
    wrapped_phase: torch.Tensor,
    magnitude: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    voxel_size: Tuple[float, ...] = (1.0, 1.0, 1.0), # (Vz, Vy, Vx) or (Vy, Vx)
    quality_threshold: float = 0.1,
    max_iterations_rg: int = 10000,
    neighbor_connectivity: int = 1
) -> torch.Tensor:
    """
    Performs phase unwrapping using a native PyTorch region-growing algorithm
    inspired by ROMEO principles.

    This implementation uses a quality map (derived from magnitude or phase gradients)
    to guide the unwrapping process, starting from a high-quality seed voxel.

    Args:
        wrapped_phase (torch.Tensor): Input wrapped phase image (in radians).
            Shape: (D, H, W) for 3D or (H, W) for 2D.
        magnitude (torch.Tensor, optional): Magnitude image. If provided, it's used
            to compute the quality map (higher magnitude = higher quality).
            Should have the same spatial dimensions as `wrapped_phase`.
        mask (torch.Tensor, optional): Boolean tensor indicating the region to unwrap
            (True values are unwrapped). If None, the whole volume is processed.
            Shape should match `wrapped_phase`.
        voxel_size (Tuple[float, ...], optional): Voxel dimensions (e.g., (Vz, Vy, Vx) for 3D
            or (Vy, Vx) for 2D). Used for calculating phase gradients if magnitude is not provided.
            Defaults to (1.0, 1.0, 1.0) or (1.0, 1.0) based on phase dim.
        quality_threshold (float, optional): Minimum quality for a voxel to be
            processed during region growing and for seed selection. Range [0,1].
            Defaults to 0.1.
        max_iterations_rg (int, optional): Maximum number of voxels to process in
            the region growing loop to prevent infinite loops. Defaults to 10000.
        neighbor_connectivity (int, optional): Defines neighborhood for region growing.
            - 1: Face neighbors (6 in 3D, 4 in 2D).
            - 2: Face and edge neighbors (18 in 3D), or face and corner (8 in 2D).
            - 3: Face, edge, and corner neighbors (26 in 3D). (Not applicable for 2D beyond 8-conn).
            Defaults to 1.

    Returns:
        torch.Tensor: Unwrapped phase image.
    """
    if not isinstance(wrapped_phase, torch.Tensor):
        raise TypeError("Input 'wrapped_phase' must be a PyTorch Tensor.")
    
    device = wrapped_phase.device
    shape = wrapped_phase.shape
    ndim = wrapped_phase.ndim

    if not (ndim == 2 or ndim == 3):
        raise ValueError(f"wrapped_phase must be 2D or 3D, got {ndim}D.")

    # Adjust voxel_size tuple length to match ndim if default is used and phase is 2D
    if len(voxel_size) == 3 and ndim == 2:
        voxel_size_adjusted = (voxel_size[1], voxel_size[2]) # Assume (Vy, Vx) from (Vz,Vy,Vx)
    elif len(voxel_size) != ndim:
        raise ValueError(f"voxel_size length {len(voxel_size)} does not match phase ndim {ndim}.")
    else:
        voxel_size_adjusted = voxel_size

    unwrapped_phase = torch.zeros_like(wrapped_phase, dtype=torch.float32, device=device)
    visited = torch.zeros_like(wrapped_phase, dtype=torch.bool, device=device)

    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            raise TypeError("Input 'mask' must be a PyTorch Tensor.")
        if mask.shape != shape:
            raise ValueError("Mask shape must match wrapped_phase shape.")
        mask = mask.to(device=device, dtype=torch.bool)
        # Apply mask to visited: voxels outside mask are considered "visited" so not processed
        visited[~mask] = True 
    else:
        # If no mask, create a full mask for consistency in _find_highest_quality_seed_voxel_romeo
        # and _get_neighbors_romeo if they expect a mask.
        # However, it's better to handle None mask in helpers. For now, let's assume they do.
        pass


    # 1. Compute Quality Map
    phase_data_for_quality = magnitude if magnitude is not None else wrapped_phase
    is_magnitude_provided = magnitude is not None
    if magnitude is not None:
        if not isinstance(magnitude, torch.Tensor):
            raise TypeError("Input 'magnitude' must be a PyTorch Tensor.")
        if magnitude.shape != shape:
            raise ValueError("Magnitude shape must match wrapped_phase shape.")
        phase_data_for_quality = magnitude.to(device=device)


    quality_map = _compute_romeo_quality_map_spatial(
        phase_data_for_quality.to(device=device),
        mask=mask,
        voxel_size=voxel_size_adjusted,
        is_magnitude_provided=is_magnitude_provided
    )

    # 2. Find Highest Quality Seed Voxel
    seed_voxel = _find_highest_quality_seed_voxel_romeo(quality_map, mask, quality_threshold)

    if seed_voxel is None:
        print("Warning: No seed voxel found above quality threshold. Returning wrapped phase.")
        return wrapped_phase.clone() # Or raise error

    # 3. Region Growing Loop
    queue: Deque[Tuple[int, ...]] = collections.deque()
    
    queue.append(seed_voxel)
    visited[seed_voxel] = True
    unwrapped_phase[seed_voxel] = wrapped_phase[seed_voxel] # Seed voxel is reference

    iterations = 0
    while queue and iterations < max_iterations_rg:
        current_voxel_idx = queue.popleft()
        iterations += 1

        neighbors = _get_neighbors_romeo(current_voxel_idx, shape, mask, neighbor_connectivity)

        for neighbor_idx in neighbors:
            if not visited[neighbor_idx].item() and quality_map[neighbor_idx].item() >= quality_threshold:
                visited[neighbor_idx] = True
                
                # Calculate phase difference and unwrap
                # diff = wrapped_phase[neighbor_idx] - wrapped_phase[current_voxel_idx] # Both are scalars
                # unwrapped_phase[neighbor_idx] = unwrapped_phase[current_voxel_idx] + \
                #                                 (diff - 2 * torch.pi * torch.round(diff / (2 * torch.pi)))
                
                # Ensure scalar tensor values are used for arithmetic if not already scalar
                wp_neighbor = wrapped_phase[neighbor_idx].item()
                wp_current = wrapped_phase[current_voxel_idx].item()
                up_current = unwrapped_phase[current_voxel_idx].item()

                diff = wp_neighbor - wp_current
                unwrapped_phase[neighbor_idx] = up_current + (diff - 2 * torch.pi * round(diff / (2 * torch.pi)))

                queue.append(neighbor_idx)
    
    if iterations >= max_iterations_rg:
        print(f"Warning: Max iterations ({max_iterations_rg}) reached during region growing.")

    # Apply mask to the final unwrapped phase: set unvisited/unmasked regions to 0 or original phase
    if mask is not None:
        unwrapped_phase.masked_fill_(~mask, 0) # Or original wrapped_phase[~mask]
    # Also, regions that were part of mask but not visited (e.g. disconnected components)
    # will remain 0 from initialization. This is often desired.
    # If they should be original phase:
    # unwrapped_phase[mask & ~visited] = wrapped_phase[mask & ~visited]


    return unwrapped_phase
