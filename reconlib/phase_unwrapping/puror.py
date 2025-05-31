# reconlib/phase_unwrapping/puror.py
"""
Native PyTorch implementation of a Voronoi-seeded region-growing phase
unwrapping algorithm, inspired by PUROR principles.
"""

import torch
import heapq
import numpy as np # For np.prod if needed, and type hints
from typing import Optional, Tuple, List, Set # PriorityQueue not directly used, heapq is

# Helper: Get Neighbors (similar to ROMEO's, can be shared or duplicated/adapted)
def _get_neighbors_puror(
    voxel_idx: Tuple[int, ...],
    shape: Tuple[int, ...],
    mask: Optional[torch.Tensor], # Voxel is already confirmed to be in mask if applicable
    connectivity: int
) -> List[Tuple[int, ...]]:
    """
    Gets valid neighbors of a voxel given connectivity.
    Assumes voxel_idx itself is valid and within mask if provided.
    """
    neighbors: List[Tuple[int, ...]] = []
    ndim = len(shape)
    
    if ndim == 2: # (H, W)
        h, w = voxel_idx
        all_offsets = []
        if connectivity >= 1: # 4-connectivity
            all_offsets.extend([(0, 1), (0, -1), (1, 0), (-1, 0)])
        if connectivity >= 2: # 8-connectivity (includes diagonals)
             all_offsets.extend([(1,1), (1,-1), (-1,1), (-1,-1)])
        # Max connectivity for 2D is typically 2 (8-way)
    elif ndim == 3: # (D, H, W)
        d, h, w = voxel_idx
        all_offsets = []
        if connectivity >= 1: # 6-connectivity (faces)
            all_offsets.extend([(0,0,1), (0,0,-1), (0,1,0), (0,-1,0), (1,0,0), (-1,0,0)])
        if connectivity >= 2: # 18-connectivity (faces + edges)
            all_offsets.extend([
                (1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0),
                (1,0,1), (1,0,-1), (-1,0,1), (-1,0,-1),
                (0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1),
            ])
        if connectivity >= 3: # 26-connectivity (faces + edges + corners)
            all_offsets.extend([
                (1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1),
                (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1),
            ])
    else:
        raise ValueError(f"Unsupported number of dimensions: {ndim}. Expected 2 or 3.")

    unique_offsets = list(set(all_offsets)) # Remove duplicates if connectivity levels overlap

    for offset in unique_offsets:
        neighbor_coords_list = [voxel_idx[i] + offset[i] for i in range(ndim)]
        
        valid = True
        for i_dim in range(ndim):
            if not (0 <= neighbor_coords_list[i_dim] < shape[i_dim]):
                valid = False
                break
        if not valid:
            continue

        neighbor_coords_tuple = tuple(neighbor_coords_list)
        if mask is not None and not mask[neighbor_coords_tuple].item():
            continue
        
        neighbors.append(neighbor_coords_tuple)
        
    return neighbors

# Helper: Compute Quality Map (spatial phase gradient based)
def _compute_puror_quality_map(
    phase_data: torch.Tensor,
    mask: Optional[torch.Tensor],
    voxel_size: Tuple[float, ...]
) -> torch.Tensor:
    """
    Computes a spatial quality map based on inverse phase gradient magnitude.
    Higher quality means lower phase gradient.
    """
    device = phase_data.device
    ndim = phase_data.ndim
    
    gradient_sq_sum = torch.zeros_like(phase_data, dtype=torch.float32, device=device)
    
    for i in range(ndim):
        current_voxel_size = voxel_size[i] if len(voxel_size) == ndim else 1.0
        
        # Difference with next neighbor (forward difference style for one side)
        diff_next = phase_data - torch.roll(phase_data, shifts=-1, dims=i)
        diff_next_wrapped = diff_next - 2 * torch.pi * torch.round(diff_next / (2 * torch.pi))
        grad_next_sq = (diff_next_wrapped / current_voxel_size)**2
        
        # Difference with previous neighbor (backward difference style for other side)
        diff_prev = phase_data - torch.roll(phase_data, shifts=1, dims=i)
        diff_prev_wrapped = diff_prev - 2 * torch.pi * torch.round(diff_prev / (2 * torch.pi))
        grad_prev_sq = (diff_prev_wrapped / current_voxel_size)**2

        # Sum contributions. This effectively weights central differences higher.
        # Or consider it as sum of squared differences to immediate neighbors along each axis.
        gradient_sq_sum += grad_next_sq
        gradient_sq_sum += grad_prev_sq 

    # Quality is higher where gradient is lower
    quality_map = 1.0 / (1.0 + gradient_sq_sum) # Add 1 to avoid division by zero

    # Normalize quality map to [0, 1]
    min_q = quality_map.min()
    max_q = quality_map.max()
    if max_q > min_q:
        quality_map = (quality_map - min_q) / (max_q - min_q)
    else: # Uniform quality
        quality_map.fill_(0.5)

    if mask is not None:
        quality_map.masked_fill_(~mask, 0) # Zero quality outside mask

    return quality_map

# Helper: Select Seeds
def _select_voronoi_seeds_puror(
    quality_map: torch.Tensor,
    mask: Optional[torch.Tensor],
    quality_threshold: float,
    max_seeds: Optional[int] = None # Optional: limit number of seeds
) -> List[Tuple[int, ...]]:
    """
    Selects seed voxels with quality > quality_threshold.
    Simplification: Returns all qualifying seeds or a random subset if too many.
    """
    seed_candidates = torch.where(quality_map > quality_threshold)
    
    if not seed_candidates[0].numel():
        return []

    seeds = list(zip(*(c.tolist() for c in seed_candidates))) # List of (d,h,w) or (h,w) tuples

    if max_seeds is not None and len(seeds) > max_seeds:
        # Simple random subsampling if too many seeds
        indices = np.random.choice(len(seeds), size=max_seeds, replace=False)
        seeds = [seeds[i] for i in indices]
        
    return seeds

# Helper: Compute Residual (placeholder/simple version)
def _compute_residual_puror(
    wrapped_phase: torch.Tensor,
    unwrapped_phase: torch.Tensor,
    mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    Computes the residual between the wrapped input phase and the wrapped output phase.
    Ideally, this should be close to zero or multiples of 2*pi.
    A simple metric could be sum of (wrap(original - unwrapped))^2.
    """
    residual = wrapped_phase - unwrapped_phase
    # Wrap residual to [-pi, pi]
    residual_wrapped = (residual + torch.pi) % (2 * torch.pi) - torch.pi
    if mask is not None:
        return residual_wrapped.masked_fill(~mask, 0)
    return residual_wrapped

# Stub functions for deferred parts
def _merge_voronoi_cells_puror(unwrapped_phase: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    print("Warning: _merge_voronoi_cells_puror is a stub and does nothing.")
    # raise NotImplementedError("_merge_voronoi_cells_puror is not implemented.")
    return unwrapped_phase

def _optimize_paths_puror(unwrapped_phase: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    print("Warning: _optimize_paths_puror is a stub and does nothing.")
    # raise NotImplementedError("_optimize_paths_puror is not implemented.")
    return unwrapped_phase


# Main PUROR function
def unwrap_phase_puror(
    phase_data: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    voxel_size: Tuple[float,...] = (1.0,1.0,1.0), # (Vz, Vy, Vx) or (Vy, Vx)
    quality_threshold: float = 0.1, # Min quality for seed and growth
    max_iterations_rg: int = 1000000, # Max elements to process in total across all seeds
    tolerance: float = 1e-6, # Not directly used in this simplified RG, more for iterative solvers
    neighbor_connectivity: int = 1,
    max_seeds_to_process: Optional[int] = None # Limit number of seeds for performance
) -> torch.Tensor:
    """
    Performs phase unwrapping using a Voronoi-seeded region-growing algorithm,
    inspired by PUROR (Phase Unwrapping using Recursive Orthogonal Referring).

    This implementation simplifies full PUROR:
    1. Computes a quality map from phase gradients.
    2. Selects multiple seed points above a quality threshold.
    3. Performs region growing from each seed (if not already visited) using a
       priority queue ordered by quality.
    4. Full Voronoi tessellation, cell merging, and path optimization steps are currently stubs.

    Args:
        phase_data (torch.Tensor): Wrapped phase data (in radians). (D,H,W) or (H,W).
        mask (torch.Tensor, optional): Boolean tensor. True values are unwrapped.
        voxel_size (Tuple[float,...], optional): Voxel dimensions.
            Adjust length to match phase_data.ndim (e.g., (vz,vy,vx) or (vy,vx)).
        quality_threshold (float, optional): Min quality for processing. Defaults to 0.1.
        max_iterations_rg (int, optional): Max total voxels processed across all region growths.
        tolerance (float, optional): Tolerance for convergence (currently unused).
        neighbor_connectivity (int, optional): 1 for face, 2 for edge, 3 for corner.
        max_seeds_to_process (Optional[int], optional): Limit the number of initial seeds.

    Returns:
        torch.Tensor: Unwrapped phase image.
    """
    if not isinstance(phase_data, torch.Tensor):
        raise TypeError("Input 'phase_data' must be a PyTorch Tensor.")
    
    device = phase_data.device
    shape = phase_data.shape
    ndim = phase_data.ndim

    if not (ndim == 2 or ndim == 3):
        raise ValueError(f"phase_data must be 2D or 3D, got {ndim}D.")

    # Adjust voxel_size tuple length to match ndim
    if len(voxel_size) != ndim:
        if ndim == 2 and len(voxel_size) == 3: # Common case: (1,vy,vx) for (vy,vx) data
            voxel_size_adjusted = (voxel_size[1], voxel_size[2])
            print(f"Adjusted voxel_size to {voxel_size_adjusted} for 2D input from {voxel_size}")
        elif ndim == 3 and len(voxel_size) == 1: # Isotropic from single value
             voxel_size_adjusted = tuple(voxel_size[0] for _ in range(ndim))
        else: # Default if mismatch is problematic
            print(f"Warning: voxel_size length {len(voxel_size)} mismatch with phase ndim {ndim}. Using isotropic (1.0,...).")
            voxel_size_adjusted = tuple(1.0 for _ in range(ndim))
    else:
        voxel_size_adjusted = voxel_size

    unwrapped_phase = torch.zeros_like(phase_data, dtype=torch.float32, device=device)
    visited = torch.zeros_like(phase_data, dtype=torch.bool, device=device)

    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            raise TypeError("Input 'mask' must be a PyTorch Tensor.")
        if mask.shape != shape:
            raise ValueError("Mask shape must match phase_data shape.")
        mask = mask.to(device=device, dtype=torch.bool)
        visited[~mask] = True # Masked out regions are considered "visited"
    
    # 1. Compute Quality Map
    quality_map = _compute_puror_quality_map(phase_data, mask, voxel_size_adjusted)

    # 2. Select Seeds
    # (Simplified: not full Voronoi partitioning, but multiple region growths)
    initial_seeds = _select_voronoi_seeds_puror(quality_map, mask, quality_threshold, max_seeds=max_seeds_to_process)

    if not initial_seeds:
        print("Warning: No seed voxels found above quality threshold. Returning wrapped phase.")
        return phase_data.clone()

    total_processed_count = 0

    for seed_idx in initial_seeds:
        if visited[seed_idx].item(): # Already processed by a previous seed's growth
            continue

        # Each seed initiates its own region growing using a priority queue (max-heap for quality)
        # heapq implements a min-heap, so store negative quality.
        pq: List[Tuple[float, Tuple[int, ...]]] = [] # (priority, voxel_idx_tuple)
        
        unwrapped_phase[seed_idx] = phase_data[seed_idx] # Seed is reference
        visited[seed_idx] = True
        # Add seed to its own PQ: (-quality, (coords))
        heapq.heappush(pq, (-quality_map[seed_idx].item(), seed_idx))
        total_processed_count +=1

        current_seed_processed_count = 0
        while pq and total_processed_count < max_iterations_rg:
            neg_q, current_voxel_idx = heapq.heappop(pq)
            current_quality = -neg_q
            current_seed_processed_count +=1

            neighbors = _get_neighbors_puror(current_voxel_idx, shape, mask, neighbor_connectivity)

            for neighbor_idx in neighbors:
                if not visited[neighbor_idx].item() and quality_map[neighbor_idx].item() >= quality_threshold:
                    visited[neighbor_idx] = True
                    total_processed_count +=1
                    
                    wp_neighbor = phase_data[neighbor_idx].item()
                    wp_current = phase_data[current_voxel_idx].item() # This should be unwrapped_phase for robustness if seeds are far
                                                                    # but for direct neighbors, wrapped phase diff is fine.
                                                                    # Let's use unwrapped_phase of current voxel.
                    up_current = unwrapped_phase[current_voxel_idx].item()

                    diff = wp_neighbor - wp_current # Difference in wrapped phase values
                    # Standard unwrapping formula: new_unwrapped = prev_unwrapped + wrap(diff)
                    # where wrap(d) = d - 2*pi*round(d / 2*pi)
                    unwrapped_phase[neighbor_idx] = up_current + (diff - 2 * torch.pi * round(diff / (2 * torch.pi)))
                    
                    heapq.heappush(pq, (-quality_map[neighbor_idx].item(), neighbor_idx))
                    if total_processed_count >= max_iterations_rg: break
            if total_processed_count >= max_iterations_rg: break


    if total_processed_count >= max_iterations_rg:
        print(f"Warning: Max iterations_rg ({max_iterations_rg}) reached during region growing.")
    
    # Placeholder calls for currently unimplemented steps
    unwrapped_phase = _merge_voronoi_cells_puror(unwrapped_phase) # Does nothing
    unwrapped_phase = _optimize_paths_puror(unwrapped_phase)       # Does nothing

    # Final residual computation (optional, for info)
    # residual_map = _compute_residual_puror(phase_data, unwrapped_phase, mask)
    # print(f"Max absolute residual: {residual_map.abs().max().item()}")

    if mask is not None:
        unwrapped_phase.masked_fill_(~mask, 0) # Zero out regions outside mask

    return unwrapped_phase
