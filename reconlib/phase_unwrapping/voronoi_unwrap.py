import torch
import numpy as np
import heapq # For priority queue in region growing
from typing import Optional, Tuple, List, Dict, Set, Union # Expanded typing

# Attempt to import compute_voronoi_tessellation
try:
    from reconlib.voronoi.voronoi_tessellation import compute_voronoi_tessellation
    RECONLIB_VORONOI_TESSELLATION_AVAILABLE = True
except ImportError:
    RECONLIB_VORONOI_TESSELLATION_AVAILABLE = False
    # Define a placeholder if the import fails, so the module can still be loaded for inspection
    def compute_voronoi_tessellation(
        shape: tuple[int, ...],
        seeds: torch.Tensor,
        voxel_size: Union[tuple[float, ...], torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        print("Warning: reconlib.voronoi.voronoi_tessellation.compute_voronoi_tessellation not found. Using placeholder.")
        # Fallback: return a map where all valid voxels are assigned to the first seed (or -1 if no seeds)
        output_tessellation = torch.full(shape, -1, dtype=torch.long, device=seeds.device if seeds.numel() > 0 else torch.device('cpu'))
        if seeds.shape[0] > 0:
            if mask is not None:
                output_tessellation[mask] = 0
            else:
                output_tessellation.fill_(0)
        return output_tessellation

# Helper functions specific to this module, prefixed with _vu_ (Voronoi Unwrap)

def _vu_wrap_to_pi(phase_diff: torch.Tensor) -> torch.Tensor:
    """Wraps phase values to the interval [-pi, pi)."""
    return (phase_diff + torch.pi) % (2 * torch.pi) - torch.pi

def _vu_is_valid_voxel(voxel_coord: Tuple[int, ...], shape: Tuple[int, ...], mask: Optional[torch.Tensor]) -> bool:
    """Checks if a voxel is within bounds and, if a mask is provided, within the mask."""
    for i, coord_val in enumerate(voxel_coord):
        if not (0 <= coord_val < shape[i]):
            return False
    if mask is not None and not mask[voxel_coord].item():
        return False
    return True

def _vu_get_neighbors(
    voxel_idx: Tuple[int, ...],
    shape: Tuple[int, ...],
    connectivity: int,
    mask: Optional[torch.Tensor] # Mask is passed to _vu_is_valid_voxel
) -> List[Tuple[int, ...]]:
    """Gets valid neighbors of a voxel given connectivity."""
    neighbors: List[Tuple[int, ...]] = []
    ndim = len(shape)

    offsets_dim_map = {
        1: [(1,), (-1,)], # For 1D if ever needed
        2: [ # 2D offsets
            [(0, 1), (0, -1), (1, 0), (-1, 0)], # Conn 1 (4-conn)
            [(1,1), (1,-1), (-1,1), (-1,-1)]    # Conn 2 Diagonals (for 8-conn)
        ],
        3: [ # 3D offsets
            [(0,0,1), (0,0,-1), (0,1,0), (0,-1,0), (1,0,0), (-1,0,0)], # Conn 1 (6-conn, faces)
            [ # Conn 2 Edges (for 18-conn)
                (1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0), (1,0,1), (1,0,-1),
                (-1,0,1), (-1,0,-1), (0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1)
            ],
            [ # Conn 3 Corners (for 26-conn)
                (1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1),
                (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1)
            ]
        ]
    }

    if ndim not in offsets_dim_map:
        raise ValueError(f"Unsupported number of dimensions: {ndim}. Expected 2 or 3.")

    current_offsets: List[Tuple[int,...]] = []
    for c_level in range(1, connectivity + 1):
        if c_level -1 < len(offsets_dim_map[ndim]):
            current_offsets.extend(offsets_dim_map[ndim][c_level-1])

    # Ensure unique offsets if connectivity levels overlap (though designed not to here)
    unique_offsets = list(set(current_offsets))


    for offset in unique_offsets:
        neighbor_coords_list = [voxel_idx[i] + offset[i] for i in range(ndim)]
        neighbor_coords_tuple = tuple(neighbor_coords_list)

        if _vu_is_valid_voxel(neighbor_coords_tuple, shape, mask):
            neighbors.append(neighbor_coords_tuple)

    return neighbors

def _vu_calculate_wrapped_gradients(phase: torch.Tensor, voxel_size: torch.Tensor) -> torch.Tensor:
    """Calculates wrapped phase gradients. Output shape (num_dims, *phase_shape)."""
    ndim = phase.ndim
    gradients_sq = torch.zeros_like(phase, dtype=phase.dtype)

    for i in range(ndim):
        # Forward difference
        diff_fwd = _vu_wrap_to_pi(torch.roll(phase, shifts=-1, dims=i) - phase) / voxel_size[i]
        # Backward difference
        diff_bwd = _vu_wrap_to_pi(phase - torch.roll(phase, shifts=1, dims=i)) / voxel_size[i]

        # Sum of squares of fwd and bwd gradients (measures local variability)
        gradients_sq += diff_fwd**2 + diff_bwd**2

    return gradients_sq # This is sum of squares, not tuple of grads

def _vu_compute_quality_map(phase: torch.Tensor, mask: Optional[torch.Tensor], voxel_size_tensor: torch.Tensor) -> torch.Tensor:
    """Computes quality map as inverse of sum of squared wrapped gradients."""
    sum_sq_gradients = _vu_calculate_wrapped_gradients(phase, voxel_size_tensor)
    quality_map = 1.0 / (1.0 + sum_sq_gradients) # Higher quality = smaller gradient

    # Normalize quality map
    min_q = quality_map.min()
    max_q = quality_map.max()
    if max_q > min_q:
        quality_map = (quality_map - min_q) / (max_q - min_q)
    else:
        quality_map.fill_(0.5)

    if mask is not None:
        quality_map.masked_fill_(~mask, 0.0)
    return quality_map

def _vu_euclidean_distance_sq_scaled(coord1_tuple: Tuple[int, ...], coord2_tuple: Tuple[int, ...], voxel_size_tensor: torch.Tensor) -> float:
    """Computes squared Euclidean distance scaled by voxel_size."""
    dist_sq = 0.0
    for i in range(len(coord1_tuple)):
        dist_sq += ((coord1_tuple[i] - coord2_tuple[i]) * voxel_size_tensor[i].item())**2
    return dist_sq

def _vu_select_voronoi_seeds(
    quality_map: torch.Tensor,
    mask: Optional[torch.Tensor],
    quality_threshold: float,
    min_seed_dist_sq: float, # Use squared distance to avoid sqrt
    voxel_size_tensor: torch.Tensor
) -> List[Tuple[int, ...]]:
    """Selects Voronoi seeds based on quality and minimum distance."""
    candidate_coords_torch = torch.where(quality_map >= quality_threshold)

    # Create a list of (quality, coord_tuple) for sorting
    candidates_with_quality = []
    for i in range(candidate_coords_torch[0].shape[0]):
        coord_tuple = tuple(c[i].item() for c in candidate_coords_torch)
        if mask is None or mask[coord_tuple].item(): # Double check mask
            candidates_with_quality.append((quality_map[coord_tuple].item(), coord_tuple))

    # Sort by quality in descending order
    candidates_with_quality.sort(key=lambda x: x[0], reverse=True)

    selected_seeds_coords: List[Tuple[int, ...]] = []
    for _, coord_tuple in candidates_with_quality:
        is_far_enough = True
        for existing_seed_coord in selected_seeds_coords:
            dist_sq = _vu_euclidean_distance_sq_scaled(coord_tuple, existing_seed_coord, voxel_size_tensor)
            if dist_sq < min_seed_dist_sq:
                is_far_enough = False
                break
        if is_far_enough:
            selected_seeds_coords.append(coord_tuple)

    return selected_seeds_coords

def _vu_merge_voronoi_cells_and_optimize_paths(
    unwrapped_phase: torch.Tensor,
    wrapped_phase: torch.Tensor,
    cell_id_map: torch.Tensor,
    quality_map: torch.Tensor,
    mask: Optional[torch.Tensor],
    voxel_size: torch.Tensor,
    tolerance: float,
    max_iterations: int
) -> torch.Tensor:
    """Placeholder for merging Voronoi cells and optimizing paths."""
    print("Warning: _vu_merge_voronoi_cells_and_optimize_paths is a placeholder and does not perform any merging or optimization.")
    # In a full implementation, this would involve analyzing boundaries between
    # different cell IDs in cell_id_map, calculating inconsistencies, and
    # potentially adding multiples of 2*pi to entire cells to resolve them.
    return unwrapped_phase # Returns the input phase unchanged for now


def unwrap_phase_voronoi_region_growing(
    phase_data: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    voxel_size: Union[Tuple[float, ...], List[float]] = (1.0, 1.0, 1.0),
    quality_threshold: float = 0.1,  # For seed selection and region growing
    max_iterations_rg: int = -1,      # Max voxels processed in region growing; -1 for no limit based on numel
    tolerance: float = 1e-6,         # For future optimization steps
    neighbor_connectivity: int = 1,  # For GetNeighbors
    min_seed_distance: float = 5.0   # Physical distance
) -> torch.Tensor:
    """
    Performs phase unwrapping using a Voronoi-seeded region-growing algorithm.
    Implementation based on user-provided pseudocode.
    """
    if not isinstance(phase_data, torch.Tensor):
        raise TypeError("Input 'phase_data' must be a PyTorch Tensor.")

    device = phase_data.device
    shape = phase_data.shape
    ndim = phase_data.ndim

    if not (ndim == 2 or ndim == 3):
        raise ValueError(f"phase_data must be 2D or 3D, got {ndim}D.")

    voxel_size_tensor = torch.tensor(list(voxel_size), dtype=torch.float32, device=device)
    if voxel_size_tensor.shape[0] != ndim:
        raise ValueError(f"voxel_size length ({len(voxel_size)}) must match phase_data ndim ({ndim}).")

    # --- Initialization ---
    unwrapped_phase = torch.zeros_like(phase_data, dtype=torch.float32, device=device)
    visited = torch.zeros_like(phase_data, dtype=torch.bool, device=device)

    if mask is not None:
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise TypeError("Mask must be a boolean PyTorch Tensor.")
        if mask.shape != shape:
            raise ValueError("Mask shape must match phase_data shape.")
        mask = mask.to(device)
        visited[~mask] = True # Masked out regions are considered "visited"

    # --- Quality Map ---
    quality_map = _vu_compute_quality_map(phase_data, mask, voxel_size_tensor)

    # --- Seed Selection ---
    # min_seed_distance is physical, helper needs squared physical distance
    min_seed_dist_sq = min_seed_distance**2
    voronoi_seeds_voxel_coords = _vu_select_voronoi_seeds(quality_map, mask, quality_threshold, min_seed_dist_sq, voxel_size_tensor)

    if not voronoi_seeds_voxel_coords:
        print("Warning: No seed voxels found. Returning original wrapped phase.")
        if mask is not None:
            return phase_data.where(mask, torch.tensor(0.0, device=device))
        return phase_data.clone()

    # --- Voronoi Tessellation (Option A: Nearest Seed Assignment) ---
    # Convert voxel seed coordinates to physical coordinates for compute_voronoi_tessellation
    seeds_physical_coords_list = []
    for seed_vc in voronoi_seeds_voxel_coords:
        phys_coord = (torch.tensor(seed_vc, dtype=torch.float32, device=device) + 0.5) * voxel_size_tensor
        seeds_physical_coords_list.append(phys_coord)

    seeds_physical_tensor = torch.stack(seeds_physical_coords_list)

    if RECONLIB_VORONOI_TESSELLATION_AVAILABLE:
        cell_id_map = compute_voronoi_tessellation(shape, seeds_physical_tensor, voxel_size_tensor, mask)
    else: # Fallback if the main function isn't available (should have been caught by import)
        # Simplified nearest seed for CellIDMap if external not found
        print("Warning: Using simplified internal nearest seed for CellIDMap as reconlib.voronoi.voronoi_tessellation.compute_voronoi_tessellation was not available.")
        cell_id_map = torch.full(shape, -1, dtype=torch.long, device=device)
        # This part would need a loop over all voxels to find nearest seed if external func fails.
        # For now, this means Merge step won't have meaningful Cell IDs if the import fails.
        # The region growing below doesn't strictly need CellIDMap for its logic, but the Merge step does.
        # Let's assume for now the import works, or the Merge step handles CellIDMap=-1.

    # Initialize unwrapping at seed points
    for seed_coord_tuple in voronoi_seeds_voxel_coords:
        unwrapped_phase[seed_coord_tuple] = phase_data[seed_coord_tuple]
        visited[seed_coord_tuple] = True

    # --- Region Growing ---
    pq: List[Tuple[float, Tuple[int, ...]]] = [] # (neg_quality, voxel_idx_tuple)
    for seed_coord_tuple in voronoi_seeds_voxel_coords:
        heapq.heappush(pq, (-quality_map[seed_coord_tuple].item(), seed_coord_tuple))

    processed_count_rg = 0
    if max_iterations_rg == -1: # If -1, set to total number of voxels in mask or image
        max_iterations_rg = mask.sum().item() if mask is not None else phase_data.numel()

    while pq and processed_count_rg < max_iterations_rg:
        neg_q, current_voxel_idx_tuple = heapq.heappop(pq)
        # current_quality = -neg_q # Not strictly needed for logic, only for pushing

        processed_count_rg += 1

        # Get neighbors using the local helper
        neighbors = _vu_get_neighbors(current_voxel_idx_tuple, shape, neighbor_connectivity, mask)

        for neighbor_idx_tuple in neighbors:
            if not visited[neighbor_idx_tuple].item() and quality_map[neighbor_idx_tuple].item() >= quality_threshold:
                visited[neighbor_idx_tuple] = True

                wp_neighbor = phase_data[neighbor_idx_tuple].item()
                up_current = unwrapped_phase[current_voxel_idx_tuple].item() # Use current unwrapped value as reference

                # Using wrapped phase of current_voxel for difference calculation
                # This is fine if current_voxel was just unwrapped from its own reference.
                # Or, more robustly: use wrapped_phase[current_voxel_idx_tuple]
                # diff = wp_neighbor - wrapped_phase[current_voxel_idx_tuple].item()
                # The pseudocode says: diff = WrappedPhase[NeighborVoxel] - WrappedPhase[CurrentVoxel]
                # Then UnwrappedPhase[NeighborVoxel] = UnwrappedPhase[CurrentVoxel] + wrapped(diff)
                # This is correct.
                diff = wp_neighbor - phase_data[current_voxel_idx_tuple].item()

                unwrapped_phase[neighbor_idx_tuple] = up_current + _vu_wrap_to_pi(torch.tensor(diff)).item()

                heapq.heappush(pq, (-quality_map[neighbor_idx_tuple].item(), neighbor_idx_tuple))

    if processed_count_rg >= max_iterations_rg and pq:
        print(f"Warning: Region growing reached max_iterations_rg ({max_iterations_rg}) with {len(pq)} items left in queue.")

    # --- Merge Voronoi Cells & Optimize Paths (Placeholder) ---
    unwrapped_phase = _vu_merge_voronoi_cells_and_optimize_paths(
        unwrapped_phase, phase_data, cell_id_map, quality_map, mask,
        voxel_size_tensor, tolerance, max_iterations # max_iterations here for optimize step
    )

    # --- Final Masking ---
    if mask is not None:
        unwrapped_phase.masked_fill_(~mask, 0.0)

    # Ensure unvisited regions within the mask are set to their original wrapped phase
    # or remain 0, depending on desired behavior. Current logic leaves them as 0.
    # If original phase desired for unvisited masked regions:
    # if mask is not None:
    #     unvisited_and_masked = mask & ~visited
    #     unwrapped_phase[unvisited_and_masked] = phase_data[unvisited_and_masked]

    return unwrapped_phase
