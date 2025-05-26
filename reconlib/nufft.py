import abc
import torch
import math
import numpy as np
from scipy.special import i0

class NUFFT(abc.ABC):
    def __init__(self, 
                 image_shape: tuple | list[int], 
                 k_trajectory: torch.Tensor, 
                 oversamp_factor: float = 2.0, 
                 width: int = 4, 
                 beta: float | None = None, 
                 device: str | torch.device = 'cpu'):
        """
        Initialize the NUFFT operator.

        Args:
            image_shape: Shape of the image (e.g., (256, 256)).
            k_trajectory: K-space trajectory, shape (N, D) or (N, M, D)
                          where N is the number of points, M is batch/coil, D is dimensionality.
            oversamp_factor: Oversampling factor for the grid.
            width: Width of the Kaiser-Bessel kernel.
            beta: Beta parameter for the Kaiser-Bessel kernel. 
                  If None, calculated using a MIRT-like heuristic.
            device: Computation device ('cpu' or 'cuda').
        """
        super().__init__()
        self.image_shape = tuple(image_shape)
        self.oversamp_factor = oversamp_factor
        self.width = width
        
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if not isinstance(k_trajectory, torch.Tensor):
            self.k_trajectory = torch.tensor(k_trajectory, dtype=torch.float32, device=self.device)
        else:
            self.k_trajectory = k_trajectory.to(self.device)

        if beta is None:
            # MIRT-like heuristic for beta
            # Formula: pi * sqrt((width/oversamp_factor)^2 * (oversamp_factor-0.5)^2 - 0.8)
            self.beta = math.pi * math.sqrt(
                (self.width / self.oversamp_factor)**2 * (self.oversamp_factor - 0.5)**2 - 0.8
            )
            if self.beta < 0: 
                self.beta = 10.0 
        else:
            self.beta = beta

    @abc.abstractmethod
    def forward(self, image_data: torch.Tensor) -> torch.Tensor:
        """
        Apply the forward NUFFT operation (image to k-space).

        Args:
            image_data: Input image data tensor.
                        Shape (batch_size, num_coils, *image_shape) or (batch_size, *image_shape)
                        or (*image_shape)

        Returns:
            Output k-space data tensor.
            Shape (batch_size, num_coils, num_k_points) or (batch_size, num_k_points)
            or (num_k_points)
        """
        pass

    @abc.abstractmethod
    def adjoint(self, kspace_data: torch.Tensor) -> torch.Tensor:
        """
        Apply the adjoint NUFFT operation (k-space to image).

        Args:
            kspace_data: Input k-space data tensor.
                         Shape (batch_size, num_coils, num_k_points) or (batch_size, num_k_points)
                         or (num_k_points)

        Returns:
            Output image data tensor.
            Shape (batch_size, num_coils, *image_shape) or (batch_size, *image_shape)
            or (*image_shape)
        """
        pass


class NUFFT2D(NUFFT):
    def __init__(self, 
                 image_shape: tuple | list[int], 
                 k_trajectory: torch.Tensor, 
                 oversamp_factor: float = 2.0, 
                 width: int = 4, 
                 beta: float | None = None, 
                 device: str | torch.device = 'cpu'):
        super().__init__(image_shape, k_trajectory, oversamp_factor, width, beta, device)

        if len(self.image_shape) != 2:
            raise ValueError(f"NUFFT2D expects a 2D image_shape, got {self.image_shape}")
        if self.k_trajectory.shape[-1] != 2:
            raise ValueError(f"NUFFT2D expects k_trajectory with last dimension 2, got {self.k_trajectory.shape}")

    def _kaiser_bessel_kernel(self, r: torch.Tensor) -> torch.Tensor:
        mask = r < (self.width / 2)
        val_inside_sqrt = torch.clamp(1 - (2 * r[mask] / self.width)**2, min=0.0)
        z = torch.sqrt(val_inside_sqrt)
        kb = torch.zeros_like(r)
        # Ensure z is on CPU for i0, then move result to original device
        kb_numpy_values = i0(self.beta * z.cpu().numpy())
        kb[mask] = torch.from_numpy(kb_numpy_values.astype(np.float32)).to(r.device) / float(i0(self.beta))
        return kb

    def _estimate_density_compensation(self, kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
        radius = torch.sqrt(kx**2 + ky**2)
        dcf = radius + 1e-3 # Small epsilon to avoid issues with zero radius
        dcf /= dcf.max()
        return dcf

    def adjoint(self, kspace_data: torch.Tensor) -> torch.Tensor:
        """
        Apply the adjoint NUFFT operation (k-space to image) for 2D.

        Args:
            kspace_data: Input k-space data tensor (num_k_points,).
                         Assumed to be complex-valued.

        Returns:
            Output image data tensor (image_shape[0], image_shape[1]).
        """
        if kspace_data.ndim != 1:
            raise ValueError(f"Expected kspace_data to be 1D, got shape {kspace_data.shape}")
        
        kspace_data = kspace_data.to(self.device)
        kx, ky = self.k_trajectory[:, 0], self.k_trajectory[:, 1]
        
        Nx, Ny = self.image_shape
        Nx_oversamp, Ny_oversamp = int(Nx * self.oversamp_factor), int(Ny * self.oversamp_factor)
        
        # Scale k-space coordinates to oversampled grid dimensions
        # Original kx, ky are in [-0.5, 0.5], map to [0, N_oversamp]
        kx_scaled = (kx + 0.5) * Nx_oversamp 
        ky_scaled = (ky + 0.5) * Ny_oversamp

        dcf = self._estimate_density_compensation(kx, ky).to(self.device)
        kspace_data_weighted = kspace_data * dcf
        
        grid = torch.zeros((Nx_oversamp, Ny_oversamp), dtype=torch.complex64, device=self.device)
        weight_grid = torch.zeros((Nx_oversamp, Ny_oversamp), dtype=torch.float32, device=self.device)
        
        half_width = self.width // 2 # Integer division

        # Iterate over kernel footprint
        for dx_offset in range(-half_width, half_width + 1):
            for dy_offset in range(-half_width, half_width + 1):
                # Calculate indices on the oversampled grid for each k-space point
                # These are the centers of where the kernel for each k-point will be placed
                x_idx_center = kx_scaled + dx_offset # Not floored yet
                y_idx_center = ky_scaled + dy_offset # Not floored yet

                # Distances from the actual k-space sample (after scaling) to the kernel sample point
                # This seems inverted from typical gridding logic.
                # Let's rethink: kernel is centered on kx_scaled, ky_scaled.
                # We are iterating grid cells (x_grid, y_grid) around each k-point.
                # No, the original logic seems to iterate kernel points around each k-point.
                # Let (k_x, k_y) be a point in k-space.
                # For each grid point (g_x, g_y) within kernel support of (k_x, k_y):
                #   dist = sqrt( (k_x - g_x)^2 + (k_y - g_y)^2 )
                #   w = kernel(dist)
                #   grid[g_x, g_y] += w * kspace_data_pt
                # This seems more standard for gridding.
                # The provided code iterates dx, dy which are offsets from k-point for kernel evaluation.
                # x_idx, y_idx are grid points.
                # x_dist, y_dist are distances from k-point to *that* grid point.
                
                # The original logic:
                # x_idx and y_idx are the grid cells that each k-point (offset by dx,dy) maps to.
                # x_dist and y_dist are then sub-pixel distances used for kernel evaluation.
                # This seems to be a "spreading" or "gridding" operation.

                # Let's follow the structure of _iternufft2d_nufft2d2_adjoint
                # It uses x_idx, y_idx as integer grid indices, and x_dist, y_dist for kernel eval.
                # kx_scaled, ky_scaled are the "true" locations of k-space samples on the oversampled grid.
                # dx, dy are offsets defining the kernel extent.
                # We are looking at grid points (ix, iy) = floor(kx_scaled + dx_offset), floor(ky_scaled + dy_offset)
                # The distance `r` for the kernel should be from the *center* of the grid cell (ix+0.5, iy+0.5)
                # to the actual k-space sample location (kx_scaled, ky_scaled).
                # Or, if dx, dy are offsets from the k-point to sample points of the kernel:
                
                # Let's try to match the logic from _iternufft2d_nufft2d2_adjoint more closely.
                # It seems the dx, dy loop is to iterate over the cells affected by each k-space sample.
                # For each k-space sample (kx_s, ky_s) = (kx_scaled[i], ky_scaled[i]):
                # Iterate grid cells (gx, gy) in the neighborhood of (kx_s, ky_s) of size width x width.
                # gx_min = floor(kx_s - width/2), gx_max = floor(kx_s + width/2)
                
                # The original code's dx, dy loop means: for each k-space point,
                # consider a set of kernel evaluation points offset by (dx, dy) from it.
                # Then find which grid cell this kernel evaluation point falls into.
                # This seems more like interpolation (forward) than gridding (adjoint).

                # Let's re-evaluate the loops from the source:
                # `x_idx, y_idx = torch.floor(kx_scaled + dx).long(), torch.floor(ky_scaled + dy).long()`
                # `x_dist, y_dist = kx_scaled - x_idx.float(), ky_scaled - y_idx.float()`
                # This x_dist, y_dist is distance from k-point to top-left of grid cell.
                # The kernel `r` is `torch.sqrt(x_dist**2 + y_dist**2)`.
                # This `r` is used to calculate `w`.
                # `grid[x_idx_mod, y_idx_mod] += kspace_data_weighted[i] * w[i]`
                
                # This is a gridding operation. For each k-space point (kx_scaled, ky_scaled):
                # it iterates over a width x width area of *grid cells* around it.
                # dx, dy are offsets from the k-space point to the *center* of these grid cells.
                # No, dx and dy are integer offsets for grid cells.
                # Let k_s = (kx_scaled[i], ky_scaled[i]) be the i-th k-space sample.
                # Let g_c = (cx, cy) be the *closest* grid cell index: cx = floor(kx_s), cy = floor(ky_s).
                # The loop `for dx in range(-half_width, half_width + 1)`:
                #   `idx_x = cx + dx` (this is a grid cell index)
                #   `dist_x = kx_s - (idx_x + 0.5)` (dist from k-point to center of this grid cell)
                # This is standard gridding.

                # Let's adapt the provided code's loop structure.
                # For each k-space point k_pt = (kx_scaled[i], ky_scaled[i])
                # Iterate integer grid cell offsets (kernel_dx, kernel_dy) from -W/2 to W/2
                for kernel_gx_offset in range(-half_width, half_width + 1):
                    for kernel_gy_offset in range(-half_width, half_width + 1):
                        # Absolute grid cell index targeted by this kernel sample
                        # This is the grid cell (gx, gy) we are currently considering.
                        gx = torch.floor(kx_scaled + kernel_gx_offset).long()
                        gy = torch.floor(ky_scaled + kernel_gy_offset).long()

                        # Distances from the k-space point (kx_scaled, ky_scaled) to the *center* of the current grid cell (gx+0.5, gy+0.5)
                        # No, the original code used:
                        # x_dist = kx_scaled - gx.float()  (dist from k-point to left edge of cell gx)
                        # y_dist = ky_scaled - gy.float()  (dist from k-point to top edge of cell gy)
                        # This is not quite right for kernel centered on k-point.
                        # The kernel argument 'r' should be the distance from the k-space sample
                        # to the center of the grid cell being updated.
                        # r_x = kx_scaled - (gx.float() + 0.5)
                        # r_y = ky_scaled - (gy.float() + 0.5)
                        # No, the original code's `r` is `sqrt((k_x - floor(k_x+dx))^2 + ...)`
                        # This means `dx, dy` are NOT grid cell offsets relative to `floor(k_x)`.
                        # They are offsets *added to k_x before flooring*.
                        # So `x_idx = floor(k_x_scaled + dx_offset)` IS the grid cell.
                        # And `x_dist = k_x_scaled - x_idx` means `k_x_scaled - floor(k_x_scaled + dx_offset)`.
                        # This is not a distance to grid center.
                        
                        # Let's use the exact structure from the source _iternufft2d_nufft2d2_adjoint
                        # dx, dy are offsets from the k-point, defining where kernel is sampled.
                        # x_idx, y_idx are grid points where these samples are accumulated.
                        
                        # These are grid indices being updated.
                        grid_x_indices = torch.floor(kx_scaled + dx_offset).long()
                        grid_y_indices = torch.floor(ky_scaled + dy_offset).long()

                        # Distances for kernel calculation: distance from k-point to the center of the kernel sample point (dx_offset, dy_offset away)
                        # No, r is distance from k-point to the grid point (grid_x_indices, grid_y_indices)
                        # dist_x_to_grid_cell_corner = kx_scaled - grid_x_indices.float()
                        # dist_y_to_grid_cell_corner = ky_scaled - grid_y_indices.float()
                        # r = torch.sqrt(dist_x_to_grid_cell_corner**2 + dist_y_to_grid_cell_corner**2)
                        # This is not what _iternufft2d_kaiser_bessel_kernel expects for `r`.
                        # The `r` for kaiser_bessel_kernel is typically |k - g| / (W/2) or similar.
                        # In _iternufft2d_kaiser_bessel_kernel, r is distance from k-point to kernel sample point.
                        # The kernel values w are computed based on distances dx_offset, dy_offset.
                        
                        # Let's use (dx_offset, dy_offset) as the argument to the kernel, representing distance.
                        # This is `r_kernel_arg = sqrt(dx_offset^2 + dy_offset^2)`.
                        # This must be a tensor for _kaiser_bessel_kernel.
                        # This means for each k-point, we sum contributions from WxW kernel samples.
                        # Each kernel sample value w(dx,dy) is placed onto grid cell floor(k_scaled + dx, k_scaled + dy).
                        
                        # Create a tensor for r for this (dx_offset, dy_offset)
                        # This is the distance of this specific kernel tap (dx_offset, dy_offset) from the kernel center (0,0)
                        r_kernel = torch.sqrt(torch.tensor(dx_offset**2 + dy_offset**2, dtype=torch.float32, device=self.device))
                        
                        # Kernel weight for this tap. r_kernel must be a tensor.
                        w_kernel_tap = self._kaiser_bessel_kernel(r_kernel.expand(kx_scaled.shape[0])) # expand to all k-points

                        # Grid cells to update
                        gx_target = torch.floor(kx_scaled - dx_offset).long() # grid cell corresponding to this kernel tap
                        gy_target = torch.floor(ky_scaled - dy_offset).long() # if kernel is centered at k_scaled

                        # This interpretation seems more aligned with how KB kernels are used in gridding.
                        # For each k-space point (k_x, k_y):
                        #   Iterate grid cells (g_x, g_y) in its neighborhood (width x width around k_x, k_y)
                        #   Calculate distance r = sqrt( (k_x - g_x)^2 + (k_y - g_y)^2 )
                        #   Calculate kernel weight w = KB(r)
                        #   grid[g_x, g_y] += w * data_k / sum_w_for_this_g_x_g_y
                        #   weight_grid[g_x, g_y] += w
                        
                        # Let's follow the loop structure of the source code for now.
                        # dx_offset, dy_offset are kernel sample positions relative to kernel center.
                        
                        # grid cell indices affected by k-space points when using this kernel tap
                        # (kx_scaled is true k-space position on oversampled grid)
                        # (dx_offset, dy_offset) is current part of kernel filter we are looking at
                        # Target grid cells are floor(kx_scaled - dx_offset), floor(ky_scaled - dy_offset)
                        # if kernel is centered at (0,0) and data point is at (kx_scaled, ky_scaled)
                        # and we are summing onto the grid.
                        
                        # Let's use the original formulation as closely as possible.
                        # x_idx, y_idx: grid cell where the data is placed.
                        # x_dist, y_dist: distance from k-space point to that grid cell's origin.
                        # r: based on x_dist, y_dist. This implies kernel value depends on sub-pixel location.
                        
                        x_idx = torch.floor(kx_scaled + dx_offset).long()
                        y_idx = torch.floor(ky_scaled + dy_offset).long()
                        
                        # distance from k-space point (kx_scaled, ky_scaled) to the point (x_idx - dx_offset, y_idx - dy_offset)
                        # This does not look like the 'r' for the KB kernel.
                        # The 'r' should be distance from k-space point to grid cell center, normalized.
                        # Or, if kernel is sampled at discrete points (dx,dy), then 'r' is distance of that sample point from kernel center.
                        
                        # The original code:
                        # x_dist, y_dist = kx_scaled - x_idx.float(), ky_scaled - y_idx.float()
                        # r = torch.sqrt(x_dist**2 + y_dist**2)
                        # This implies r is distance from k-space point to the corner of the grid cell x_idx, y_idx.
                        # And x_idx, y_idx are floor(k_scaled + dx_offset), floor(k_scaled + dy_offset)
                        # This is confusing.

                        # Let's assume the kernel is centered at each k-space point (kx_s, ky_s).
                        # We iterate over grid cells (gx, gy) that are within W/2 of (kx_s, ky_s).
                        # gx_min = floor(kx_s - W/2), gx_max = ceil(kx_s + W/2)
                        # This is what typical gridding does.
                        # The loop `for dx in range(-half_width, half_width + 1)` iterates W times.
                        
                        # Sticking to the original structure:
                        # dx_offset, dy_offset are relative coordinates for kernel samples.
                        # The kernel is evaluated AT these relative coordinates.
                        r_for_kernel = torch.sqrt(torch.tensor(dx_offset**2 + dy_offset**2, dtype=torch.float32, device=self.device))
                        # This needs to be tensor for each k-point if it varies, but here it's fixed for given (dx,dy)
                        # The kernel function itself handles scalar r if needed, but here it's vectorized.
                        # So, `r` must be a tensor of distances.
                        # The original `_iternufft2d_kaiser_bessel_kernel` takes `r` as a tensor.
                        # This `r` should be the distance from the center of the kernel to the sample point.
                        # So, `r_values_for_this_offset` should be `sqrt(dx_offset^2 + dy_offset^2)`
                        # This is a single scalar value for fixed (dx_offset, dy_offset).
                        # We need to make it a tensor of the same size as kx_scaled for the kernel function.
                        
                        # `r_arg` should be the distance from the k-space sample to the grid point, scaled by width.
                        # Let's use the structure from the forward pass for calculating w, as it seems more standard for interpolation.
                        # For adjoint (gridding):
                        # For each k-space point k_s = (kx_scaled[i], ky_scaled[i]):
                        #   For each grid point g = (gx, gy) in the WxW neighborhood of k_s:
                        #     dist_x = kx_scaled[i] - gx 
                        #     dist_y = ky_scaled[i] - gy
                        #     r = sqrt(dist_x^2 + dist_y^2)  <- this is the argument for KB kernel
                        #     w = _kaiser_bessel_kernel(r)
                        #     grid[gx % Nx_oversamp, gy % Ny_oversamp] += kspace_data_weighted[i] * w
                        #     weight_grid[gx % Nx_oversamp, gy % Ny_oversamp] += w

                        # This requires iterating neighborhood for EACH k-point.
                        # The current loops iterate WxW times, and inside, vectorized over all k-points.
                        
                        # Let's use the provided `_iternufft2d_nufft2d2_adjoint` logic directly.
                        # dx_offset, dy_offset are offsets from the *grid point* to the *k-space sample*.
                        # No, this is also not right.

                        # The loop structure `for dx_offset... for dy_offset...` implies that for each
                        # k-space point, we are considering W*W "kernel taps".
                        # `x_idx, y_idx` are the grid cells that the (k_point + offset) falls into.
                        # `x_dist, y_dist` are `k_point - x_idx` (subpixel parts).
                        # `r = sqrt(x_dist^2 + y_dist^2)` is the argument to the kernel.
                        # This means the kernel shape depends on the subpixel location of (k_point+offset).
                        # This is complex.

                        # Let's use the exact formulation from the source:
                        # `kx_scaled`, `ky_scaled` are the k-space sample positions on the oversampled grid.
                        # `dx_offset`, `dy_offset` are integers from -W/2 to W/2.
                        
                        # `gx_absolute = torch.floor(kx_scaled + dx_offset).long()` : grid x-indices for these kernel points
                        # `gy_absolute = torch.floor(ky_scaled + dy_offset).long()` : grid y-indices

                        # `dist_x_to_gx_corner = kx_scaled - gx_absolute.float()`
                        # `dist_y_to_gy_corner = ky_scaled - gy_absolute.float()`
                        # `r_kernel_arg = torch.sqrt(dist_x_to_gx_corner**2 + dist_y_to_gy_corner**2)`
                        # This `r` is the distance from k-space sample to corner of grid cell `(gx_absolute, gy_absolute)`.
                        # This still feels off. The kernel argument `r` in `_iternufft2d_kaiser_bessel_kernel`
                        # is `distance_from_center_of_kernel_support / (width/2)`.
                        # If kernel is centered at `(kx_scaled, ky_scaled)`, and we are evaluating its effect on
                        # grid cell `(gx,gy)`, then `r` should be `sqrt( (kx_scaled-gx)^2 + (ky_scaled-gy)^2 )`.

                        # Let's use the forward pass's kernel evaluation logic as a reference, it's often more intuitive.
                        # Forward: `x_idx = floor(kx_scaled + dx)`, `y_idx = floor(ky_scaled + dy)`
                        # `x_dist = kx_scaled - x_idx`, `y_dist = ky_scaled - y_idx`
                        # `r = sqrt(x_dist^2 + y_dist^2)`
                        # `w = kernel(r)`
                        # `kspace_data += grid[x_idx, y_idx] * w`
                        # This `r` is distance from `kx_scaled` to `x_idx` (corner).
                        # The roles of dx,dy are to select different grid points `x_idx, y_idx` around `kx_scaled`.

                        # Re-interpreting the original _iternufft2d_nufft2d2_adjoint:
                        # For each k-space point `k_s = (kx_scaled[i], ky_scaled[i])`:
                        #   Loop `dx` from -W/2 to W/2, `dy` from -W/2 to W/2.
                        #   These `dx, dy` define the grid cell `g = (floor(k_s_x + dx), floor(k_s_y + dy))`.
                        #   No, this is not it. `dx,dy` are not offsets from `k_s`. They are absolute offsets.
                        
                        # The most straightforward gridding:
                        # For each k-space point `k_s = (kx_scaled[i], ky_scaled[i])`:
                        #   `center_gx = round(kx_scaled[i])`, `center_gy = round(ky_scaled[i])` (or floor)
                        #   Loop `gx_offset` from -W/2 to W/2, `gy_offset` from -W/2 to W/2.
                        #     `gx = center_gx + gx_offset`
                        #     `gy = center_gy + gy_offset`
                        #     `dist_x = kx_scaled[i] - gx` (or `gx_center = gx + 0.5`)
                        #     `dist_y = ky_scaled[i] - gy`
                        #     `r = torch.sqrt(dist_x**2 + dist_y**2)`
                        #     `w = self._kaiser_bessel_kernel(r)`
                        #     `grid[gx % Nxo, gy % Nyo] += kspace_data_weighted[i] * w`
                        #     `weight_grid[gx % Nxo, gy % Nyo] += w`
                        # This is how I'll implement it. It's standard.

                        center_gx_approx = torch.round(kx_scaled).long() # nearest grid point to k_sample
                        center_gy_approx = torch.round(ky_scaled).long()

                        # Iterate over WxW grid cells around each k-space sample
                        # dx_offset, dy_offset are offsets from center_gx_approx, center_gy_approx
                        current_gx = center_gx_approx + dx_offset # absolute grid index
                        current_gy = center_gy_approx + dy_offset # absolute grid index
                        
                        # Check bounds (optional, as modulo arithmetic handles it, but good for clarity)
                        # if torch.any(current_gx < 0) or torch.any(current_gx >= Nx_oversamp) or \
                        #    torch.any(current_gy < 0) or torch.any(current_gy >= Ny_oversamp):
                        #    continue # This check is problematic with tensors. Rely on modulo.

                        dist_x_to_cell_center = kx_scaled - (current_gx.float() + 0.5)
                        dist_y_to_cell_center = ky_scaled - (current_gy.float() + 0.5)
                        
                        r_vals = torch.sqrt(dist_x_to_cell_center**2 + dist_y_to_cell_center**2)
                        kernel_weights = self._kaiser_bessel_kernel(r_vals) # r_vals must be scaled by W/2 inside kernel or here

                        # Modulo for periodic boundary conditions
                        gx_target_mod = current_gx % Nx_oversamp
                        gy_target_mod = current_gy % Ny_oversamp

                        # Accumulate weighted k-space data onto the grid
                        # This needs to be done carefully with tensor indexing if not looping k-points
                        # The loop is over kernel offsets, operations inside are vectorized over k-points.
                        # So, we need scatter_add_ or index_add_ for `grid` and `weight_grid`.
                        
                        # Using a simpler loop for now, similar to the original, assuming it was correct.
                        # The original code's `x_idx, y_idx` were calculated based on `kx_scaled + dx_offset`.
                        # This means `dx_offset, dy_offset` are not offsets from `center_gx_approx`.
                        # They are offsets from the k-space sample location *used to determine which grid cell to update*.
                        # And `r` was `kx_scaled - x_idx`, which is `kx_scaled - floor(kx_scaled + dx_offset)`.
                        # This is `(kx_scaled + dx_offset) - floor(kx_scaled + dx_offset) - dx_offset`
                        # i.e., `frac(kx_scaled + dx_offset) - dx_offset`. This `r` is unusual for KB.

                        # Let's revert to the structure from the provided `_iternufft2d_nufft2d2_adjoint`
                        # and trust its formulation of `r` and indexing.
                        
                        # gx_target = floor(kx_scaled + dx_offset)
                        # gy_target = floor(ky_scaled + dy_offset)
                        gx_target = torch.floor(kx_scaled + dx_offset).long()
                        gy_target = torch.floor(ky_scaled + dy_offset).long()

                        # dist_for_kernel_arg_x = kx_scaled - gx_target.float() # This is kx_s - floor(kx_s+dx) = frac(kx_s+dx) - dx (if dx != 0)
                        # dist_for_kernel_arg_y = ky_scaled - gy_target.float()
                        
                        # The `r` in the original `_iternufft2d_kaiser_bessel_kernel` is distance from kernel sample point to kernel center.
                        # So, if `(dx_offset, dy_offset)` is the coordinate of the kernel sample point relative to its center,
                        # then `r = sqrt(dx_offset^2 + dy_offset^2)`.
                        # This `r` is then used for `w`.
                        # And `w` is applied to grid cell `floor(kx_scaled - dx_offset), floor(ky_scaled - dy_offset)`.
                        # (Assuming kernel is centered at kx_scaled, and its tap at (-dx_offset, -dy_offset) affects grid cell floor(kx_scaled+dx_offset, k_scaled+dy_offset))
                        # Or, if kernel tap is at (dx_offset, dy_offset) relative to kx_scaled, it affects grid cell floor(kx_scaled+dx_offset).
                        
                        # Using the direct translation of _iternufft2d_nufft2d_adjoint's r:
                        dist_x = kx_scaled - gx_target.float() # dist from k-point to target grid cell's corner
                        dist_y = ky_scaled - gy_target.float() # dist from k-point to target grid cell's corner
                        r_kernel_arg = torch.sqrt(dist_x**2 + dist_y**2)
                        # This `r` is passed to KB. KB expects `r` to be distance from center, normalized by W/2.
                        # If `r` is `|k_actual - grid_target_corner|`, this seems not directly what KB expects.
                        # However, the original code used this.
                        
                        # Let's assume `r` for `_kaiser_bessel_kernel` is simply the distance.
                        # The normalization `2*r/width` happens inside `_kaiser_bessel_kernel`.
                        # So `r` should be `sqrt( (k_center_x - grid_cell_center_x)^2 + ...)`
                        # The original code: `r = torch.sqrt( (kx_scaled - x_idx.float())**2 + (ky_scaled - y_idx.float())**2 )`
                        # where `x_idx = floor(kx_scaled + dx_offset)`.
                        # This is `r = | k_s - floor(k_s + d) |`.
                        
                        # The most robust way is to calculate distance from k-space sample to the *center* of each grid cell in the window.
                        # (current_gx_center, current_gy_center) = ( (gx_target.float() + 0.5), (gy_target.float() + 0.5) )
                        # r_dist_k_to_cell_center_x = kx_scaled - (gx_target.float() + 0.5)
                        # r_dist_k_to_cell_center_y = ky_scaled - (gy_target.float() + 0.5)
                        # r_vals = torch.sqrt(r_dist_k_to_cell_center_x**2 + r_dist_k_to_cell_center_y**2)
                        # kernel_weights = self._kaiser_bessel_kernel(r_vals)
                        
                        # This seems the most standard interpretation for gridding.
                        # (gx_target, gy_target) are the grid cells we are iterating over relative to k-point.
                        # Let dx_offset, dy_offset be the relative indices of grid cells around k_point.
                        # gx_absolute = floor(kx_scaled) + dx_offset
                        # gy_absolute = floor(ky_scaled) + dy_offset
                        # r_x = kx_scaled - (gx_absolute.float() + 0.5)
                        # r_y = ky_scaled - (gy_absolute.float() + 0.5)
                        # r = sqrt(r_x^2 + r_y^2)
                        # w = KB(r)
                        # grid[gx_abs % Nxo, gy_abs % Nyo] += kspace_data_w[i] * w[i]
                        # weight_grid[gx_abs % Nxo, gy_abs % Nyo] += w[i]
                        # This is the structure I will use. dx_offset, dy_offset loop over kernel extent.

                        gx_absolute_cell_index = torch.floor(kx_scaled).long() + dx_offset
                        gy_absolute_cell_index = torch.floor(ky_scaled).long() + dy_offset
                        
                        dist_x_k_to_cell_center = kx_scaled - (gx_absolute_cell_index.float() + 0.5)
                        dist_y_k_to_cell_center = ky_scaled - (gy_absolute_cell_index.float() + 0.5)
                        
                        r_for_kb = torch.sqrt(dist_x_k_to_cell_center**2 + dist_y_k_to_cell_center**2)
                        kernel_sinc_weights = self._kaiser_bessel_kernel(r_for_kb) # these are w_j(k_i) in math notation

                        gx_mod = gx_absolute_cell_index % Nx_oversamp
                        gy_mod = gy_absolute_cell_index % Ny_oversamp
                        
                        # Vectorized update using index_add_ (scatter_add_)
                        # Create flat indices for accumulation
                        flat_indices = gx_mod * Ny_oversamp + gy_mod # Check if k-points dim is first or last
                        
                        # kspace_data_weighted and kernel_sinc_weights are (num_k_points,)
                        # grid and weight_grid are (Nx_oversamp, Ny_oversamp)
                        # Need to loop over k-points if using simple indexing, or use scatter_add.
                        # The original code did:
                        # for i in range(kspace_data_weighted.shape[0]):
                        #   grid[gx_mod[i], gy_mod[i]] += kspace_data_weighted[i] * kernel_sinc_weights[i]
                        #   weight_grid[gx_mod[i], gy_mod[i]] += kernel_sinc_weights[i]
                        # This is correct but slow if not JITed.
                        # PyTorch equivalent:
                        # gx_mod_flat = gx_mod.flatten()
                        # gy_mod_flat = gy_mod.flatten()
                        # kspace_flat = kspace_data_weighted # already flat
                        # kernel_w_flat = kernel_sinc_weights # already flat
                        
                        # grid.view(-1).index_add_(0, gx_mod * Ny_oversamp + gy_mod, kspace_data_weighted * kernel_sinc_weights)
                        # weight_grid.view(-1).index_add_(0, gx_mod * Ny_oversamp + gy_mod, kernel_sinc_weights)
                        # This is for unique indices. If indices repeat (multiple k-points map to same grid cell with this offset),
                        # index_add_ is correct. gx_mod is (num_k_points), so indices can repeat.

                        # Flatten target indices for grid
                        target_flat_indices = gx_mod * Ny_oversamp + gy_mod
                        
                        grid.view(-1).index_add_(0, target_flat_indices, kspace_data_weighted * kernel_sinc_weights)
                        weight_grid.view(-1).index_add_(0, target_flat_indices, kernel_sinc_weights)

        weight_grid = torch.where(weight_grid == 0, torch.ones_like(weight_grid), weight_grid)
        grid = grid / weight_grid
        
        # Inverse FFT and cropping
        img = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(grid)))
        start_x, start_y = (Nx_oversamp - Nx) // 2, (Ny_oversamp - Ny) // 2
        img_cropped = img[start_x:start_x + Nx, start_y:start_y + Ny]
        
        return img_cropped * float(Nx_oversamp * Ny_oversamp) # Scaling factor based on FFT size (like sigpy)


    def forward(self, image_data: torch.Tensor) -> torch.Tensor:
        """
        Apply the forward NUFFT operation (image to k-space) for 2D.

        Args:
            image_data: Input image data tensor (image_shape[0], image_shape[1]).
                        Assumed to be complex-valued.

        Returns:
            Output k-space data tensor (num_k_points,).
        """
        if image_data.shape != self.image_shape:
            raise ValueError(f"Input image_data shape {image_data.shape} does not match expected {self.image_shape}")

        image_data = image_data.to(self.device)
        kx, ky = self.k_trajectory[:, 0], self.k_trajectory[:, 1]

        Nx, Ny = self.image_shape
        Nx_oversamp, Ny_oversamp = int(Nx * self.oversamp_factor), int(Ny * self.oversamp_factor)

        # Pad image
        pad_x, pad_y = (Nx_oversamp - Nx) // 2, (Ny_oversamp - Ny) // 2
        image_padded = torch.zeros((Nx_oversamp, Ny_oversamp), dtype=torch.complex64, device=self.device)
        image_padded[pad_x:pad_x + Nx, pad_y:pad_y + Ny] = image_data
        
        # FFT of padded image
        kspace_cart_oversamp = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image_padded)))
        
        # Scale k-space coordinates
        kx_scaled = (kx + 0.5) * Nx_oversamp
        ky_scaled = (ky + 0.5) * Ny_oversamp
        
        half_width = self.width // 2
        
        kspace_noncart = torch.zeros(kx.shape[0], dtype=torch.complex64, device=self.device)
        weight_sum = torch.zeros(kx.shape[0], dtype=torch.float32, device=self.device)

        # Interpolation from oversampled Cartesian grid to non-Cartesian points
        # For each non-Cartesian point (kx_s, ky_s) = (kx_scaled[i], ky_scaled[i]):
        #   Loop `dx_offset` from -W/2 to W/2, `dy_offset` from -W/2 to W/2.
        #   These `dx_offset, dy_offset` define the grid cell `g = (floor(kx_s + dx_offset), floor(ky_s + dy_offset))`
        #   No, this is not right.
        #   The loop is over kernel sample points relative to the *output* non-Cartesian point.
        #   For each k-space point `k_s = (kx_scaled[i], ky_scaled[i])`:
        #     `val = 0`, `w_sum = 0`
        #     Loop `dx_offset` from -W/2 to W/2, `dy_offset` from -W/2 to W/2 (these are grid cell offsets from k_s)
        #       `gx = floor(kx_scaled[i]) + dx_offset`
        #       `gy = floor(ky_scaled[i]) + dy_offset`
        #       `dist_x = kx_scaled[i] - (gx.float() + 0.5)` (dist from k-point to center of this grid cell)
        #       `dist_y = ky_scaled[i] - (gy.float() + 0.5)`
        #       `r = sqrt(dist_x^2 + dist_y^2)`
        #       `w = self._kaiser_bessel_kernel(r)`
        #       `val += kspace_cart_oversamp[gx % Nxo, gy % Nyo] * w`
        #       `w_sum += w`
        #     `kspace_noncart[i] = val / w_sum`
        # This is the standard interpolation logic.

        for dx_offset in range(-half_width, half_width + 1):
            for dy_offset in range(-half_width, half_width + 1):
                # Grid cell indices from which to interpolate
                # These are neighbors of the k-space sample point on the oversampled grid.
                # gx_source = floor(kx_scaled) + dx_offset -> this is what I used for adjoint.
                # The original code for forward:
                # x_idx = floor(kx_scaled + dx_offset).long()
                # y_idx = floor(ky_scaled + dy_offset).long()
                # x_dist = kx_scaled - x_idx.float()
                # y_dist = ky_scaled - y_idx.float()
                # r = torch.sqrt(x_dist**2 + y_dist**2)
                # w = _kaiser_bessel_kernel(r, self.width, self.beta)
                # kspace_data += kspace_cart[x_idx_mod, y_idx_mod] * w
                # weight_sum += w
                # This means x_idx, y_idx are grid points. r is distance from kx_scaled to corner of x_idx,y_idx.
                # This is not distance from kx_scaled to center of grid cell x_idx,y_idx.
                # Let's use the standard interpolation interpretation.

                # gx_source_absolute indices of grid points contributing to kx_scaled
                gx_source_absolute = torch.floor(kx_scaled).long() + dx_offset
                gy_source_absolute = torch.floor(ky_scaled).long() + dy_offset

                dist_x_k_to_grid_center = kx_scaled - (gx_source_absolute.float() + 0.5)
                dist_y_k_to_grid_center = ky_scaled - (gy_source_absolute.float() + 0.5)

                r_for_kb = torch.sqrt(dist_x_k_to_grid_center**2 + dist_y_k_to_grid_center**2)
                kernel_interp_weights = self._kaiser_bessel_kernel(r_for_kb)

                gx_source_mod = gx_source_absolute % Nx_oversamp
                gy_source_mod = gy_source_absolute % Ny_oversamp
                
                kspace_noncart += kspace_cart_oversamp[gx_source_mod, gy_source_mod] * kernel_interp_weights
                weight_sum += kernel_interp_weights
        
        weight_sum = torch.where(weight_sum == 0, torch.ones_like(weight_sum), weight_sum)
        kspace_noncart /= weight_sum
        
        return kspace_noncart / float(Nx_oversamp * Ny_oversamp) # Scaling factor (like sigpy)

