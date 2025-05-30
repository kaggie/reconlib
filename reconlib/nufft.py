import abc
import torch
import math
import numpy as np
import scipy.special # For iv, i0

class NUFFT(abc.ABC):
    def __init__(self, 
                 image_shape: tuple[int, ...], 
                 k_trajectory: torch.Tensor, 
                 oversamp_factor: tuple[float, ...],
                 kb_J: tuple[int, ...],
                 kb_alpha: tuple[float, ...],
                 Ld: tuple[int, ...],
                 kb_m: tuple[float, ...] | None = None,
                 Kd: tuple[int, ...] | None = None,
                 density_comp_weights: torch.Tensor = None, # New parameter
                 device: str | torch.device = 'cpu'):
        """
        Initialize the NUFFT operator with MIRT-style parameters.

        Args:
            image_shape: Shape of the image (e.g., (256, 256)).
            k_trajectory: K-space trajectory, shape (N, D) or (N, M, D).
            oversamp_factor: Oversampling factor per dimension for the grid (e.g., (2.0, 2.0)).
            kb_J: Kaiser-Bessel kernel width per dimension (e.g., (4, 4)).
            kb_alpha: Kaiser-Bessel alpha shape parameter per dimension (e.g., (2.34*4, 2.34*4)).
            Ld: Table oversampling factor per dimension (e.g., (2**10, 2**10)).
            kb_m: Kaiser-Bessel m order parameter per dimension (e.g., (0.0, 0.0)).
                  Defaults to (0.0,) * len(kb_J) if None.
            Kd: Oversampled grid dimensions (e.g., (512, 512)). 
                  If None, calculated as tuple(int(N * os) for N, os in zip(image_shape, oversamp_factor)).
            density_comp_weights (torch.Tensor, optional): Precomputed density compensation weights.
                                                           Shape should match the number of k-space points.
                                                           Defaults to None.
            device: Computation device ('cpu' or 'cuda').
        """
        super().__init__()
        self.image_shape = tuple(image_shape)
        self.oversamp_factor = tuple(oversamp_factor)
        self.kb_J = tuple(kb_J)
        self.kb_alpha = tuple(kb_alpha)
        self.Ld = tuple(Ld)

        if kb_m is None:
            self.kb_m = (0.0,) * len(self.kb_J)
        else:
            self.kb_m = tuple(kb_m)

        if Kd is None:
            self.Kd = tuple(int(N * os) for N, os in zip(self.image_shape, self.oversamp_factor))
        else:
            self.Kd = tuple(Kd)
        
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if not isinstance(k_trajectory, torch.Tensor):
            self.k_trajectory = torch.tensor(k_trajectory, dtype=torch.float32, device=self.device)
        else:
            self.k_trajectory = k_trajectory.to(self.device)

        if density_comp_weights is not None:
            if not isinstance(density_comp_weights, torch.Tensor):
                raise TypeError("density_comp_weights must be a PyTorch Tensor.")
            
            # Determine the number of k-space points from k_trajectory for validation
            # k_trajectory is expected to be (num_total_k_points, num_dims) after potential reshaping by user
            num_k_points_in_traj = self.k_trajectory.shape[0]
            if self.k_trajectory.ndim > 2 : # e.g. (shots, samples_per_shot, D)
                # If k_trajectory is multi-dimensional beyond (N,D), this check assumes
                # density_comp_weights matches the first dimension, or it needs to be flattened
                # by the caller to match a flattened k_trajectory.
                # The iterative_reconstruction passes sampling_points as (N,d) and weights as (N,).
                # So, this condition is less likely with current iterative_reconstruction.
                pass

            if density_comp_weights.ndim == 0 or density_comp_weights.shape[0] != num_k_points_in_traj:
                 # Basic check for 1D and matching first dimension.
                 # Does not cover all k_trajectory shape possibilities (e.g. if k_trajectory is not yet flattened)
                raise ValueError(
                    f"density_comp_weights must be a 1D tensor with length matching the number of "
                    f"k-space points ({num_k_points_in_traj}), got shape {density_comp_weights.shape}."
                )
            # Store as float32, as DCWs are typically real-valued scaling factors.
            self.density_comp_weights = density_comp_weights.to(device=self.device, dtype=torch.float32)
        else:
            self.density_comp_weights = None

        # Validations for tuple lengths
        num_dims = len(self.image_shape)
        if len(self.oversamp_factor) != num_dims:
            raise ValueError(f"Length of oversamp_factor {len(self.oversamp_factor)} must match image dimensionality {num_dims}")
        if len(self.kb_J) != num_dims:
            raise ValueError(f"Length of kb_J {len(self.kb_J)} must match image dimensionality {num_dims}")
        if len(self.kb_alpha) != num_dims:
            raise ValueError(f"Length of kb_alpha {len(self.kb_alpha)} must match image dimensionality {num_dims}")
        if len(self.Ld) != num_dims:
            raise ValueError(f"Length of Ld {len(self.Ld)} must match image dimensionality {num_dims}")
        if len(self.kb_m) != num_dims:
            raise ValueError(f"Length of kb_m {len(self.kb_m)} must match image dimensionality {num_dims}")
        if len(self.Kd) != num_dims:
            raise ValueError(f"Length of Kd {len(self.Kd)} must match image dimensionality {num_dims}")


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
                 image_shape: tuple[int, int], 
                 k_trajectory: torch.Tensor, 
                 oversamp_factor: tuple[float, float] = (2.0, 2.0),
                 kb_J: tuple[int, int] = (4, 4),
                 kb_alpha: tuple[float, float] | None = None,
                 Ld: tuple[int, int] = (1024, 1024),
                 kb_m: tuple[float, float] = (0.0, 0.0),
                 Kd: tuple[int, int] | None = None,
                 density_comp_weights: torch.Tensor | None = None,
                 device: str | torch.device = 'cpu'):
        """Initializes the 2D Non-Uniform Fast Fourier Transform (NUFFT) operator.

        This operator uses a Kaiser-Bessel kernel for interpolation between the
        non-uniform k-space samples and a Cartesian grid.

        Args:
            image_shape: Shape of the target image (Ny, Nx), e.g., (128, 128).
            k_trajectory: Tensor of k-space trajectory coordinates, normalized to
                the range [-0.5, 0.5] in each dimension.
                Shape: (num_k_points, 2).
            oversamp_factor: Oversampling factor for the Cartesian grid for NUFFT
                operations. Default is (2.0, 2.0).
            kb_J: Width of the Kaiser-Bessel interpolation kernel in grid units
                (number of neighbors). Default is (4, 4).
            kb_alpha: Shape parameter for the Kaiser-Bessel kernel. If None,
                it's automatically calculated as `2.34 * J` for each dimension,
                which is a common heuristic for `oversamp_factor=2.0`.
                Default is None.
            Ld: Size of the lookup table for Kaiser-Bessel kernel interpolation.
                Larger values provide more accuracy but increase memory.
                Default is (1024, 1024).
            kb_m: Order of the Kaiser-Bessel kernel (typically 0.0 for standard
                MRI applications). Default is (0.0, 0.0).
            Kd: Dimensions of the oversampled Cartesian grid (Kdy, Kdx). If None,
                it's calculated as `image_shape * oversamp_factor`.
                Default is None.
            density_comp_weights: Optional tensor of precomputed density
                compensation weights. If provided, these are applied during the
                `adjoint` operation. Shape: (num_k_points,).
                Default is None.
            device: Computation device ('cpu' or 'cuda' or torch.device object).
                Default is 'cpu'.
        """
        # Determine kb_alpha if not provided
        final_kb_alpha: tuple[float, float]
        if kb_alpha is None:
            # Common heuristic for oversamp_factor = 2.0
            final_kb_alpha = (2.34 * kb_J[0], 2.34 * kb_J[1])
        else:
            final_kb_alpha = kb_alpha
        
        super().__init__(image_shape=image_shape, 
                         k_trajectory=k_trajectory, 
                         oversamp_factor=oversamp_factor, 
                         kb_J=kb_J, 
                         kb_alpha=final_kb_alpha,
                         kb_m=kb_m, 
                         Ld=Ld, 
                         Kd=Kd,
                         density_comp_weights=density_comp_weights,
                         device=device)

        if len(self.image_shape) != 2: # This check is also in parent, but good for explicitness
            raise ValueError(f"NUFFT2D expects a 2D image_shape, got {self.image_shape}")
        if self.k_trajectory.shape[-1] != 2:
            raise ValueError(f"NUFFT2D expects k_trajectory with last dimension 2, got {self.k_trajectory.shape}")

    def _kaiser_bessel_kernel(self, r: torch.Tensor) -> torch.Tensor:
        """
        Computes the generalized Kaiser-Bessel kernel for 2D (isotropic).
        Formula: (f^m * I_m(alpha*f)) / I_m(alpha) where f = sqrt(1 - (r/(J/2))^2).
        r: distance tensor |x|
        J: self.kb_J[0] (kernel width)
        alpha: self.kb_alpha[0] (shape parameter)
        m: self.kb_m[0] (order parameter)
        """
        J_dim = self.kb_J[0]
        alpha_dim = self.kb_alpha[0]
        m_dim = self.kb_m[0]

        # Create mask for r < J/2
        mask = r < (J_dim / 2.0)
        
        # Calculate f = sqrt(1 - (r / (J/2))^2) for r within J/2
        # Ensure values inside sqrt are non-negative due to potential floating point issues
        val_inside_sqrt = torch.clamp(1.0 - (2.0 * r[mask] / J_dim)**2, min=0.0)
        f = torch.sqrt(val_inside_sqrt)
        
        kb_vals = torch.zeros_like(r)

        # Numerator: (f^m * I_m(alpha*f))
        # Ensure f and alpha_dim * f are on CPU for scipy bessel functions
        f_cpu = f.cpu().numpy()
        alpha_f_cpu = alpha_dim * f_cpu
        
        if m_dim == 0.0:
            numerator_bessel_vals = scipy.special.i0(alpha_f_cpu)
            denominator_bessel_val = scipy.special.i0(alpha_dim)
        else:
            numerator_bessel_vals = scipy.special.iv(m_dim, alpha_f_cpu)
            denominator_bessel_val = scipy.special.iv(m_dim, alpha_dim)

        numerator_bessel_torch = torch.from_numpy(numerator_bessel_vals.astype(np.float32)).to(r.device)
        
        if m_dim == 0.0: # f^0 = 1
            numerator = numerator_bessel_torch
        else:
            numerator = (f**m_dim) * numerator_bessel_torch
            
        # Denominator: I_m(alpha)
        if np.isclose(denominator_bessel_val, 0.0): # Avoid division by zero if I_m(alpha) is zero
             # This case should ideally not happen with typical alpha values for m=0. For m > 0, iv(m,0)=0.
             # However, alpha_dim itself should not be zero with typical KB params.
            kb_vals[mask] = 0.0 # Or handle as an error / specific value
        else:
            kb_vals[mask] = numerator / float(denominator_bessel_val)
            
        return kb_vals

    def _estimate_density_compensation(self, kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
        radius = torch.sqrt(kx**2 + ky**2)
        dcf = radius + 1e-3 # Small epsilon to avoid issues with zero radius
        dcf /= dcf.max() # Normalize DCF to [epsilon, 1]
        return dcf

    def adjoint(self, kspace_data: torch.Tensor) -> torch.Tensor:
        """Applies the adjoint NUFFT operation (k-space to image domain).

        Transforms non-uniform k-space data to an image on a Cartesian grid.
        This operation is commonly referred to as gridding.

        If `density_comp_weights` were provided during initialization, they are
        multiplied with `kspace_data` before gridding. Otherwise, a simple
        internally estimated radial density compensation function is applied.

        Args:
            kspace_data: Input non-uniform k-space data tensor. Expected to be
                complex-valued.
                Shape: (num_k_points,), matching the number of points in
                `self.k_trajectory`.
                Device: Should match `self.device`.

        Returns:
            Tensor containing the reconstructed image data on a Cartesian grid.
            Complex-valued.
            Shape: (image_shape[0], image_shape[1]), matching `self.image_shape`.
            Device: `self.device`.
        """
        if kspace_data.ndim != 1: # Assuming k_trajectory is (num_k_points, 2)
            raise ValueError(f"Expected kspace_data to be 1D (num_k_points,), got shape {kspace_data.shape}")
        
        kspace_data = kspace_data.to(self.device) # Ensure data is on correct device
        # Ensure kspace_data is complex, as it should be for NUFFT input
        if not kspace_data.is_complex(): # This should ideally be handled by caller or checked more strictly
            kspace_data = kspace_data.to(torch.complex64)
            
        kx, ky = self.k_trajectory[:, 0], self.k_trajectory[:, 1]
        
        Nx, Ny = self.image_shape
        # Use self.Kd for oversampled grid dimensions
        Nx_oversamp, Ny_oversamp = self.Kd[0], self.Kd[1] 
        
        # Scale k-space coordinates to oversampled grid dimensions
        # kx, ky in [-0.5, 0.5] map to [0, Kd]
        kx_scaled = (kx + 0.5) * Nx_oversamp 
        ky_scaled = (ky + 0.5) * Ny_oversamp

        if self.density_comp_weights is not None:
            # self.density_comp_weights is already on self.device and float32
            kspace_data_weighted = kspace_data * self.density_comp_weights 
        else:
            dcf = self._estimate_density_compensation(kx, ky).to(self.device)
            kspace_data_weighted = kspace_data * dcf # Element-wise multiplication
        
        grid = torch.zeros((Nx_oversamp, Ny_oversamp), dtype=torch.complex64, device=self.device)
        weight_grid = torch.zeros((Nx_oversamp, Ny_oversamp), dtype=torch.float32, device=self.device)
        
        # Use self.kb_J for kernel width (assuming isotropic for 2D, so kb_J[0])
        half_width = self.kb_J[0] // 2 

        # Standard gridding: iterate kernel window around each k-space sample
        for dx_offset in range(-half_width, half_width + 1):
            for dy_offset in range(-half_width, half_width + 1):
                # gx_absolute_cell_index: absolute grid cell index we are calculating contribution TO.
                # dx_offset, dy_offset define which cell in the WxW neighborhood around kx_scaled.floor()
                # This is the cell (floor(k_s_x) + dx, floor(k_s_y) + dy)
                gx_absolute_cell_index = torch.floor(kx_scaled).long() + dx_offset
                gy_absolute_cell_index = torch.floor(ky_scaled).long() + dy_offset
                
                # dist_x_k_to_cell_center: distance from k-space sample to *center* of this grid cell
                dist_x_k_to_cell_center = kx_scaled - (gx_absolute_cell_index.float() + 0.5)
                dist_y_k_to_cell_center = ky_scaled - (gy_absolute_cell_index.float() + 0.5)
                
                r_for_kb = torch.sqrt(dist_x_k_to_cell_center**2 + dist_y_k_to_cell_center**2)
                kernel_weights = self._kaiser_bessel_kernel(r_for_kb)

                # Modulo for periodic boundary conditions on the oversampled grid
                gx_mod = gx_absolute_cell_index % Nx_oversamp
                gy_mod = gy_absolute_cell_index % Ny_oversamp
                
                # Flatten target indices for index_add_
                target_flat_indices = gx_mod * Ny_oversamp + gy_mod
                
                # Accumulate weighted k-space data and kernel weights
                grid.view(-1).index_add_(0, target_flat_indices, kspace_data_weighted * kernel_weights)
                weight_grid.view(-1).index_add_(0, target_flat_indices, kernel_weights)

        # Normalize grid by sum of weights
        weight_grid = torch.where(weight_grid == 0, torch.ones_like(weight_grid), weight_grid) # Avoid div by zero
        grid = grid / weight_grid
        
        # Inverse FFT, shift, and crop to original image size
        img = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(grid)))
        start_x, start_y = (Nx_oversamp - Nx) // 2, (Ny_oversamp - Ny) // 2
        img_cropped = img[start_x:start_x + Nx, start_y:start_y + Ny]
        
        # Scaling factor to approximate signal energy preservation (like sigpy, etc.)
        return img_cropped * float(Nx_oversamp * Ny_oversamp)


    def forward(self, image_data: torch.Tensor) -> torch.Tensor:
        """Applies the forward NUFFT operation (image domain to k-space).

        Transforms an image on a Cartesian grid to non-uniform k-space samples
        defined by the `k_trajectory`.

        Args:
            image_data: Input image data tensor. Expected to be complex-valued.
                Shape: (image_shape[0], image_shape[1]), matching `self.image_shape`.
                Device: Should match `self.device`.

        Returns:
            Tensor containing the simulated k-space data at the non-uniform
            `k_trajectory` points. Complex-valued.
            Shape: (num_k_points,).
            Device: `self.device`.
        """
        if image_data.shape != self.image_shape:
            raise ValueError(f"Input image_data shape {image_data.shape} does not match expected {self.image_shape}")

        image_data = image_data.to(self.device) # Ensure data is on correct device
        kx, ky = self.k_trajectory[:, 0], self.k_trajectory[:, 1]

        Nx, Ny = self.image_shape
        # Use self.Kd for oversampled grid dimensions
        Nx_oversamp, Ny_oversamp = self.Kd[0], self.Kd[1]

        # Pad image to oversampled size
        pad_x, pad_y = (Nx_oversamp - Nx) // 2, (Ny_oversamp - Ny) // 2
        image_padded = torch.zeros((Nx_oversamp, Ny_oversamp), dtype=torch.complex64, device=self.device)
        image_padded[pad_x:pad_x + Nx, pad_y:pad_y + Ny] = image_data
        
        # FFT of padded image
        kspace_cart_oversamp = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image_padded)))
        
        # Scale k-space coordinates (target non-Cartesian points on oversampled grid)
        kx_scaled = (kx + 0.5) * Nx_oversamp
        ky_scaled = (ky + 0.5) * Ny_oversamp
        
        # Use self.kb_J for kernel width
        half_width = self.kb_J[0] // 2 
        
        kspace_noncart = torch.zeros(kx.shape[0], dtype=torch.complex64, device=self.device)
        weight_sum = torch.zeros(kx.shape[0], dtype=torch.float32, device=self.device)

        # Standard interpolation: iterate kernel window around each target k-space sample
        for dx_offset in range(-half_width, half_width + 1):
            for dy_offset in range(-half_width, half_width + 1):
                # gx_source_absolute: absolute grid cell index on kspace_cart_oversamp we are reading FROM.
                # dx_offset, dy_offset define which cell in the WxW neighborhood around kx_scaled.floor()
                gx_source_absolute = torch.floor(kx_scaled).long() + dx_offset
                gy_source_absolute = torch.floor(ky_scaled).long() + dy_offset

                # dist_x_k_to_grid_center: distance from target k-space sample to *center* of this source grid cell
                dist_x_k_to_grid_center = kx_scaled - (gx_source_absolute.float() + 0.5)
                dist_y_k_to_grid_center = ky_scaled - (gy_source_absolute.float() + 0.5)

                r_for_kb = torch.sqrt(dist_x_k_to_grid_center**2 + dist_y_k_to_grid_center**2)
                kernel_interp_weights = self._kaiser_bessel_kernel(r_for_kb)

                # Modulo for periodic boundary conditions when reading from kspace_cart_oversamp
                gx_source_mod = gx_source_absolute % Nx_oversamp
                gy_source_mod = gy_source_absolute % Ny_oversamp
                
                kspace_noncart += kspace_cart_oversamp[gx_source_mod, gy_source_mod] * kernel_interp_weights
                weight_sum += kernel_interp_weights
        
        # Normalize by sum of weights
        weight_sum = torch.where(weight_sum == 0, torch.ones_like(weight_sum), weight_sum) # Avoid div by zero
        kspace_noncart /= weight_sum
        
        # Scaling factor (inverse of adjoint scaling)
        return kspace_noncart / float(Nx_oversamp * Ny_oversamp)


class NUFFT3D(NUFFT):
    def __init__(self, 
                 image_shape: tuple[int, int, int], 
                 k_trajectory: torch.Tensor, 
                 oversamp_factor: tuple[float, float, float] = (1.5, 1.5, 1.5),
                 kb_J: tuple[int, int, int] = (4, 4, 4),
                 kb_alpha: tuple[float, float, float] | None = None,
                 Ld: tuple[int, int, int] = (512, 512, 512),
                 kb_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 Kd: tuple[int, int, int] | None = None,
                 n_shift: tuple[float, float, float] | None = None,
                 interpolation_order: int = 1,
                 density_comp_weights: torch.Tensor | None = None,
                 device: str | torch.device = 'cpu'):
        """Initializes the 3D Non-Uniform Fast Fourier Transform (NUFFT) operator.

        This operator uses a table-based approach with Kaiser-Bessel interpolation
        for transforming data between non-uniform k-space samples and a
        Cartesian grid. It precomputes interpolation tables and scaling factors.

        Args:
            image_shape: Shape of the target image (Nz, Ny, Nx), e.g., (64, 64, 64).
            k_trajectory: Tensor of k-space trajectory coordinates, normalized to
                the range [-0.5, 0.5] in each dimension.
                Shape: (num_k_points, 3).
            oversamp_factor: Oversampling factor for the Cartesian grid.
                Default is (1.5, 1.5, 1.5).
            kb_J: Width of the Kaiser-Bessel interpolation kernel in grid units
                for each dimension. Default is (4, 4, 4).
            kb_alpha: Shape parameter for the Kaiser-Bessel kernel for each
                dimension. If None, automatically calculated as `2.34 * J_dim`.
                Default is None.
            Ld: Size of the lookup table for Kaiser-Bessel kernel interpolation
                for each dimension. Default is (512, 512, 512).
            kb_m: Order of the Kaiser-Bessel kernel for each dimension.
                Default is (0.0, 0.0, 0.0).
            Kd: Dimensions of the oversampled Cartesian grid (Kdz, Kdy, Kdx).
                If None, calculated as `image_shape * oversamp_factor`.
                Default is None.
            n_shift: Optional tuple (sz, sy, sx) specifying shifts in image
                domain samples. This translates to phase shifts in k-space.
                Useful for sub-pixel shifts or aligning field-of-view.
                Default is None (no shift).
            interpolation_order: Order for table interpolation.
                0 for Nearest Neighbor, 1 for Linear Interpolation.
                Default is 1 (Linear).
            density_comp_weights: Optional tensor of precomputed density
                compensation weights. Applied during the `adjoint` operation.
                Shape: (num_k_points,). Default is None.
            device: Computation device ('cpu', 'cuda', or torch.device object).
                Default is 'cpu'.
        """
        final_kb_alpha: tuple[float, float, float]
        if kb_alpha is None:
            final_kb_alpha = (2.34 * kb_J[0], 2.34 * kb_J[1], 2.34 * kb_J[2])
        else:
            final_kb_alpha = kb_alpha

        super().__init__(image_shape=image_shape, 
                         k_trajectory=k_trajectory, 
                         oversamp_factor=oversamp_factor, 
                         kb_J=kb_J, 
                         kb_alpha=final_kb_alpha,
                         kb_m=kb_m, 
                         Ld=Ld, 
                         Kd=Kd,
                         density_comp_weights=density_comp_weights,
                         device=device)

        if len(self.image_shape) != 3:
            raise ValueError(f"NUFFT3D expects a 3D image_shape, got {self.image_shape}")
        if self.k_trajectory.shape[-1] != 3:
            raise ValueError(f"NUFFT3D expects k_trajectory with last dimension 3, got {self.k_trajectory.shape}")

        if n_shift is None:
            self.n_shift = (0.0,) * len(self.image_shape)
        else:
            self.n_shift = tuple(n_shift)
            if len(self.n_shift) != len(self.image_shape):
                 raise ValueError(f"Length of n_shift {len(self.n_shift)} must match image dimensionality {len(self.image_shape)}")

        self.interpolation_order = interpolation_order
        if self.interpolation_order not in [0, 1]:
            raise ValueError("interpolation_order must be 0 (Nearest Neighbor) or 1 (Linear)")

        self.interp_tables: list[torch.Tensor] | None = None
        self.scaling_factors: torch.Tensor | None = None
        self.phase_shifts: torch.Tensor | None = None
        
        self._precompute_interpolation_tables()
        self._precompute_scaling_factors()

        if not all(s == 0.0 for s in self.n_shift):
            n_shift_tensor = torch.tensor(self.n_shift, dtype=torch.float32, device=self.device)
            k_traj_float = self.k_trajectory.float() 
            self.phase_shifts = torch.exp(1j * (k_traj_float @ n_shift_tensor))

    def _compute_kb_values_1d(self, r_vals: torch.Tensor, J: int, alpha: float, m: float) -> torch.Tensor:
        """
        Computes 1D generalized Kaiser-Bessel kernel values.
        r_vals: distances |x|, can be outside [-J/2, J/2].
        J: kernel width for this dimension
        alpha: shape parameter for this dimension
        m: order parameter for this dimension
        """
        kb_kernel_vals = torch.zeros_like(r_vals, dtype=torch.complex64) # Output complex
        
        # Mask for r_vals within kernel support abs(r_vals) <= J/2
        # Note: r_vals are distances, so they are non-negative.
        # The original MIRT code uses x for r_vals, which can be negative.
        # Here, if r_vals are |x|, then mask is r_vals <= J/2.
        # If r_vals are x, then mask is torch.abs(r_vals) <= (J / 2.0)
        # Assuming r_vals are actual coordinates x, not necessarily |x|.
        mask = torch.abs(r_vals) <= (J / 2.0)
        
        # f = sqrt(1 - (x / (J/2))^2) for x within support
        # We use r_vals[mask] to ensure we only compute for valid points.
        val_inside_sqrt = torch.clamp(1.0 - (r_vals[mask] / (J / 2.0))**2, min=0.0)
        f = torch.sqrt(val_inside_sqrt)
        
        # Numerator: (f^m * I_m(alpha*f))
        f_cpu = f.cpu().numpy()
        alpha_f_cpu = alpha * f_cpu # alpha is scalar
        
        if m == 0.0:
            numerator_bessel_vals_np = scipy.special.i0(alpha_f_cpu)
            denominator_bessel_val_np = scipy.special.i0(alpha)
        else:
            numerator_bessel_vals_np = scipy.special.iv(m, alpha_f_cpu)
            denominator_bessel_val_np = scipy.special.iv(m, alpha)

        numerator_bessel_torch = torch.from_numpy(numerator_bessel_vals_np.astype(np.float32)).to(r_vals.device)
        
        if m == 0.0: # f^0 = 1
            numerator = numerator_bessel_torch
        else:
            numerator = (f**m) * numerator_bessel_torch
            
        if np.isclose(float(denominator_bessel_val_np), 0.0):
            kb_kernel_vals[mask] = 0.0 # Or some other handling
        else:
            kb_kernel_vals[mask] = numerator / float(denominator_bessel_val_np)
            
        return kb_kernel_vals

    def _precompute_interpolation_tables(self):
        dd = len(self.image_shape)
        self.interp_tables = []
        for d_idx in range(dd):
            J_d = self.kb_J[d_idx]
            alpha_d = self.kb_alpha[d_idx]
            m_d = self.kb_m[d_idx]
            L_d = self.Ld[d_idx]
            
            # Table query points for one dimension from -J_d/2 to J_d/2
            # These are the 'x' arguments (distances from center) for the KB kernel
            table_query_points = torch.linspace(-J_d / 2.0, J_d / 2.0, steps=J_d * L_d + 1, device=self.device)
            
            h_d = self._compute_kb_values_1d(table_query_points, J_d, alpha_d, m_d)
            self.interp_tables.append(h_d) # Already complex from _compute_kb_values_1d

    def _kaiser_bessel_ft_1d(self, u: torch.Tensor, J: int, alpha: float, m: float) -> torch.Tensor:
        """
        Computes the 1D Fourier Transform of the Kaiser-Bessel kernel.
        u: normalized frequency arguments (u_d / Kd_d)
        J, alpha, m: kernel parameters for this dimension
        Returns real-valued FT.
        """
        # z = sqrt( (2*pi*(J/2)*u)^2 - alpha^2 )
        # Add small imaginary epsilon to sqrt argument to handle negative real parts correctly (for complex z)
        z_arg_sq = (2 * np.pi * (J / 2.0) * u)**2 - alpha**2
        z = torch.sqrt(z_arg_sq.to(torch.complex64) + 1e-12j) # Ensure complex sqrt
        
        nu = 0.5 + m

        # Denominator I_m(alpha)
        if m == 0.0:
            den_bessel_val = scipy.special.i0(alpha)
        else:
            den_bessel_val = scipy.special.iv(m, alpha)
        
        if np.isclose(float(den_bessel_val), 0.0):
            # This should not happen with typical alpha > 0
            return torch.zeros_like(u, dtype=torch.float32)

        # Term J_nu(z) / z^nu
        # Handle z=0 case
        # J_nu(z) / z^nu -> 1 / (2^nu * Gamma(nu+1)) as z -> 0
        # J_0(0) = 1. So for nu=0, J_0(z)/z^0 = J_0(z), at z=0, this is 1.
        # Limit for nu=0: 1 / (2^0 * Gamma(1)) = 1.
        
        z_cpu_numpy = z.cpu().numpy() # Scipy works on numpy arrays
        jn_z_val_np = scipy.special.jv(nu, z_cpu_numpy)
        
        # Ratio J_nu(z) / z^nu
        ratio_val = torch.zeros_like(z, dtype=torch.complex64)
        
        # Non-zero z
        mask_z_nonzero = torch.abs(z) > 1e-9 # Threshold for non-zero
        if torch.any(mask_z_nonzero):
            ratio_val[mask_z_nonzero] = torch.from_numpy(
                jn_z_val_np[mask_z_nonzero.cpu().numpy()]
            ).to(z.device) / (z[mask_z_nonzero]**nu)

        # Zero z
        mask_z_zero = ~mask_z_nonzero
        if torch.any(mask_z_zero):
            limit_val = 1.0 / ( (2**nu) * math.gamma(nu + 1) )
            ratio_val[mask_z_zero] = limit_val
            
        # Main formula: Y(u) = (2*pi)^(1/2) * (J/2) * alpha^m / I_m(alpha) * ratio_val
        # MIRT formula seems to be (2*pi)^(d/2) * prod_d( (Jd/2) * alpha_d^m_d / I_m_d(alpha_d) * ratio_d )
        # For 1D, d=1.
        const_factor = np.sqrt(2 * np.pi) * (J / 2.0) * (alpha**m) / float(den_bessel_val)
        ft_vals_complex = const_factor * ratio_val
        
        return ft_vals_complex.real # FT of KB kernel is real

    def _precompute_scaling_factors(self):
        dd = len(self.image_shape)
        s_factors_list_1d = []

        for d_idx in range(dd):
            J_d = self.kb_J[d_idx]
            alpha_d = self.kb_alpha[d_idx]
            m_d = self.kb_m[d_idx]
            Kd_d = self.Kd[d_idx]
            Nd_d = self.image_shape[d_idx]

            # Frequency coordinates for this dimension: -(Nd/2) to (Nd/2)-1 or similar
            # MIRT uses: nc = [0:N-1] - N/2; if N is even, range is -N/2 to N/2-1
            # if N is odd, range is -(N-1)/2 to (N-1)/2
            # torch.arange(Nd_d) - (Nd_d -1)/2.0 gives correct center for both even/odd
            u_d_grid = torch.arange(Nd_d, device=self.device) - (Nd_d - 1.0) / 2.0
            u_d_normalized = u_d_grid / Kd_d # Normalized by oversampled grid size Kd

            ft_kb_d = self._kaiser_bessel_ft_1d(u_d_normalized, J_d, alpha_d, m_d)
            
            # Inverse, handle potential division by zero if ft_kb_d can be zero
            # (though FT of KB is generally non-zero in passband)
            scaling_factor_1d = torch.where(
                torch.abs(ft_kb_d) < 1e-9, # Threshold for effectively zero
                torch.zeros_like(ft_kb_d), 
                1.0 / ft_kb_d
            ).to(torch.complex64) # Ensure complex type
            s_factors_list_1d.append(scaling_factor_1d)

        # Combine 1D scaling factors using broadcasting via meshgrid
        # grids will be a list of tensors, each having shape like (N1,1,1), (1,N2,1), (1,1,N3) for 3D
        # This is for 'ij' indexing. For 'xy' (Cartesian), shapes would be different.
        # MIRT's st.sn is Nd1 x Nd2 x Nd3.
        reshaped_factors = []
        for i, sf_1d in enumerate(s_factors_list_1d):
            new_shape = [1] * dd
            new_shape[i] = self.image_shape[i]
            reshaped_factors.append(sf_1d.view(new_shape))
        
        # Element-wise product via broadcasting
        self.scaling_factors = reshaped_factors[0]
        for i in range(1, dd):
            self.scaling_factors = self.scaling_factors * reshaped_factors[i]
        
        # Ensure final scaling_factors has shape self.image_shape and is complex
        self.scaling_factors = self.scaling_factors.reshape(self.image_shape).to(torch.complex64)


    @abc.abstractmethod
    def _lookup_1d_table(self, table: torch.Tensor, relative_offset_grid_units: torch.Tensor, L_d: int) -> torch.Tensor:
        """
        Performs 1D linear interpolation on a precomputed table.
        Args:
            table: The 1D interpolation table (complex tensor).
            relative_offset_grid_units: Fractional offset from the nearest grid point, in grid units.
                                         Shape should be compatible for broadcasting with table lookups.
            L_d: Table oversampling factor for this dimension.
        Returns:
            Interpolated values (complex tensor).
        """
        if not table.is_complex():
            # This should not happen if interp_tables are complex
            table = table.to(torch.complex64)

        table_len = table.shape[0]
        table_center_idx = (table_len - 1) / 2.0

        # Calculate floating point index into the table
        # relative_offset_grid_units is (delta_d - j_offset_d)
        # This value represents the distance from the true k-space sample (tm_d) to the
        # (nearest_grid_idx_d + j_offset_d)-th grid point, which is where the kernel tap is centered.
        # The table itself is indexed by (distance_from_kernel_center * L_d).
        # So, if relative_offset_grid_units is this distance, then the index is center + this_dist * L_d.
        table_idx_float = table_center_idx + relative_offset_grid_units * L_d

        if self.interpolation_order == 0: # Nearest Neighbor
            nearest_idx = torch.round(table_idx_float).long()
            nearest_idx = torch.clamp(nearest_idx, 0, table_len - 1)
            interpolated_val = table[nearest_idx]
        elif self.interpolation_order == 1: # Linear Interpolation
            idx_low = torch.floor(table_idx_float).long()
            idx_high = torch.ceil(table_idx_float).long()
            
            # Fractional part for interpolation
            frac = table_idx_float - idx_low.float()

            # Boundary conditions: clamp indices to table bounds
            idx_low = torch.clamp(idx_low, 0, table_len - 1)
            idx_high = torch.clamp(idx_high, 0, table_len - 1)

            val_low = table[idx_low]
            val_high = table[idx_high]
            
            # Perform interpolation: (1-frac)*val_low + frac*val_high
            interpolated_val = (1.0 - frac) * val_low + frac * val_high
        else:
            # This case should be caught by __init__ validation
            raise ValueError(f"Unsupported interpolation_order: {self.interpolation_order}")
            
        return interpolated_val

    def forward(self, image_data: torch.Tensor) -> torch.Tensor:
        """Applies the forward 3D NUFFT (image domain to k-space).

        Transforms a 3D image on a Cartesian grid to non-uniform k-space
        samples defined by `self.k_trajectory`. Uses precomputed scaling
        factors and table-based interpolation with Kaiser-Bessel kernels.
        Applies phase shifts if `n_shift` was specified during initialization.

        Args:
            image_data: Input 3D image data tensor. Expected to be complex-valued.
                Shape: (image_shape[0], image_shape[1], image_shape[2]),
                matching `self.image_shape`.
                Device: Should match `self.device`.

        Returns:
            Tensor containing simulated k-space data at `self.k_trajectory` points.
            Complex-valued.
            Shape: (num_k_points,).
            Device: `self.device`.
        """
        # 1. Input Validation
        if image_data.shape != self.image_shape:
            raise ValueError(f"Input image_data shape {image_data.shape} must match NUFFT image_shape {self.image_shape}")
        if image_data.device != self.device:
            image_data = image_data.to(self.device)
        if not image_data.is_complex():
            image_data = image_data.to(torch.complex64)
        
        # 2. Apply Scaling Factors (Image Domain)
        if self.scaling_factors is None:
            raise RuntimeError("Scaling factors not precomputed. Call _precompute_scaling_factors first.")
        scaled_image = image_data * self.scaling_factors

        # 3. Oversampled FFT
        # Pad to self.Kd if image_shape is smaller, then FFT
        pad_amount = []
        for kd_dim, im_dim in zip(self.Kd, self.image_shape):
            pad_before = (kd_dim - im_dim) // 2
            pad_after = kd_dim - im_dim - pad_before
            pad_amount.extend([pad_before, pad_after])
        
        # PyTorch padding format is (pad_dimN_start, pad_dimN_end, pad_dimN-1_start, ...)
        # We need to reverse it for torch.nn.functional.pad
        # For 3D: (pad_dim2_start, pad_dim2_end, pad_dim1_start, pad_dim1_end, pad_dim0_start, pad_dim0_end)
        torch_pad_format = []
        for i in range(len(self.image_shape) -1, -1, -1): # Iterate dims from last to first
            torch_pad_format.extend([pad_amount[2*i], pad_amount[2*i+1]])

        padded_scaled_image = torch.nn.functional.pad(scaled_image, torch_pad_format, mode='constant', value=0)
        
        Xk_grid = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(padded_scaled_image), s=self.Kd))

        # 4. Prepare for Interpolation
        dd = len(self.image_shape) # Should be 3
        if dd != 3: # Should have been caught by __init__ but good to be safe
            raise ValueError(f"NUFFT3D forward method expects 3D data, image_shape has {dd} dimensions.")

        num_k_points = self.k_trajectory.shape[0]
        interpolated_k_space_values = torch.zeros(num_k_points, dtype=torch.complex64, device=self.device)
        interpolated_k_space_weights = torch.zeros(num_k_points, dtype=torch.float32, device=self.device) # For sum of kernel weights

        om = self.k_trajectory # Shape (num_k_points, dd)

        # 5. Coordinate Scaling (tm)
        scaled_coords_tm = [] # List to store tm_d for each dimension
        for d_idx in range(dd):
            gamma_d = 2 * np.pi / self.Kd[d_idx]
            # om already on self.device, ensure float for division safety
            tm_d = om[:, d_idx].float() / gamma_d 
            scaled_coords_tm.append(tm_d)

        # 6. Table-Based Interpolation
        if self.interp_tables is None:
             raise RuntimeError("Interpolation tables not precomputed. Call _precompute_interpolation_tables first.")

        # Nearest grid points and fractional offsets for all k-points (vectorized)
        k_nearest_all_kpoints = [] # List of tensors [k_nearest_x_all, k_nearest_y_all, k_nearest_z_all]
        delta_all_kpoints = []   # List of tensors [delta_x_all, delta_y_all, delta_z_all]
        for d_idx in range(dd):
            current_tm_d_all_kpoints = scaled_coords_tm[d_idx] # (num_k_points,)
            k_nearest_d = torch.round(current_tm_d_all_kpoints) # .long() later for indexing
            delta_d = current_tm_d_all_kpoints - k_nearest_d
            k_nearest_all_kpoints.append(k_nearest_d.long())
            delta_all_kpoints.append(delta_d)

        # Kernel tap iteration loops (innermost part of the MIRT interp)
        # These loops iterate over Jx * Jy * Jz taps for the kernel
        # For each k-point, we sum contributions from these Jx*Jy*Jz grid points around it.
        
        # Create indices for J taps for each dimension
        # j_offsets_dim[d] will be a tensor of offsets: [-Jd/2, ..., Jd/2-1] or similar
        j_offsets_dim = []
        for d_idx in range(dd):
            J_d = self.kb_J[d_idx]
            # Offsets from kernel center, e.g., for J=4: -2, -1, 0, 1. For J=5: -2, -1, 0, 1, 2
            # MIRT typically uses j = 0..J-1, and then offset by -J/2 or similar.
            # Let's use offsets directly: range from -floor(J/2) to ceil(J/2)-1 or similar
            # A simple way: j_coords = torch.arange(J_d, device=self.device) - J_d // 2
            # This gives for J=4: [-2, -1, 0, 1]. For J=5: [-2, -1, 0, 1, 2] (center at index 2)
            j_coords = torch.arange(-(J_d // 2), (J_d + 1) // 2, device=self.device) # e.g. J=4 -> -2,-1,0,1. J=5 -> -2,-1,0,1,2
            j_offsets_dim.append(j_coords)

        # Iterate over all k-points (can be slow, but per instructions)
        for m_idx in range(num_k_points):
            current_val_sum = torch.tensor(0.0, dtype=torch.complex64, device=self.device)
            current_weight_sum = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            # Get nearest grid point and delta for the current k-point m_idx
            k_nearest_m = [k_nearest_all_kpoints[d][m_idx] for d in range(dd)] # [k_near_x_m, k_near_y_m, k_near_z_m]
            delta_m = [delta_all_kpoints[d][m_idx] for d in range(dd)]         # [delta_x_m, delta_y_m, delta_z_m]

            # Nested loops for Jz, Jy, Jx kernel taps
            for jz_offset in j_offsets_dim[2]:
                abs_gz = k_nearest_m[2] + jz_offset
                # Relative offset for table lookup: delta_z - jz_offset
                # This is distance from k-space sample (tm_z) to this specific grid tap (k_nearest_z + jz_offset)
                kernel_val_z = self._lookup_1d_table(self.interp_tables[2], delta_m[2] - jz_offset, self.Ld[2])

                for jy_offset in j_offsets_dim[1]:
                    abs_gy = k_nearest_m[1] + jy_offset
                    kernel_val_y = self._lookup_1d_table(self.interp_tables[1], delta_m[1] - jy_offset, self.Ld[1])

                    for jx_offset in j_offsets_dim[0]:
                        abs_gx = k_nearest_m[0] + jx_offset
                        kernel_val_x = self._lookup_1d_table(self.interp_tables[0], delta_m[0] - jx_offset, self.Ld[0])
                        
                        effective_kernel_weight = kernel_val_x * kernel_val_y * kernel_val_z
                        
                        # Apply modulo for periodic boundary conditions on Xk_grid
                        gz_mod = abs_gz % self.Kd[2]
                        gy_mod = abs_gy % self.Kd[1]
                        gx_mod = abs_gx % self.Kd[0]
                        
                        grid_val_from_Xk = Xk_grid[gz_mod, gy_mod, gx_mod]
                        
                        current_val_sum += grid_val_from_Xk * effective_kernel_weight
                        current_weight_sum += effective_kernel_weight.real # As per MIRT for complex kernel

            if torch.abs(current_weight_sum) > 1e-9: # Avoid division by zero
                interpolated_k_space_values[m_idx] = current_val_sum / current_weight_sum
            else:
                interpolated_k_space_values[m_idx] = 0.0 # Or handle as error/NaN
        
        # 7. Apply Phase Shift
        if self.phase_shifts is not None:
            interpolated_k_space_values = interpolated_k_space_values * self.phase_shifts

        # 8. Return
        return interpolated_k_space_values

    def adjoint(self, kspace_data: torch.Tensor) -> torch.Tensor:
        """Applies the adjoint 3D NUFFT (k-space to image domain).

        Transforms non-uniform 3D k-space data to an image on a Cartesian grid.
        This operation involves table-based gridding with Kaiser-Bessel kernels
        and application of conjugate scaling factors.

        If `density_comp_weights` were provided during `__init__`, they are
        multiplied with `kspace_data` before gridding.
        If `n_shift` was specified, corresponding conjugate phase shifts are
        applied to `kspace_data`.

        Args:
            kspace_data: Input non-uniform 3D k-space data. Complex-valued.
                Shape: (num_k_points,), matching `self.k_trajectory`.
                Device: Should match `self.device`.

        Returns:
            Reconstructed 3D image data on a Cartesian grid. Complex-valued.
            Shape: (image_shape[0], image_shape[1], image_shape[2]),
            matching `self.image_shape`.
            Device: `self.device`.
        """
        # 1. Input Validation
        if kspace_data.ndim != 1:
            raise ValueError(f"Input kspace_data must be a 1D tensor, got shape {kspace_data.shape}")
        if kspace_data.shape[0] != self.k_trajectory.shape[0]:
            raise ValueError(f"Input kspace_data shape {kspace_data.shape[0]} must match k_trajectory points {self.k_trajectory.shape[0]}")
        if kspace_data.device != self.device:
            kspace_data = kspace_data.to(self.device)
        if not kspace_data.is_complex(): # Ensure input is complex
            kspace_data = kspace_data.to(torch.complex64)

        # 2. Apply Density Compensation (if provided)
        if self.density_comp_weights is not None:
            # self.density_comp_weights is already on self.device and float32
            kspace_data_processed = kspace_data * self.density_comp_weights
        else:
            kspace_data_processed = kspace_data # No DCF applied if not provided
        
        # 3. Apply Adjoint Phase Shift
        if self.phase_shifts is not None:
            phase_adjusted_kspace_data = kspace_data_processed * self.phase_shifts.conj()
        else:
            phase_adjusted_kspace_data = kspace_data_processed

        # 4. Prepare for Gridding
        dd = len(self.image_shape) # Should be 3
        num_k_points = self.k_trajectory.shape[0]
        gridded_k_space = torch.zeros(self.Kd, dtype=torch.complex64, device=self.device)
        om = self.k_trajectory

        # 5. Coordinate Scaling (tm) & Nearest Grid Points/Deltas (vectorized over k-points)
        scaled_coords_tm = []
        k_nearest_all_kpoints = []
        delta_all_kpoints = []
        for d_idx in range(dd):
            gamma_d = 2 * np.pi / self.Kd[d_idx]
            tm_d_all = om[:, d_idx].float() / gamma_d
            scaled_coords_tm.append(tm_d_all)
            
            k_nearest_d_all = torch.round(tm_d_all)
            delta_d_all = tm_d_all - k_nearest_d_all
            k_nearest_all_kpoints.append(k_nearest_d_all.long())
            delta_all_kpoints.append(delta_d_all)

        # 6. Table-Based Gridding
        if self.interp_tables is None:
             raise RuntimeError("Interpolation tables not precomputed. Call _precompute_interpolation_tables first.")

        j_offsets_dim = []
        for d_idx in range(dd):
            J_d = self.kb_J[d_idx]
            j_coords = torch.arange(-(J_d // 2), (J_d + 1) // 2, device=self.device)
            j_offsets_dim.append(j_coords)
        
        # Loop over k-points
        for m_idx in range(num_k_points):
            current_k_sample_val = phase_adjusted_kspace_data[m_idx]
            if torch.abs(current_k_sample_val) < 1e-15: # Optimization: skip if sample is effectively zero
                continue

            k_nearest_m = [k_nearest_all_kpoints[d][m_idx] for d in range(dd)]
            delta_m = [delta_all_kpoints[d][m_idx] for d in range(dd)]

            # Innermost 3D loop for kernel taps
            for jz_offset in j_offsets_dim[2]:
                abs_gz = k_nearest_m[2] + jz_offset
                kernel_val_z = self._lookup_1d_table(self.interp_tables[2], delta_m[2] - jz_offset, self.Ld[2])

                for jy_offset in j_offsets_dim[1]:
                    abs_gy = k_nearest_m[1] + jy_offset
                    kernel_val_y = self._lookup_1d_table(self.interp_tables[1], delta_m[1] - jy_offset, self.Ld[1])
                    
                    # Potential optimization: if kernel_val_z * kernel_val_y is too small, skip inner loop
                    # if torch.abs(kernel_val_z * kernel_val_y) < 1e-9: continue 

                    for jx_offset in j_offsets_dim[0]:
                        abs_gx = k_nearest_m[0] + jx_offset
                        kernel_val_x = self._lookup_1d_table(self.interp_tables[0], delta_m[0] - jx_offset, self.Ld[0])
                        
                        effective_kernel_weight = kernel_val_x * kernel_val_y * kernel_val_z
                        
                        # if torch.abs(effective_kernel_weight) < 1e-9: continue # Optimization

                        value_to_add = current_k_sample_val * effective_kernel_weight.conj()
                        
                        # Modulo for periodic boundary conditions
                        gz_mod = abs_gz % self.Kd[2]
                        gy_mod = abs_gy % self.Kd[1]
                        gx_mod = abs_gx % self.Kd[0]
                        
                        gridded_k_space[gz_mod, gy_mod, gx_mod] += value_to_add
        
        # 7. Inverse FFT
        # Scaling by prod(Kd) is common for adjoint NUFFT to match forward scaling
        image_oversampled = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(gridded_k_space), s=self.Kd))
        image_oversampled = image_oversampled * float(torch.prod(torch.tensor(self.Kd, dtype=torch.float32)))

        # 8. Cropping
        start_indices = [(kd_dim - self.image_shape[d_idx]) // 2 for d_idx, kd_dim in enumerate(self.Kd)]
        
        image_cropped = image_oversampled[
            start_indices[0] : start_indices[0] + self.image_shape[0],
            start_indices[1] : start_indices[1] + self.image_shape[1],
            start_indices[2] : start_indices[2] + self.image_shape[2]
        ]

        # 9. Apply Conjugate Scaling Factors (Image Domain)
        if self.scaling_factors is None: # Should have been caught by forward, but good practice
            raise RuntimeError("Scaling factors not precomputed.")
        final_image = image_cropped * self.scaling_factors.conj()
        
        # 10. Return
        return final_image
