"""Module for defining Operator classes for MRI reconstruction."""

import torch
import numpy as np
import math # Added for IRadon
import torch.nn.functional as F # Added for IRadon, though not used in current IRadon draft
from abc import ABC, abstractmethod
from scipy.signal.windows import tukey # For SlidingWindowNUFFTOperator
from reconlib.nufft import NUFFT2D, NUFFT3D


# Operator Base Class
class Operator(ABC):
    @abstractmethod
    def op(self, x): pass
    @abstractmethod
    def op_adj(self, y): pass

class NUFFTOperator(Operator):
    """
    NUFFT Operator that wraps NUFFT2D or NUFFT3D table-based implementations,
    or uses a direct NDFT for 3D.
    k_trajectory coordinates are assumed to be normalized in [-0.5, 0.5] for each dimension.
    """
    def __init__(self, 
                 k_trajectory: torch.Tensor, 
                 image_shape: tuple[int, ...], 
                 oversamp_factor: tuple[float, ...], 
                 kb_J: tuple[int, ...], 
                 kb_alpha: tuple[float, ...], 
                 Ld: tuple[int, ...], 
                 kb_m: tuple[float, ...] | None = None, 
                 Kd: tuple[int, ...] | None = None, 
                 n_shift: tuple[float, ...] | None = None, 
                 device: str | torch.device = 'cpu', 
                 nufft_type_3d: str = 'table'):
        
        self.image_shape = tuple(image_shape)
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if not isinstance(k_trajectory, torch.Tensor):
            k_trajectory = torch.tensor(k_trajectory, dtype=torch.float32)
        self.k_trajectory = k_trajectory.to(self.device)

        self.dimensionality = len(image_shape)
        self.nufft_impl = None
        self.grid_flat_3d = None 
        self.nufft_type_3d = nufft_type_3d # Store for 3D case

        if self.dimensionality == 2:
            if self.k_trajectory.ndim != 2 or self.k_trajectory.shape[1] != 2:
                raise ValueError(f"For 2D NUFFT, k_trajectory must have shape (num_k_points, 2), got {self.k_trajectory.shape}")
            self.nufft_impl = NUFFT2D(image_shape=self.image_shape, 
                                      k_trajectory=self.k_trajectory, 
                                      oversamp_factor=oversamp_factor, 
                                      kb_J=kb_J, 
                                      kb_alpha=kb_alpha, 
                                      kb_m=kb_m, 
                                      Ld=Ld, 
                                      Kd=Kd, 
                                      device=self.device)
        elif self.dimensionality == 3:
            if self.k_trajectory.ndim != 2 or self.k_trajectory.shape[1] != 3:
                raise ValueError(f"For 3D, k_trajectory must have shape (num_k_points, 3), got {self.k_trajectory.shape}")
            
            if self.nufft_type_3d == 'table':
                self.nufft_impl = NUFFT3D(image_shape=self.image_shape, 
                                          k_trajectory=self.k_trajectory, 
                                          oversamp_factor=oversamp_factor, 
                                          kb_J=kb_J, 
                                          kb_alpha=kb_alpha, 
                                          kb_m=kb_m, 
                                          Ld=Ld, 
                                          Kd=Kd, 
                                          n_shift=n_shift, 
                                          device=self.device)
            elif self.nufft_type_3d == 'direct':
                self.nufft_impl = None # Signal to use direct NDFT
                coords_z = torch.linspace(-0.5, 0.5, image_shape[0], device=self.device, dtype=torch.float32)
                coords_y = torch.linspace(-0.5, 0.5, image_shape[1], device=self.device, dtype=torch.float32)
                coords_x = torch.linspace(-0.5, 0.5, image_shape[2], device=self.device, dtype=torch.float32)
                grid_z, grid_y, grid_x = torch.meshgrid(coords_z, coords_y, coords_x, indexing='ij')
                self.grid_flat_3d = torch.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), dim=1)
                if n_shift is not None and not all(s == 0.0 for s in n_shift):
                    print("Warning: n_shift is provided for 'direct' 3D NUFFT but currently not implemented for it. It will be ignored.")
            else:
                raise ValueError(f"Unknown nufft_type_3d: {self.nufft_type_3d}. Must be 'table' or 'direct'.")
        else:
            raise ValueError(f"Unsupported dimensionality: {self.dimensionality}. Must be 2 or 3.")

    def op(self, image_data_tensor):
        image_data_tensor = torch.as_tensor(image_data_tensor, dtype=torch.complex64, device=self.device)
        if image_data_tensor.shape != self.image_shape:
             raise ValueError(f"Input image_data_tensor shape {image_data_tensor.shape} does not match expected {self.image_shape}")

        if self.nufft_impl is not None: # Covers 2D and 3D 'table'
            return self.nufft_impl.forward(image_data_tensor)
        elif self.dimensionality == 3 and self.nufft_type_3d == 'direct':
            if self.grid_flat_3d is None:
                 raise RuntimeError("grid_flat_3d not initialized for direct 3D NUFFT.")
            image_flat = image_data_tensor.flatten().unsqueeze(0) 
            dot_product_matrix = torch.matmul(self.k_trajectory, self.grid_flat_3d.T)
            exponent_matrix = -2j * torch.pi * dot_product_matrix
            kspace_data = torch.sum(image_flat * torch.exp(exponent_matrix), dim=1)
            return kspace_data
        else: 
            raise RuntimeError(f"NUFFT operation not supported for dimensionality {self.dimensionality} and type {self.nufft_type_3d}")

    def op_adj(self, k_space_data_tensor, output_voxel_coords_flat=None):
        k_space_data_tensor = torch.as_tensor(k_space_data_tensor, dtype=torch.complex64, device=self.device)
        
        if self.nufft_impl is not None: # Covers 2D and 3D 'table'
            if output_voxel_coords_flat is not None:
                print("Warning: NUFFTOperator.op_adj with table-based NUFFT implementation does not support output_voxel_coords_flat. It will be ignored.")
            return self.nufft_impl.adjoint(k_space_data_tensor)
        elif self.dimensionality == 3 and self.nufft_type_3d == 'direct':
            if self.grid_flat_3d is None:
                 raise RuntimeError("grid_flat_3d not initialized for direct 3D NUFFT adjoint.")
            k_space_data_expanded = k_space_data_tensor.unsqueeze(1)  
            
            grid_to_use = self.grid_flat_3d
            reshape_output = True
            if output_voxel_coords_flat is not None:
                if not isinstance(output_voxel_coords_flat, torch.Tensor) or output_voxel_coords_flat.ndim != 2 or output_voxel_coords_flat.shape[1] != 3:
                        raise ValueError("output_voxel_coords_flat must be a 2D tensor of shape (num_voxels, 3).")
                grid_to_use = output_voxel_coords_flat.to(device=self.device, dtype=torch.float32)
                reshape_output = False # Output will be flat vector matching output_voxel_coords_flat

            dot_product_matrix = torch.matmul(self.k_trajectory, grid_to_use.T) 
            exponent_matrix = 2j * torch.pi * dot_product_matrix 
            image_flat = torch.sum(k_space_data_expanded * torch.exp(exponent_matrix), dim=0) 
            
            if reshape_output:
                return image_flat.reshape(self.image_shape)
            else:
                return image_flat # Return flat vector
        else:
            raise RuntimeError(f"NUFFT adjoint operation not supported for dimensionality {self.dimensionality} and type {self.nufft_type_3d}")

class CoilSensitivityOperator(Operator):
    def __init__(self, coil_sensitivities_tensor):
        if not isinstance(coil_sensitivities_tensor, torch.Tensor):
            coil_sensitivities_tensor = torch.tensor(coil_sensitivities_tensor, dtype=torch.complex64)
        self.coil_sensitivities_tensor = coil_sensitivities_tensor
        self.device = self.coil_sensitivities_tensor.device 
    def op(self, image_data_tensor):
        image_data_tensor = torch.as_tensor(image_data_tensor, dtype=torch.complex64, device=self.device)
        return image_data_tensor.unsqueeze(0) * self.coil_sensitivities_tensor
    def op_adj(self, coil_images_data_tensor):
        coil_images_data_tensor = torch.as_tensor(coil_images_data_tensor, dtype=torch.complex64, device=self.device)
        return torch.sum(coil_images_data_tensor * torch.conj(self.coil_sensitivities_tensor), dim=0)

class MRIForwardOperator(Operator):
    def __init__(self, nufft_operator: NUFFTOperator, coil_operator: CoilSensitivityOperator = None, num_coils_if_no_sens: int = None):
        self.nufft_operator = nufft_operator
        self.coil_operator = coil_operator
        self.num_coils_if_no_sens = num_coils_if_no_sens
        self.device = nufft_operator.device 
        self.image_shape = nufft_operator.image_shape 
        if self.coil_operator is None and self.num_coils_if_no_sens is None:
            raise ValueError("If coil_operator is None, num_coils_if_no_sens must be provided.")
    def op(self, image_data_tensor):
        image_data_tensor = torch.as_tensor(image_data_tensor, dtype=torch.complex64, device=self.device)
        if image_data_tensor.shape != self.image_shape:
             raise ValueError(f"Input image_data_tensor shape {image_data_tensor.shape} does not match expected {self.image_shape}")
        if self.coil_operator is None:
            k_per_coil = self.nufft_operator.op(image_data_tensor) 
            return k_per_coil.unsqueeze(0).expand(self.num_coils_if_no_sens, -1)
        else:
            coil_output = self.coil_operator.op(image_data_tensor) 
            num_coils = coil_output.shape[0]
            num_k_points = self.nufft_operator.k_trajectory.shape[0] 
            k_space_result = torch.zeros((num_coils, num_k_points), dtype=coil_output.dtype, device=self.device)
            for c in range(num_coils):
                k_space_result[c, :] = self.nufft_operator.op(coil_output[c])
            return k_space_result
    def op_adj(self, k_space_data_coils_tensor):
        k_space_data_coils_tensor = torch.as_tensor(k_space_data_coils_tensor, dtype=torch.complex64, device=self.device)
        if self.coil_operator is None:
            num_coils_from_data = k_space_data_coils_tensor.shape[0]
            if self.num_coils_if_no_sens is not None and num_coils_from_data != self.num_coils_if_no_sens:
                raise ValueError(f"Mismatch between k_space_data_coils_tensor.shape[0] ({num_coils_from_data}) and num_coils_if_no_sens ({self.num_coils_if_no_sens})")
            accumulated_image = torch.zeros(self.image_shape, dtype=k_space_data_coils_tensor.dtype, device=self.device)
            for c in range(num_coils_from_data): 
                accumulated_image += self.nufft_operator.op_adj(k_space_data_coils_tensor[c])
            return accumulated_image
        else:
            num_coils = k_space_data_coils_tensor.shape[0]
            coil_adj_output_shape = (num_coils,) + self.image_shape
            coil_adj_output = torch.zeros(coil_adj_output_shape, dtype=k_space_data_coils_tensor.dtype, device=self.device)
            for c in range(num_coils):
                coil_adj_output[c, ...] = self.nufft_operator.op_adj(k_space_data_coils_tensor[c])
            return self.coil_operator.op_adj(coil_adj_output)

# --- Sliding Window NUFFT Operator ---
class SlidingWindowNUFFTOperator(Operator):
    """
    Wraps a NUFFTOperator to perform memory-efficient adjoint operation for 3D direct NDFT
    by reconstructing the image in overlapping blocks.
    The forward operation currently calls the base operator directly.
    """
    def __init__(self, base_nufft_operator: NUFFTOperator, block_size: tuple, overlap_ratio=0.25):
        if not isinstance(base_nufft_operator, NUFFTOperator):
            raise TypeError("base_nufft_operator must be an instance of NUFFTOperator.")
        
        self.base_nufft_operator = base_nufft_operator
        self.full_image_shape = self.base_nufft_operator.image_shape
        self.dimensionality = self.base_nufft_operator.dimensionality
        self.device = self.base_nufft_operator.device
        
        if self.dimensionality != 3:
            print("Warning: SlidingWindowNUFFTOperator is primarily designed for 3D. May not offer benefits for 2D.")
        
        # Updated condition to check nufft_type_3d of the base_nufft_operator
        if self.dimensionality == 3 and hasattr(self.base_nufft_operator, 'nufft_type_3d') and \
           self.base_nufft_operator.nufft_type_3d != 'direct':
            print("Warning: SlidingWindowNUFFTOperator op_adj is optimized for 'direct' 3D NDFT backend. "
                  "The provided base_nufft_operator is not using 'direct' type. op_adj will call base operator's op_adj directly.")

        self.block_size = block_size
        if len(self.block_size) != self.dimensionality:
            raise ValueError(f"block_size dimension {len(self.block_size)} does not match operator dimensionality {self.dimensionality}.")

        self.overlap_ratio = overlap_ratio
        self.overlap_pixels = [int(b * self.overlap_ratio) for b in self.block_size]
        # Ensure overlap is even for symmetric Tukey window center point
        self.overlap_pixels = [o if o % 2 == 0 else o + 1 for o in self.overlap_pixels]
        
        self.window = self._create_tukey_window_nd(self.block_size, alpha=0.5).to(self.device) # alpha for Tukey

    def _create_tukey_window_nd(self, block_shape_tuple, alpha=0.5):
        """Creates an N-D Tukey window by outer product of 1D Tukey windows."""
        if not (0 <= alpha <= 1):
            raise ValueError("Tukey window alpha parameter must be between 0 and 1.")
        
        individual_windows = []
        for dim_size in block_shape_tuple:
            win_1d = torch.from_numpy(tukey(dim_size, alpha=alpha, sym=True).astype(np.float32))
            individual_windows.append(win_1d)
        
        if len(block_shape_tuple) == 1:
            return individual_windows[0]
        elif len(block_shape_tuple) == 2:
            return torch.outer(individual_windows[0], individual_windows[1])
        elif len(block_shape_tuple) == 3:
            current_window = torch.ones(block_shape_tuple, dtype=torch.float32)
            for d_idx, win_1d in enumerate(individual_windows):
                view_shape = [1] * len(block_shape_tuple)
                view_shape[d_idx] = block_shape_tuple[d_idx]
                current_window *= win_1d.view(view_shape)
            return current_window
        else:
            raise ValueError("SlidingWindow currently supports 1D, 2D, or 3D blocks.")


    def op(self, image_data_tensor):
        return self.base_nufft_operator.op(image_data_tensor)

    def op_adj(self, k_space_data_tensor):
        # Check if base operator is suitable for sliding window (3D direct)
        is_3d_direct_nufft = (
            self.dimensionality == 3 and
            hasattr(self.base_nufft_operator, 'nufft_type_3d') and
            self.base_nufft_operator.nufft_type_3d == 'direct'
        )

        if not is_3d_direct_nufft:
            # print("Warning: SlidingWindowNUFFTOperator.op_adj is optimized for 3D 'direct' NDFT. "
            #       "Calling base operator's op_adj for current configuration.")
            return self.base_nufft_operator.op_adj(k_space_data_tensor) # No custom args for non-direct

        k_space_data_tensor = torch.as_tensor(k_space_data_tensor, dtype=torch.complex64, device=self.device)
        final_image = torch.zeros(self.full_image_shape, dtype=torch.complex64, device=self.device)
        weight_map = torch.zeros(self.full_image_shape, dtype=torch.float32, device=self.device) # Real-valued weights

        # Calculate strides for block iteration
        strides = [b - o for b, o in zip(self.block_size, self.overlap_pixels)]
        strides = [max(1, s) for s in strides] # Ensure stride is at least 1

        # Iterate over blocks
        # Assuming full_image_shape and block_size are (Depth, Height, Width)
        for z_start in range(0, self.full_image_shape[0], strides[0]):
            for y_start in range(0, self.full_image_shape[1], strides[1]):
                for x_start in range(0, self.full_image_shape[2], strides[2]):
                    z_end = min(z_start + self.block_size[0], self.full_image_shape[0])
                    y_end = min(y_start + self.block_size[1], self.full_image_shape[1])
                    x_end = min(x_start + self.block_size[2], self.full_image_shape[2])

                    current_block_shape = (z_end - z_start, y_end - y_start, x_end - x_start)
                    if any(s == 0 for s in current_block_shape): continue # Skip empty blocks

                    # Generate normalized voxel coordinates for the current block
                    coords_z_block = torch.linspace(
                        (z_start / (self.full_image_shape[0] -1) if self.full_image_shape[0]>1 else 0.0) - 0.5, 
                        ((z_end -1) / (self.full_image_shape[0] -1) if self.full_image_shape[0]>1 else 0.0) - 0.5, 
                        current_block_shape[0], device=self.device, dtype=torch.float32
                    )
                    coords_y_block = torch.linspace(
                        (y_start / (self.full_image_shape[1] -1) if self.full_image_shape[1]>1 else 0.0) - 0.5, 
                        ((y_end -1) / (self.full_image_shape[1] -1) if self.full_image_shape[1]>1 else 0.0) - 0.5, 
                        current_block_shape[1], device=self.device, dtype=torch.float32
                    )
                    coords_x_block = torch.linspace(
                        (x_start / (self.full_image_shape[2] -1) if self.full_image_shape[2]>1 else 0.0) - 0.5, 
                        ((x_end -1) / (self.full_image_shape[2] -1) if self.full_image_shape[2]>1 else 0.0) - 0.5, 
                        current_block_shape[2], device=self.device, dtype=torch.float32
                    )
                    
                    grid_z_block, grid_y_block, grid_x_block = torch.meshgrid(
                        coords_z_block, coords_y_block, coords_x_block, indexing='ij'
                    )
                    block_voxel_coords_flat = torch.stack(
                        (grid_x_block.flatten(), grid_y_block.flatten(), grid_z_block.flatten()), dim=1
                    )

                    # Reconstruct block using modified op_adj
                    block_image_flat = self.base_nufft_operator.op_adj(
                        k_space_data_tensor, 
                        output_voxel_coords_flat=block_voxel_coords_flat
                    )
                    block_image = block_image_flat.reshape(current_block_shape)

                    # Determine current window slice (if block is smaller than full window at edges)
                    current_window_slice = self.window[
                        :current_block_shape[0], 
                        :current_block_shape[1], 
                        :current_block_shape[2]
                    ]
                    
                    # Add weighted block to final image and update weight map
                    final_image[z_start:z_end, y_start:y_end, x_start:x_end] += block_image * current_window_slice
                    weight_map[z_start:z_end, y_start:y_end, x_start:x_end] += current_window_slice
        
        # Normalize by weight map
        final_image = final_image / (weight_map + 1e-9) # Avoid division by zero
        return final_image

# --- Inverse Radon Transform Operator ---
class IRadon(Operator):
    """
    Inverse Radon Transform Operator (Filtered Backprojection).
    """
    def __init__(self, 
                 img_size: tuple[int, int], 
                 angles: np.ndarray | torch.Tensor, 
                 filter_type: str | None = "ramp", 
                 device: str | torch.device = 'cpu'):
        """
        Args:
            img_size (tuple[int, int]): Size of the image (height, width).
            angles (np.ndarray | torch.Tensor): Projection angles in radians.
            filter_type (str | None): Type of filter to use ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None).
                                      If None, no filtering is applied. Defaults to "ramp".
            device (str | torch.device): Device to perform computations on.
        """
        self.img_size = img_size
        if isinstance(angles, np.ndarray):
            angles = torch.from_numpy(angles).float()
        self.angles = angles.to(device)
        self.filter_type = filter_type
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.grid = self._create_grid()
        self.filter = self._ramp_filter()

    def _create_grid(self) -> torch.Tensor:
        """
        Creates the reconstruction grid.
        The grid is normalized to be in the range [-1, 1] for both x and y.
        """
        nx, ny = self.img_size
        x = torch.arange(nx, device=self.device, dtype=torch.float32) - (nx - 1) / 2
        y = torch.arange(ny, device=self.device, dtype=torch.float32) - (ny - 1) / 2
        Y, X = torch.meshgrid(y, x, indexing='ij') # Consistent with common image indexing

        # Normalize grid to [-1, 1] range based on the image diagonal for proper projection coverage
        max_dim = max(nx, ny) 
        norm_factor = (max_dim -1) / 2 
        
        X_norm = X / norm_factor
        Y_norm = Y / norm_factor
        
        num_angles = len(self.angles)
        grid_repeated_X = X_norm.unsqueeze(0).repeat(num_angles, 1, 1)
        grid_repeated_Y = Y_norm.unsqueeze(0).repeat(num_angles, 1, 1)

        cos_a = torch.cos(self.angles).view(-1, 1, 1)
        sin_a = torch.sin(self.angles).view(-1, 1, 1)

        # Calculate t-coordinates for each angle
        # t = x*cos(theta) + y*sin(theta)
        # This 't' corresponds to the radial distance in the sinogram
        t_coords = grid_repeated_X * cos_a + grid_repeated_Y * sin_a
        return t_coords # Shape: (num_angles, ny, nx) 

    def _ramp_filter(self) -> torch.Tensor | None:
        """
        Creates the ramp filter or other specified filters in the frequency domain.
        """
        if self.filter_type is None:
            return None

        # Determine the size of the filter (length of the detector/sinogram width)
        # This should match the number of radial samples in the sinogram.
        # Assuming sinogram width is related to the diagonal of the image.
        n_detector = int(np.ceil(np.sqrt(self.img_size[0]**2 + self.img_size[1]**2)))
        if n_detector % 2 == 0: # Ensure odd length for FFT symmetry
            n_detector += 1
        
        # Create frequency axis
        freqs = torch.fft.fftfreq(n_detector, device=self.device).unsqueeze(0) # Shape (1, n_detector)

        filter_val = torch.abs(freqs) # Ramp filter: |f|

        if self.filter_type == "ramp":
            pass # Already initialized
        elif self.filter_type == "shepp-logan":
            # sinc(f) = sin(pi*f) / (pi*f)
            # Shepp-Logan: |f| * (sinc(f/2))^2 for some implementations, or |f| * sinc(f)
            # Using a common version: |f| * sinc(f * pi / (2 * max_freq))
            # Here, freqs are already scaled by 1/N, so effectively max_freq is 0.5
            omega = torch.pi * freqs 
            shepp_logan_filter = torch.where(omega == 0, torch.tensor(1.0, device=self.device), torch.sin(omega / 2) / (omega / 2))
            filter_val *= shepp_logan_filter**2 # This is a common variant, others exist.
        elif self.filter_type == "cosine":
            filter_val *= torch.cos(torch.pi * freqs) # Cosine filter: |f| * cos(pi*f)
        elif self.filter_type == "hamming":
            # Hamming window in frequency domain: (0.54 + 0.46 * cos(2*pi*f/F_max))
            # F_max corresponds to freqs = 0.5 here. So 2*pi*f / (0.5) = 4*pi*f
            filter_val *= (0.54 + 0.46 * torch.cos(2 * torch.pi * freqs / (0.5 * 2))) # Correct scaling for freqs in [-0.5, 0.5]
        elif self.filter_type == "hann":
            # Hann window in frequency domain: (0.5 * (1 + cos(2*pi*f/F_max)))
            filter_val *= (0.5 * (1 + torch.cos(2 * torch.pi * freqs / (0.5 * 2))))
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
        
        # The filter should be applied along the detector dimension of the sinogram
        # Return shape (1, n_detector) for broadcasting with sinogram FFT
        return filter_val.float() # Ensure float

    def op(self, sino: torch.Tensor) -> torch.Tensor:
        """
        Performs filtered backprojection (inverse Radon transform).
        Args:
            sino (torch.Tensor): Sinogram data. Shape (num_angles, num_detector_pixels).
        Returns:
            torch.Tensor: Reconstructed image. Shape (img_height, img_width).
        """
        if not isinstance(sino, torch.Tensor):
            sino = torch.from_numpy(sino).float()
        sino = sino.to(self.device)

        if sino.ndim != 2:
            raise ValueError(f"Sinogram must be 2D (num_angles, num_detector_pixels), got {sino.shape}")
        if sino.shape[0] != len(self.angles):
            raise ValueError(f"Sinogram angle dimension ({sino.shape[0]}) does not match "
                             f"number of angles provided at init ({len(self.angles)}).")

        num_angles, n_detector_sino = sino.shape
        
        # Apply filter if specified
        if self.filter is not None:
            # Check if filter size matches sinogram detector size
            if self.filter.shape[1] != n_detector_sino:
                # If not, it implies the filter was created with a default n_detector.
                # We need to recreate the filter or interpolate it.
                # For simplicity, let's try to recreate with the actual sinogram detector size.
                # This assumes the user might pass a sinogram with a different detector resolution
                # than what was estimated from img_size.
                # print(f"Warning: Filter size ({self.filter.shape[1]}) mismatch with sinogram detector size ({n_detector_sino}). "
                #       f"Recreating filter. This may happen if sinogram resolution differs from image diagonal estimate.")
                
                original_filter_type = self.filter_type # Store original
                original_n_detector = self.filter.shape[1]

                # Temporarily update n_detector for filter creation based on sino
                temp_n_detector = n_detector_sino
                
                # Create frequency axis for this specific sinogram
                freqs_sino = torch.fft.fftfreq(temp_n_detector, device=self.device).unsqueeze(0)
                
                current_filter = torch.abs(freqs_sino) # Ramp

                if self.filter_type == "ramp":
                    pass
                elif self.filter_type == "shepp-logan":
                    omega = torch.pi * freqs_sino
                    shepp_filter = torch.where(omega == 0, torch.tensor(1.0, device=self.device), torch.sin(omega/2) / (omega/2))
                    current_filter *= shepp_filter**2
                elif self.filter_type == "cosine":
                    current_filter *= torch.cos(torch.pi * freqs_sino)
                elif self.filter_type == "hamming":
                    current_filter *= (0.54 + 0.46 * torch.cos(2 * torch.pi * freqs_sino / (0.5*2) ))
                elif self.filter_type == "hann":
                    current_filter *= (0.5 * (1 + torch.cos(2 * torch.pi * freqs_sino/ (0.5*2) )))
                
                filter_to_apply = current_filter.float()
            else:
                filter_to_apply = self.filter

            # FFT of the sinogram (along detector dimension)
            sino_fft = torch.fft.fft(sino, dim=1)
            # Apply filter
            filtered_sino_fft = sino_fft * filter_to_apply
            # IFFT
            filtered_sino = torch.fft.ifft(filtered_sino_fft, dim=1).real
        else:
            filtered_sino = sino.real # Ensure real if no filter applied

        # Backprojection
        # The self.grid (t_coords) has shape (num_angles, ny, nx)
        # It contains the 't' sample locations for each pixel (x,y) and angle.
        # These t_coords are normalized to [-1, 1] if the sinogram detector dimension also spans [-1, 1].
        # We need to map these t_coords to indices in the filtered_sino.

        # filtered_sino has shape (num_angles, n_detector_sino)
        # Let detector pixels be indexed from 0 to n_detector_sino - 1.
        # A t_coord of -1 should map to index 0.
        # A t_coord of +1 should map to index n_detector_sino - 1.
        # So, index = (t_coord + 1) / 2 * (n_detector_sino - 1)
        
        # Normalize grid values to [0, n_detector_sino - 1] for indexing
        # self.grid has values in approx [-1, 1] (can be slightly outside due to image corners)
        # We need to ensure they map correctly to the sinogram's radial dimension
        t_indices = (self.grid + 1) / 2 * (n_detector_sino - 1)

        # Interpolation (linear)
        # We need to sample filtered_sino at (angle_idx, t_indices)
        # angle_idx is straightforward (0 to num_angles-1)
        # t_indices are the difficult part.
        
        # Create dummy angle indices for interpolation: shape (num_angles, 1, 1)
        angle_indices_for_interp = torch.arange(num_angles, device=self.device).view(-1, 1, 1)
        angle_indices_for_interp = angle_indices_for_interp.expand(-1, self.img_size[0], self.img_size[1])
        
        # filtered_sino needs to be (num_angles, n_detector_sino)
        # t_indices needs to be (num_angles, H, W)
        # We need to sample along the n_detector_sino dimension of filtered_sino.
        # grid_sample expects normalized coordinates in [-1, 1].
        # Our t_indices are currently in [0, n_detector_sino - 1].
        # Normalize t_indices to [-1, 1] for grid_sample:
        # norm_t_indices = (t_indices / (n_detector_sino - 1)) * 2 - 1
        # This simplifies to self.grid if self.grid was already correctly scaled for a sinogram spanning [-1,1]
        
        # The `align_corners` argument in `grid_sample` is crucial.
        # If `align_corners=True`, -1 and 1 correspond to the centers of the corner pixels.
        # If `align_corners=False`, -1 and 1 correspond to the edges of the corner pixels.
        # Given our `t_indices` logic: `(t_coord + 1) / 2 * (n_detector_sino - 1)`
        # This maps -1 to 0 and 1 to `n_detector_sino - 1`, which are pixel centers. So `align_corners=True` seems appropriate.

        # `grid_sample` needs the input tensor to be of shape (N, C, Din, Hin, Win) or (N, C, Din, Hin) etc.
        # `filtered_sino` is (num_angles, n_detector_sino). Let's make it (num_angles, 1, 1, n_detector_sino)
        # The grid for sampling should be (N, Hout, Wout, 2) for 2D sampling.
        # Our `self.grid` (t_coords) is (num_angles, ny, nx). These are the 'x' coordinates for sampling.
        # The 'y' coordinates for sampling are implicitly the angle dimension, which we handle by iterating or careful stacking.

        # Simpler approach: iterate through angles for backprojection, as commonly done.
        reconstructed_image = torch.zeros(self.img_size, device=self.device, dtype=torch.float32)
        
        # t_indices are (num_angles, ny, nx)
        # For each angle, t_indices[angle_idx, :, :] gives the detector coordinates for that angle.
        for i in range(num_angles):
            # Current angle's filtered sinogram data: shape (n_detector_sino)
            sino_line = filtered_sino[i, :]
            # Current angle's t-coordinates for each pixel: shape (ny, nx)
            current_t_indices = t_indices[i, :, :]
            
            # Interpolate sino_line at current_t_indices
            # We need 1D interpolation. `torch.interpolat` is not available.
            # Manual linear interpolation:
            t_floor = torch.floor(current_t_indices).long()
            t_ceil = torch.ceil(current_t_indices).long()
            
            # Clamp indices to be within bounds [0, n_detector_sino - 1]
            t_floor = torch.clamp(t_floor, 0, n_detector_sino - 1)
            t_ceil = torch.clamp(t_ceil, 0, n_detector_sino - 1)
            
            # Interpolation weights
            dt = current_t_indices - t_floor.float()
            
            val_floor = sino_line[t_floor]
            val_ceil = sino_line[t_ceil]
            
            interpolated_val = val_floor * (1 - dt) + val_ceil * dt
            reconstructed_image += interpolated_val

        # Normalize by number of angles (or pi/num_angles depending on convention)
        # The factor np.pi / (2 * num_angles) is common in FBP.
        # Or sometimes just 1/num_angles. Let's use a common one from skimage.
        # Skimage uses scaling by `np.pi / (2 * num_projections)` if filtered.
        # Or `1.0 / num_projections` if not filtered.
        if self.filter is not None:
             reconstructed_image *= (torch.pi / (2.0 * num_angles))
        else:
             reconstructed_image *= (1.0 / num_angles) # Or maybe no scaling if not filtered?
                                                     # Let's assume some scaling is needed.

        return reconstructed_image

    def op_adj(self, img: torch.Tensor) -> torch.Tensor:
        """
        Performs the adjoint of filtered backprojection, which is the Radon transform (projection).
        Args:
            img (torch.Tensor): Image data. Shape (img_height, img_width).
        Returns:
            torch.Tensor: Sinogram data. Shape (num_angles, num_detector_pixels).
        """
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).float()
        img = img.to(self.device)

        if tuple(img.shape) != self.img_size:
            raise ValueError(f"Input image shape {img.shape} does not match expected {self.img_size}")

        num_angles = len(self.angles)
        # Determine n_detector_sino. This should be consistent with the filter size if a filter was created.
        # If no filter, use the estimate from image diagonal.
        if self.filter is not None:
            n_detector_sino = self.filter.shape[1]
        else:
            n_detector_sino = int(np.ceil(np.sqrt(self.img_size[0]**2 + self.img_size[1]**2)))
            if n_detector_sino % 2 == 0:
                n_detector_sino +=1

        sinogram = torch.zeros((num_angles, n_detector_sino), device=self.device, dtype=torch.float32)

        # self.grid (t_coords) has shape (num_angles, ny, nx)
        # It contains the 't' sample locations for each pixel (x,y) and angle.
        # These t_coords are normalized to [-1, 1].
        # We need to map these t_coords to indices in the sinogram's detector dimension.
        t_indices = (self.grid + 1) / 2 * (n_detector_sino - 1)

        # Adjoint of interpolation (summing contributions)
        # For each angle:
        for i in range(num_angles):
            current_t_indices = t_indices[i, :, :] # Shape (ny, nx)
            
            # Get pixel values from the image: img has shape (ny, nx)
            # For each pixel in the image, its value `img[y,x]` contributes to `sinogram[i, t_idx]`
            # where `t_idx` is derived from `current_t_indices[y,x]`.
            
            # This is essentially "splatting" the image values onto the sinogram grid.
            # For each (y,x) pixel in the image, find its corresponding t_idx.
            # Add img[y,x] to sinogram[i, round(t_idx)].
            # For better accuracy, distribute energy to neighboring bins (linear interpolation adjoint).

            t_floor = torch.floor(current_t_indices).long()
            t_ceil = torch.ceil(current_t_indices).long()
            
            dt = current_t_indices - t_floor.float()

            # Ensure indices are within bounds for accumulation
            # We need to be careful here. If t_idx is out of [0, n_detector_sino-1], that contribution is lost.
            valid_mask_floor = (t_floor >= 0) & (t_floor < n_detector_sino)
            valid_mask_ceil = (t_ceil >= 0) & (t_ceil < n_detector_sino)

            # Accumulate contributions (adjoint of linear interpolation)
            # Flatten image and indices for scatter_add_
            img_flat = img.flatten() # Shape (ny*nx)
            
            # For t_floor
            indices_floor_flat = t_floor.flatten() # Shape (ny*nx)
            weights_floor = (1 - dt).flatten() # Shape (ny*nx)
            
            valid_indices_floor = indices_floor_flat[valid_mask_floor.flatten()]
            valid_img_values_floor = img_flat[valid_mask_floor.flatten()]
            valid_weights_floor = weights_floor[valid_mask_floor.flatten()]
            
            # sinogram[i, valid_indices_floor] += valid_img_values_floor * valid_weights_floor
            # Use index_add_ for this, which is like scatter_add but for a specific dimension
            sinogram[i].index_add_(0, valid_indices_floor, valid_img_values_floor * valid_weights_floor)

            # For t_ceil
            indices_ceil_flat = t_ceil.flatten() # Shape (ny*nx)
            weights_ceil = dt.flatten() # Shape (ny*nx)

            valid_indices_ceil = indices_ceil_flat[valid_mask_ceil.flatten()]
            valid_img_values_ceil = img_flat[valid_mask_ceil.flatten()]
            valid_weights_ceil = weights_ceil[valid_mask_ceil.flatten()]
            
            sinogram[i].index_add_(0, valid_indices_ceil, valid_img_values_ceil * valid_weights_ceil)


        # Adjoint of filtering: apply the filter again (or its conjugate transpose if complex)
        # Since our filter is real and symmetric in magnitude, applying it again is correct for adjoint.
        if self.filter is not None:
            # Check for filter size mismatch as in op()
            if self.filter.shape[1] != n_detector_sino:
                # print(f"Warning: Filter size mismatch in op_adj. Recreating filter.")
                original_filter_type = self.filter_type
                temp_n_detector = n_detector_sino
                freqs_sino = torch.fft.fftfreq(temp_n_detector, device=self.device).unsqueeze(0)
                current_filter = torch.abs(freqs_sino)
                if self.filter_type == "ramp": pass
                elif self.filter_type == "shepp-logan":
                    omega = torch.pi * freqs_sino
                    shepp_filter = torch.where(omega == 0, torch.tensor(1.0, device=self.device), torch.sin(omega/2) / (omega/2))
                    current_filter *= shepp_filter**2
                elif self.filter_type == "cosine": current_filter *= torch.cos(torch.pi * freqs_sino)
                elif self.filter_type == "hamming": current_filter *= (0.54 + 0.46 * torch.cos(2 * torch.pi * freqs_sino / (0.5*2)))
                elif self.filter_type == "hann": current_filter *= (0.5 * (1 + torch.cos(2 * torch.pi * freqs_sino / (0.5*2))))
                filter_to_apply = current_filter.float()
            else:
                filter_to_apply = self.filter

            sino_fft = torch.fft.fft(sinogram, dim=1)
            filtered_sino_fft = sino_fft * filter_to_apply # Filter is real, so same application
            sinogram = torch.fft.ifft(filtered_sino_fft, dim=1).real
        
        # Adjoint of scaling factor
        # If op used `*= C`, then op_adj should use `*= C` (if C is real)
        # The scaling factor np.pi / (2 * num_angles) or 1/num_angles was applied in op.
        # So, we apply it here as well.
        if self.filter is not None:
             sinogram *= (torch.pi / (2.0 * num_angles))
        else:
             sinogram *= (1.0 / num_angles)

        return sinogram


# --- PET Forward Projection Operator ---
class PETForwardProjection(Operator):
    """
    Basic PET Forward Projection Operator (Radon Transform without filtering).
    op: Image -> Sinogram (sum along lines)
    op_adj: Sinogram -> Image (simple backprojection)
    """
    def __init__(self, 
                 img_size: tuple[int, int], 
                 angles: np.ndarray | torch.Tensor, 
                 device: str | torch.device = 'cpu'):
        """
        Args:
            img_size (tuple[int, int]): Size of the image (height, width).
            angles (np.ndarray | torch.Tensor): Projection angles in radians.
            device (str | torch.device): Device to perform computations on.
        """
        self.img_size = img_size
        if isinstance(angles, np.ndarray):
            angles = torch.from_numpy(angles).float()
        self.angles = angles.to(device)
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Estimate number of detector pixels (radial bins in sinogram)
        # Based on the diagonal of the image to capture all information.
        self.n_detector_pixels = int(np.ceil(np.sqrt(img_size[0]**2 + img_size[1]**2)))
        if self.n_detector_pixels % 2 == 0: # Ensure odd length for symmetry if needed, though less critical here
            self.n_detector_pixels += 1
            
        self.grid = self._create_grid() # Create the geometric mapping

    def _create_grid(self) -> torch.Tensor:
        """
        Creates the reconstruction grid, mapping image pixels to sinogram radial coordinates.
        This is identical to IRadon._create_grid.
        The grid t_coords are normalized to be in the range [-1, 1] for x and y.
        """
        nx, ny = self.img_size # Note: Traditionally nx is width, ny is height.
                               # PyTorch typically uses (H, W), so img_size[0] is ny, img_size[1] is nx.
                               # Let's stick to img_size[0] = H (ny), img_size[1] = W (nx) for consistency with IRadon.
        
        # Create pixel coordinates
        # y_coords range from -(ny-1)/2 to (ny-1)/2
        # x_coords range from -(nx-1)/2 to (nx-1)/2
        y_pixel_coords = torch.arange(self.img_size[0], device=self.device, dtype=torch.float32) - (self.img_size[0] - 1) / 2
        x_pixel_coords = torch.arange(self.img_size[1], device=self.device, dtype=torch.float32) - (self.img_size[1] - 1) / 2
        
        # Create meshgrid. Note: PyTorch's meshgrid defaults to 'xy' indexing if not specified.
        # For image operations, 'ij' (matrix indexing) is often more intuitive: Y, X = meshgrid(y_coords, x_coords)
        # Here, Y will have shape (img_size[0], img_size[1]), X will have shape (img_size[0], img_size[1])
        Y_mesh, X_mesh = torch.meshgrid(y_pixel_coords, x_pixel_coords, indexing='ij')

        # Normalize grid to roughly [-1, 1] range.
        # The normalization factor should ensure that the extreme corners of the image project
        # to values near -1 or 1 in the 't' (radial) coordinate of the sinogram.
        # The diagonal of the image is sqrt(nx^2 + ny^2). Half of this is a good norm_factor.
        # Or, use max_dim / 2 as in IRadon for consistency if sinogram spans based on max_dim.
        # Let's use (max_dim - 1) / 2, consistent with IRadon's t-coord normalization.
        max_img_dim = max(self.img_size[0], self.img_size[1])
        norm_factor = (max_img_dim - 1) / 2
        if norm_factor == 0: # Avoid division by zero for 1x1 image
            norm_factor = 1 

        X_norm = X_mesh / norm_factor
        Y_norm = Y_mesh / norm_factor
        
        num_angles = len(self.angles)
        # Expand X_norm and Y_norm for each angle
        grid_repeated_X = X_norm.unsqueeze(0).repeat(num_angles, 1, 1) # Shape: (num_angles, H, W)
        grid_repeated_Y = Y_norm.unsqueeze(0).repeat(num_angles, 1, 1) # Shape: (num_angles, H, W)

        # Precompute cos and sin of angles
        cos_a = torch.cos(self.angles).view(-1, 1, 1) # Shape: (num_angles, 1, 1)
        sin_a = torch.sin(self.angles).view(-1, 1, 1) # Shape: (num_angles, 1, 1)

        # Calculate t-coordinates for each pixel and angle: t = x*cos(theta) + y*sin(theta)
        # Note on coordinate systems for t = x*cos + y*sin:
        # If using standard Cartesian x (horizontal right) and y (vertical up),
        # and theta is angle from positive x-axis, this is correct.
        # Our pixel coordinates: x_pixel_coords increase to the right, y_pixel_coords increase downwards.
        # If angles are defined relative to the positive x-axis (horizontal),
        # then for y increasing downwards, we might use t = X*cos(a) - Y*sin(a) if Y was positive upwards.
        # Or, if Y is positive downwards, and angles are from positive X:
        # Standard Radon: t = x cos(theta) + y sin(theta) where y is "up".
        # If our Y_norm is (pixel_y - center_y), which means it's negative "above" center, positive "below".
        # If angles are measured from positive x-axis counter-clockwise:
        # This definition of t_coords (grid_repeated_X * cos_a + grid_repeated_Y * sin_a) is standard
        # when X and Y are Cartesian coordinates. Our X_norm and Y_norm are like this.
        t_coords = grid_repeated_X * cos_a + grid_repeated_Y * sin_a
        return t_coords # Shape: (num_angles, img_size[0], img_size[1])

    def op(self, img: torch.Tensor) -> torch.Tensor:
        """
        Performs PET forward projection (Radon transform without filtering).
        Args:
            img (torch.Tensor): Image data. Shape (img_height, img_width).
        Returns:
            torch.Tensor: Sinogram data. Shape (num_angles, num_detector_pixels).
        """
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).float()
        img = img.to(self.device)

        if tuple(img.shape) != self.img_size:
            raise ValueError(f"Input image shape {img.shape} does not match expected {self.img_size}")

        num_angles = len(self.angles)
        sinogram = torch.zeros((num_angles, self.n_detector_pixels), device=self.device, dtype=torch.float32)

        # self.grid (t_coords) has shape (num_angles, H, W)
        # These t_coords are normalized (approx [-1, 1]). Map to detector indices [0, n_detector_pixels-1]
        t_indices = (self.grid + 1) / 2 * (self.n_detector_pixels - 1)

        # Adjoint of linear interpolation (summing contributions into sinogram bins)
        for i in range(num_angles):
            current_t_indices_for_angle = t_indices[i, :, :] # Shape (H, W)
            
            t_floor = torch.floor(current_t_indices_for_angle).long()
            t_ceil = torch.ceil(current_t_indices_for_angle).long()
            
            # Weights for linear interpolation (distance to floor and ceil)
            dt = current_t_indices_for_angle - t_floor.float()

            # Ensure indices are within bounds for accumulation into sinogram
            valid_mask_floor = (t_floor >= 0) & (t_floor < self.n_detector_pixels)
            valid_mask_ceil = (t_ceil >= 0) & (t_ceil < self.n_detector_pixels)

            img_flat = img.flatten() # Shape (H*W)
            
            # Contributions to t_floor bins
            indices_floor_flat = t_floor.flatten() # Shape (H*W)
            weights_floor = (1 - dt).flatten()     # Shape (H*W)
            
            # Filter by valid_mask_floor before calling index_add_
            active_indices_floor = indices_floor_flat[valid_mask_floor.flatten()]
            active_img_values_floor = img_flat[valid_mask_floor.flatten()]
            active_weights_floor = weights_floor[valid_mask_floor.flatten()]
            
            sinogram[i].index_add_(0, active_indices_floor, active_img_values_floor * active_weights_floor)

            # Contributions to t_ceil bins
            indices_ceil_flat = t_ceil.flatten()   # Shape (H*W)
            weights_ceil = dt.flatten()            # Shape (H*W)

            active_indices_ceil = indices_ceil_flat[valid_mask_ceil.flatten()]
            active_img_values_ceil = img_flat[valid_mask_ceil.flatten()]
            active_weights_ceil = weights_ceil[valid_mask_ceil.flatten()]

            sinogram[i].index_add_(0, active_indices_ceil, active_img_values_ceil * active_weights_ceil)
            
        # No filtering and no specific scaling like in FBP's IRadon.
        # The sum itself is the operation.
        return sinogram

    def op_adj(self, sino: torch.Tensor) -> torch.Tensor:
        """
        Performs adjoint of PET forward projection (simple backprojection without filtering).
        Args:
            sino (torch.Tensor): Sinogram data. Shape (num_angles, num_detector_pixels).
        Returns:
            torch.Tensor: Reconstructed image. Shape (img_height, img_width).
        """
        if not isinstance(sino, torch.Tensor):
            sino = torch.from_numpy(sino).float()
        sino = sino.to(self.device)

        if sino.ndim != 2:
            raise ValueError(f"Sinogram must be 2D (num_angles, num_detector_pixels), got {sino.shape}")
        if sino.shape[0] != len(self.angles):
            raise ValueError(f"Sinogram angle dimension ({sino.shape[0]}) does not match "
                             f"number of angles provided at init ({len(self.angles)}).")
        if sino.shape[1] != self.n_detector_pixels:
            raise ValueError(f"Sinogram detector dimension ({sino.shape[1]}) does not match "
                             f"expected num_detector_pixels ({self.n_detector_pixels}).")

        num_angles = len(self.angles)
        backprojected_image = torch.zeros(self.img_size, device=self.device, dtype=torch.float32)

        # self.grid (t_coords) has shape (num_angles, H, W)
        # Map these normalized t_coords to actual sinogram detector indices
        t_indices = (self.grid + 1) / 2 * (self.n_detector_pixels - 1)

        # Linear interpolation for sampling sinogram values
        for i in range(num_angles):
            sino_line = sino[i, :] # Current angle's sinogram data: shape (n_detector_pixels)
            current_t_indices_for_angle = t_indices[i, :, :] # t-coords for this angle: shape (H, W)
            
            t_floor = torch.floor(current_t_indices_for_angle).long()
            t_ceil = torch.ceil(current_t_indices_for_angle).long()
            
            # Clamp indices to be within sinogram bounds [0, n_detector_pixels - 1]
            t_floor = torch.clamp(t_floor, 0, self.n_detector_pixels - 1)
            t_ceil = torch.clamp(t_ceil, 0, self.n_detector_pixels - 1)
            
            # Interpolation weights
            dt = current_t_indices_for_angle - t_floor.float()
            
            val_floor = sino_line[t_floor] # Sample from sinogram at floor indices
            val_ceil = sino_line[t_ceil]   # Sample from sinogram at ceil indices
            
            # Linearly interpolated value from sinogram to be added to the image pixel
            interpolated_sino_val = val_floor * (1 - dt) + val_ceil * dt
            backprojected_image += interpolated_sino_val # Accumulate contributions

        # No filtering and no specific scaling like in FBP's IRadon.
        # The sum of contributions is the simple backprojection.
        return backprojected_image


# --- Photoacoustic Tomography (PAT) Forward Projection Operator ---
class PATForwardProjection(Operator):
    """
    Basic PAT Forward Projection Operator for a 2D scenario.
    Models the generation of sensor data from an initial pressure distribution.
    op: InitialPressureImage -> SensorData (Time-series for each sensor)
    op_adj: Not implemented for this task.
    """
    def __init__(self, 
                 img_shape: tuple[int, int], 
                 sensor_positions: torch.Tensor | np.ndarray, 
                 sound_speed: float, 
                 time_points: torch.Tensor | np.ndarray, 
                 device: str | torch.device = 'cpu'):
        """
        Args:
            img_shape (tuple[int, int]): Shape of the 2D image grid (ny, nx) i.e. (height, width).
            sensor_positions (torch.Tensor | np.ndarray): Positions of sensors, shape (num_sensors, 2).
                                                          Assumed to be in the same coordinate system as the image pixels.
            sound_speed (float): Speed of sound in the medium.
            time_points (torch.Tensor | np.ndarray): Time samples at which data is recorded, shape (num_time_samples,).
            device (str | torch.device): Device to perform computations on.
        """
        self.img_shape = img_shape # (ny, nx) -> (height, width)
        self.sound_speed = sound_speed
        
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if isinstance(sensor_positions, np.ndarray):
            sensor_positions = torch.from_numpy(sensor_positions).float()
        self.sensor_positions = sensor_positions.to(self.device)

        if isinstance(time_points, np.ndarray):
            time_points = torch.from_numpy(time_points).float()
        self.time_points = time_points.to(self.device)

        if self.sensor_positions.ndim != 2 or self.sensor_positions.shape[1] != 2:
            raise ValueError(f"sensor_positions must have shape (num_sensors, 2), got {self.sensor_positions.shape}")
        if self.time_points.ndim != 1:
            raise ValueError(f"time_points must have shape (num_time_samples,), got {self.time_points.shape}")

        # Create pixel coordinate grid
        # Assume pixel spacing is 1.0 for simplicity.
        # Coordinates are relative to the center of the image.
        ny, nx = self.img_shape
        
        # Create 1D coordinate vectors for y and x
        # y_coords from -(ny-1)/2 to (ny-1)/2
        # x_coords from -(nx-1)/2 to (nx-1)/2
        y_coords_1d = torch.arange(ny, device=self.device, dtype=torch.float32) - (ny - 1) / 2.0
        x_coords_1d = torch.arange(nx, device=self.device, dtype=torch.float32) - (nx - 1) / 2.0

        # Create 2D meshgrid
        # self.pixel_y_coords will have shape (ny, nx)
        # self.pixel_x_coords will have shape (ny, nx)
        self.pixel_y_coords, self.pixel_x_coords = torch.meshgrid(y_coords_1d, x_coords_1d, indexing='ij')
        
        # Store pixel coordinates as (ny, nx, 2) for easier iteration if needed, or flatten
        # For the current loop structure, separate meshgrids are fine.
        # self.pixel_coords = torch.stack((self.pixel_x_coords, self.pixel_y_coords), dim=-1) # Shape (ny, nx, 2)

        # Pixel size approximation (assuming square pixels with spacing 1.0)
        self.pixel_size = 1.0 


    def op(self, initial_pressure_image: torch.Tensor) -> torch.Tensor:
        """
        Calculates the photoacoustic sensor data for a given initial pressure distribution.
        Args:
            initial_pressure_image (torch.Tensor): 2D tensor of shape self.img_shape (ny, nx).
        Returns:
            torch.Tensor: Sensor data, shape (num_sensors, num_time_samples).
        """
        if not isinstance(initial_pressure_image, torch.Tensor):
            initial_pressure_image = torch.from_numpy(initial_pressure_image).float()
        initial_pressure_image = initial_pressure_image.to(self.device)

        if initial_pressure_image.shape != self.img_shape:
            raise ValueError(f"Input initial_pressure_image shape {initial_pressure_image.shape} "
                             f"does not match expected {self.img_shape}")

        num_sensors = self.sensor_positions.shape[0]
        num_time_samples = self.time_points.shape[0]
        ny, nx = self.img_shape

        sensor_data = torch.zeros((num_sensors, num_time_samples), device=self.device, dtype=torch.float32)

        # Tolerance for checking if a pixel is on the integration shell (approx. half pixel size)
        # This defines the "thickness" of the spherical/circular shell for integration.
        shell_tolerance = self.pixel_size / 2.0

        for s in range(num_sensors):
            sensor_pos_x, sensor_pos_y = self.sensor_positions[s, 0], self.sensor_positions[s, 1]
            
            for t_idx in range(num_time_samples):
                current_t = self.time_points[t_idx]
                radius_of_integration = self.sound_speed * current_t

                # Calculate distances from all pixels to the current sensor
                # dist_sq = (self.pixel_x_coords - sensor_pos_x)**2 + (self.pixel_y_coords - sensor_pos_y)**2
                # distances = torch.sqrt(dist_sq) # Shape (ny, nx)
                
                # Instead of calculating all distances and then masking, 
                # iterate pixels for clarity, as per prompt, though less efficient.
                accumulated_pressure = 0.0
                for y_idx in range(ny):
                    for x_idx in range(nx):
                        pixel_x = self.pixel_x_coords[y_idx, x_idx]
                        pixel_y = self.pixel_y_coords[y_idx, x_idx]
                        
                        distance = torch.sqrt((pixel_x - sensor_pos_x)**2 + (pixel_y - sensor_pos_y)**2)
                        
                        if torch.abs(distance - radius_of_integration) < shell_tolerance:
                            accumulated_pressure += initial_pressure_image[y_idx, x_idx]
                
                sensor_data[s, t_idx] = accumulated_pressure
        
        return sensor_data

    def op_adj(self, sensor_data_tensor: torch.Tensor) -> torch.Tensor:
        """
        Adjoint of PAT Forward Projection. This performs backprojection of sensor data.
        Args:
            sensor_data_tensor (torch.Tensor): 2D tensor of shape (num_sensors, num_time_samples).
        Returns:
            torch.Tensor: Reconstructed image, shape self.img_shape (ny, nx).
        """
        if not isinstance(sensor_data_tensor, torch.Tensor):
            sensor_data_tensor = torch.from_numpy(sensor_data_tensor).float()
        sensor_data_tensor = sensor_data_tensor.to(self.device)

        num_sensors_data = sensor_data_tensor.shape[0]
        num_time_samples_data = sensor_data_tensor.shape[1]

        num_sensors_op = self.sensor_positions.shape[0]
        num_time_samples_op = self.time_points.shape[0]

        if num_sensors_data != num_sensors_op or num_time_samples_data != num_time_samples_op:
            raise ValueError(
                f"Input sensor_data_tensor shape ({num_sensors_data}, {num_time_samples_data}) "
                f"does not match expected operator dimensions ({num_sensors_op}, {num_time_samples_op})."
            )

        reconstructed_image = torch.zeros(self.img_shape, device=self.device, dtype=torch.float32)
        ny, nx = self.img_shape

        # Tolerance for checking if a pixel is on the integration shell (approx. half pixel size)
        shell_tolerance = self.pixel_size / 2.0
        epsilon = 1e-9 # To avoid computation for zero values

        for s in range(num_sensors_op):
            sensor_pos_x, sensor_pos_y = self.sensor_positions[s, 0], self.sensor_positions[s, 1]
            
            for t_idx in range(num_time_samples_op):
                current_val = sensor_data_tensor[s, t_idx]

                # If val is non-zero (or above a small epsilon)
                if torch.abs(current_val) < epsilon:
                    continue

                current_t = self.time_points[t_idx]
                radius_of_integration = self.sound_speed * current_t
                
                # This part can be slow due to iterating all pixels for each sensor/time point.
                # For a more efficient implementation, one might consider a different approach,
                # but following the provided algorithm structure.
                for y_idx in range(ny):
                    for x_idx in range(nx):
                        pixel_x = self.pixel_x_coords[y_idx, x_idx]
                        pixel_y = self.pixel_y_coords[y_idx, x_idx]
                        
                        distance = torch.sqrt((pixel_x - sensor_pos_x)**2 + (pixel_y - sensor_pos_y)**2)
                        
                        if torch.abs(distance - radius_of_integration) < shell_tolerance:
                            reconstructed_image[y_idx, x_idx] += current_val
                            
        return reconstructed_image


# --- Radio Interferometry Operator ---
class RadioInterferometryOperator(Operator):
    """
    Operator for Radio Interferometry.

    Models the relationship between a sky image and measured visibilities,
    which are samples of the image's 2D Fourier transform.
    """
    def __init__(self, uv_coordinates: torch.Tensor, image_shape: tuple[int, int], device: str = 'cpu'):
        """
        Initializes the RadioInterferometryOperator.

        Args:
            uv_coordinates (torch.Tensor): Tensor of shape (num_visibilities, 2)
                representing the (u,v) spatial frequency coordinates.
                These coordinates are assumed to be integers and pre-scaled to directly
                index a zero-centered, fftshift-ed 2D FFT grid of the image.
                For an image of shape (Ny, Nx):
                u coordinates should range from -Nx/2 to Nx/2 - 1.
                v coordinates should range from -Ny/2 to Ny/2 - 1.
            image_shape (tuple[int, int]): Shape of the sky image (Ny, Nx).
            device (str): Device ('cpu' or 'cuda').
        """
        super().__init__() # Operator is ABC, no specific super init needed unless it becomes nn.Module
        self.device = torch.device(device)
        self.uv_coordinates = uv_coordinates.to(device=self.device, dtype=torch.long)
        self.image_shape = image_shape # (Ny, Nx)
        
        if self.uv_coordinates.ndim != 2 or self.uv_coordinates.shape[1] != 2:
            raise ValueError("uv_coordinates must be a 2D tensor of shape (num_visibilities, 2).")

        # Basic validation for uv_coordinates ranges based on image_shape
        Ny, Nx = self.image_shape
        u_min, u_max = -Nx // 2, (Nx - 1) // 2 # Integer division for range
        v_min, v_max = -Ny // 2, (Ny - 1) // 2
        
        if not (torch.all(self.uv_coordinates[:, 0] >= u_min) and
                torch.all(self.uv_coordinates[:, 0] <= u_max)):
            raise ValueError(f"U coordinates are out of range [{u_min}, {u_max}]. Found min {self.uv_coordinates[:,0].min()}, max {self.uv_coordinates[:,0].max()}")
        
        if not (torch.all(self.uv_coordinates[:, 1] >= v_min) and
                torch.all(self.uv_coordinates[:, 1] <= v_max)):
            raise ValueError(f"V coordinates are out of range [{v_min}, {v_max}]. Found min {self.uv_coordinates[:,1].min()}, max {self.uv_coordinates[:,1].max()}")


    def op(self, sky_image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward operation: Sky image to visibilities.
        Performs 2D FFT of the image and samples at uv_coordinates.
        """
        if sky_image_tensor.shape != self.image_shape:
            raise ValueError(f"Input sky_image_tensor shape {sky_image_tensor.shape} "
                             f"does not match expected image_shape {self.image_shape}.")
        if sky_image_tensor.device != self.device:
            sky_image_tensor = sky_image_tensor.to(self.device)
        if not sky_image_tensor.is_complex():
            # FFT expects complex input, or real and will output complex.
            # To be safe, cast to complex if it's real.
            sky_image_tensor = sky_image_tensor.to(torch.complex64)


        # Perform 2D FFT
        f_image = torch.fft.fft2(sky_image_tensor, norm='ortho')
        f_image_shifted = torch.fft.fftshift(f_image) # Zero-frequency is at the center

        # uv_coordinates are (u,v) where u is horizontal (corresponds to Nx, image_shape[1])
        # and v is vertical (corresponds to Ny, image_shape[0])
        # FFT output f_image_shifted has shape (Ny, Nx)
        
        # Map zero-centered uv_coordinates to array indices
        # u-coords (dim 1, Nx) range -Nx/2 to Nx/2-1 -> 0 to Nx-1
        # v-coords (dim 0, Ny) range -Ny/2 to Ny/2-1 -> 0 to Ny-1
        u_indices = self.uv_coordinates[:, 0] + self.image_shape[1] // 2
        v_indices = self.uv_coordinates[:, 1] + self.image_shape[0] // 2
        
        # Ensure indices are within bounds (clamping)
        # This is important if uv_coordinates were not perfectly within the assumed range,
        # although the __init__ method already validates this. Clamping is a safeguard.
        u_indices = torch.clamp(u_indices, 0, self.image_shape[1] - 1)
        v_indices = torch.clamp(v_indices, 0, self.image_shape[0] - 1)

        # Sample the Fourier plane
        visibilities = f_image_shifted[v_indices, u_indices]
        
        return visibilities

    def op_adj(self, visibilities_tensor: torch.Tensor) -> torch.Tensor:
        """
        Adjoint operation: Visibilities to sky image (dirty image).
        Places visibilities onto a Fourier grid and performs inverse 2D FFT.
        """
        if visibilities_tensor.ndim != 1 or visibilities_tensor.shape[0] != self.uv_coordinates.shape[0]:
            raise ValueError(f"Input visibilities_tensor has incorrect shape or length. "
                             f"Expected 1D tensor of length {self.uv_coordinates.shape[0]}, "
                             f"got shape {visibilities_tensor.shape}.")
        if visibilities_tensor.device != self.device:
            visibilities_tensor = visibilities_tensor.to(self.device)
        if not visibilities_tensor.is_complex():
            # Ensure visibilities are complex, matching typical FFT output
            visibilities_tensor = visibilities_tensor.to(torch.complex64)

        # Create an empty Fourier grid (for fftshifted data)
        f_grid_shifted = torch.zeros(self.image_shape, dtype=torch.complex64, device=self.device)

        # Map zero-centered uv_coordinates to array indices
        u_indices = self.uv_coordinates[:, 0] + self.image_shape[1] // 2
        v_indices = self.uv_coordinates[:, 1] + self.image_shape[0] // 2
        
        # Ensure indices are within bounds (clamping)
        u_indices = torch.clamp(u_indices, 0, self.image_shape[1] - 1)
        v_indices = torch.clamp(v_indices, 0, self.image_shape[0] - 1)

        # Place visibilities onto the grid.
        # Using index_put with accumulate=True ensures summation if multiple uv-points
        # map to the same grid cell, which is crucial for a correct adjoint.
        f_grid_shifted.index_put_((v_indices, u_indices), visibilities_tensor, accumulate=True)
        
        # Inverse FFT
        # First, inverse shift (zero-frequency from center to corner)
        f_grid_ifftshifted = torch.fft.ifftshift(f_grid_shifted)
        # Then, inverse FFT
        sky_image_estimate = torch.fft.ifft2(f_grid_ifftshifted, norm='ortho')
        
        return sky_image_estimate



