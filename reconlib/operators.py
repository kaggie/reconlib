"""Module for defining Operator classes for MRI reconstruction."""

import torch
import numpy as np
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
