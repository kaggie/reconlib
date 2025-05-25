"""Module for defining Operator classes for MRI reconstruction."""

import torch
import numpy as np
from abc import ABC, abstractmethod
from scipy.special import i0 # For _iternufft2d_kaiser_bessel_kernel

# Operator Base Class
class Operator(ABC):
    """
    Abstract base class for operators.

    Defines the interface for forward and adjoint operations.
    """
    @abstractmethod
    def op(self, x):
        """
        Forward operation.

        Args:
            x: Input data.

        Returns:
            Result of the forward operation.
        """
        pass

    @abstractmethod
    def op_adj(self, y):
        """
        Adjoint operation.

        Args:
            y: Input data.

        Returns:
            Result of the adjoint operation.
        """
        pass

# --- Internal 2D NUFFT Helper Functions (from iternufft2d.py) ---
# Prefixed with _iternufft2d_

def _iternufft2d_kaiser_bessel_kernel(r, width, beta):
    # r is expected to be a torch tensor
    mask = r < (width / 2)
    # Clamp value inside sqrt to be non-negative to avoid NaN due to precision issues
    val_inside_sqrt = torch.clamp(1 - (2 * r[mask] / width)**2, min=0.0)
    z = torch.sqrt(val_inside_sqrt)
    kb = torch.zeros_like(r)
    # scipy.special.i0 expects NumPy array, convert back to tensor and move to original device
    kb_numpy_values = i0(beta * z.cpu().numpy())
    kb[mask] = torch.from_numpy(kb_numpy_values.astype(np.float32)).to(r.device) / float(i0(beta))
    return kb

def _iternufft2d_estimate_density_compensation(kx, ky):
    # kx, ky are expected to be torch tensors
    radius = torch.sqrt(kx**2 + ky**2)
    dcf = radius + 1e-3  # avoid zero center
    dcf /= dcf.max()
    return dcf

def _iternufft2d_nufft2d2_adjoint(kx, ky, kspace_data, image_shape, oversamp=2.0, width=4, beta=13.9085):
    # Inputs kx, ky, kspace_data are expected to be torch tensors on the same device.
    device = kx.device
    Nx, Ny = image_shape
    Nx_oversamp = int(Nx * oversamp)
    Ny_oversamp = int(Ny * oversamp)

    # Scale k-space coords to oversampled grid (assuming kx,ky in [-0.5, 0.5])
    kx_scaled = (kx + 0.5) * Nx_oversamp
    ky_scaled = (ky + 0.5) * Ny_oversamp

    # Density compensation weights
    dcf = _iternufft2d_estimate_density_compensation(kx, ky).to(device)
    kspace_data_weighted = kspace_data * dcf # Ensure this is used

    # Initialize oversampled grid and weight grid
    grid = torch.zeros((Nx_oversamp, Ny_oversamp), dtype=torch.complex64, device=device)
    weight_grid = torch.zeros((Nx_oversamp, Ny_oversamp), dtype=torch.float32, device=device) # Renamed from 'weight' in source

    half_width = width // 2

    # Gridding: interpolate k-space data to Cartesian grid with Kaiser-Bessel kernel
    for dx in range(-half_width, half_width + 1):
        for dy in range(-half_width, half_width + 1):
            x_idx = torch.floor(kx_scaled + dx).long() 
            y_idx = torch.floor(ky_scaled + dy).long() 

            x_dist = kx_scaled - x_idx.float() 
            y_dist = ky_scaled - y_idx.float()
            r = torch.sqrt(x_dist**2 + y_dist**2)
            
            w = _iternufft2d_kaiser_bessel_kernel(r, width, beta)

            x_idx_mod = x_idx % Nx_oversamp
            y_idx_mod = y_idx % Ny_oversamp
            
            for i in range(kspace_data_weighted.shape[0]):
                grid[x_idx_mod[i], y_idx_mod[i]] += kspace_data_weighted[i] * w[i]
                weight_grid[x_idx_mod[i], y_idx_mod[i]] += w[i]

    weight_grid = torch.where(weight_grid == 0, torch.ones_like(weight_grid), weight_grid)
    grid = grid / weight_grid

    img = torch.fft.ifftshift(grid)
    img = torch.fft.ifft2(img)
    img = torch.fft.fftshift(img)

    start_x = (Nx_oversamp - Nx) // 2
    start_y = (Ny_oversamp - Ny) // 2
    img_cropped = img[start_x:start_x + Nx, start_y:start_y + Ny]

    return img_cropped

def _iternufft2d_nufft2d2_forward(kx, ky, image, oversamp=2.0, width=4, beta=13.9085):
    device = image.device 
    Nx, Ny = image.shape
    Nx_oversamp = int(Nx * oversamp)
    Ny_oversamp = int(Ny * oversamp)

    pad_x = (Nx_oversamp - Nx) // 2
    pad_y = (Ny_oversamp - Ny) // 2
    image_padded = torch.zeros((Nx_oversamp, Ny_oversamp), dtype=torch.complex64, device=device)
    image_padded[pad_x:pad_x + Nx, pad_y:pad_y + Ny] = image

    kspace_cart = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image_padded)))

    kx_scaled = (kx + 0.5) * Nx_oversamp
    ky_scaled = (ky + 0.5) * Ny_oversamp

    half_width = width // 2
    kspace_data = torch.zeros(kx.shape[0], dtype=torch.complex64, device=device)
    weight_sum = torch.zeros(kx.shape[0], dtype=torch.float32, device=device) 

    for dx in range(-half_width, half_width + 1):
        for dy in range(-half_width, half_width + 1):
            x_idx = torch.floor(kx_scaled + dx).long() 
            y_idx = torch.floor(ky_scaled + dy).long()

            x_dist = kx_scaled - x_idx.float()
            y_dist = ky_scaled - y_idx.float()
            r = torch.sqrt(x_dist**2 + y_dist**2)
            
            w = _iternufft2d_kaiser_bessel_kernel(r, width, beta)

            x_idx_mod = x_idx % Nx_oversamp
            y_idx_mod = y_idx % Ny_oversamp
            
            kspace_data += kspace_cart[x_idx_mod, y_idx_mod] * w
            weight_sum += w

    weight_sum = torch.where(weight_sum == 0, torch.ones_like(weight_sum), weight_sum)
    kspace_data /= weight_sum

    return kspace_data

# --- Core Operator Classes ---

class NUFFTOperator(Operator):
    """NUFFT Operator using PyTorch for 2D or 3D data."""
    def __init__(self, k_trajectory, image_shape, device='cpu', **kwargs_2d_nufft):
        self.image_shape = image_shape
        self.device = torch.device(device) 
        
        if not isinstance(k_trajectory, torch.Tensor):
            k_trajectory = torch.tensor(k_trajectory, dtype=torch.float32)
        self.k_trajectory = k_trajectory.to(self.device)

        self.dimensionality = len(image_shape)
        self.kwargs_2d_nufft = kwargs_2d_nufft

        if self.dimensionality == 3:
            coords_z = torch.linspace(-0.5, 0.5, image_shape[0], device=self.device) 
            coords_y = torch.linspace(-0.5, 0.5, image_shape[1], device=self.device) 
            coords_x = torch.linspace(-0.5, 0.5, image_shape[2], device=self.device) 
            
            grid_z, grid_y, grid_x = torch.meshgrid(coords_z, coords_y, coords_x, indexing='ij')
            self.grid_flat_3d = torch.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), dim=1)

    def op(self, image_data_tensor):
        image_data_tensor = torch.as_tensor(image_data_tensor, dtype=torch.complex64, device=self.device)
        if image_data_tensor.shape != self.image_shape:
             raise ValueError(f"Input image_data_tensor shape {image_data_tensor.shape} does not match expected image_shape {self.image_shape}")

        if self.dimensionality == 2:
            kx = self.k_trajectory[:, 0]
            ky = self.k_trajectory[:, 1]
            return _iternufft2d_nufft2d2_forward(kx, ky, image_data_tensor, **self.kwargs_2d_nufft)
        elif self.dimensionality == 3:
            image_flat = image_data_tensor.flatten().unsqueeze(0) 
            k_traj_normalized = self.k_trajectory 
            
            exponent_matrix = -2j * torch.pi * torch.matmul(k_traj_normalized, self.grid_flat_3d.T)
            kspace_data = torch.sum(image_flat * torch.exp(exponent_matrix), dim=1)
            return kspace_data
        else:
            raise ValueError(f"Unsupported dimensionality: {self.dimensionality}")

    def op_adj(self, k_space_data_tensor):
        k_space_data_tensor = torch.as_tensor(k_space_data_tensor, dtype=torch.complex64, device=self.device)

        if self.dimensionality == 2:
            kx = self.k_trajectory[:, 0]
            ky = self.k_trajectory[:, 1]
            return _iternufft2d_nufft2d2_adjoint(kx, ky, k_space_data_tensor, self.image_shape, **self.kwargs_2d_nufft)
        elif self.dimensionality == 3:
            k_space_data_expanded = k_space_data_tensor.unsqueeze(1)  
            k_traj_normalized = self.k_trajectory

            exponent_matrix = 2j * torch.pi * torch.matmul(k_traj_normalized, self.grid_flat_3d.T)
            image_flat = torch.sum(k_space_data_expanded * torch.exp(exponent_matrix), dim=0)
            return image_flat.reshape(self.image_shape)
        else:
            raise ValueError(f"Unsupported dimensionality: {self.dimensionality}")

class CoilSensitivityOperator(Operator):
    """Operator for applying coil sensitivities."""
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
    """
    Combines NUFFT and optionally Coil Sensitivity operations.
    If coil_operator is None, it assumes a single-coil acquisition or
    that sensitivities are incorporated elsewhere (e.g. within NUFFT via density comp).
    """
    def __init__(self, nufft_operator: NUFFTOperator, coil_operator: CoilSensitivityOperator = None, num_coils_if_no_sens: int = None):
        self.nufft_operator = nufft_operator
        self.coil_operator = coil_operator
        self.num_coils_if_no_sens = num_coils_if_no_sens
        self.device = nufft_operator.device 
        self.image_shape = nufft_operator.image_shape 

        if self.coil_operator is None and self.num_coils_if_no_sens is None:
            raise ValueError("If coil_operator is None, num_coils_if_no_sens must be provided.")
        
        if self.coil_operator is not None and self.num_coils_if_no_sens is not None:
            # Optional: could warn or raise if both are provided, as num_coils_if_no_sens might be redundant
            # print("Warning: Both coil_operator and num_coils_if_no_sens provided. num_coils_if_no_sens will be ignored when coil_operator is present.")
            pass


    def op(self, image_data_tensor):
        image_data_tensor = torch.as_tensor(image_data_tensor, dtype=torch.complex64, device=self.device)
        if image_data_tensor.shape != self.image_shape:
             raise ValueError(f"Input image_data_tensor shape {image_data_tensor.shape} does not match expected image_shape {self.image_shape}")

        if self.coil_operator is None:
            # Single image input, NUFFT directly, then expand to num_coils_if_no_sens
            k_per_coil = self.nufft_operator.op(image_data_tensor) # Shape: (num_k_points,)
            # Expand to (num_coils_if_no_sens, num_k_points)
            return k_per_coil.unsqueeze(0).expand(self.num_coils_if_no_sens, -1)
        else:
            # SENSE-like operation
            coil_output = self.coil_operator.op(image_data_tensor) 
            num_coils = coil_output.shape[0]
            num_k_points = self.nufft_operator.k_trajectory.shape[0] 
            
            k_space_result = torch.zeros((num_coils, num_k_points), dtype=coil_output.dtype, device=self.device)
            
            for c in range(num_coils):
                k_space_result[c, :] = self.nufft_operator.op(coil_output[c])
            return k_space_result

    def op_adj(self, k_space_data_coils_tensor):
        k_space_data_coils_tensor = torch.as_tensor(k_space_data_coils_tensor, dtype=torch.complex64, device=self.device)
        # k_space_data_coils_tensor shape: (num_coils, num_k_points)

        if self.coil_operator is None:
            num_coils_from_data = k_space_data_coils_tensor.shape[0]
            if self.num_coils_if_no_sens is not None and num_coils_from_data != self.num_coils_if_no_sens:
                raise ValueError(f"Mismatch between k_space_data_coils_tensor.shape[0] ({num_coils_from_data}) and num_coils_if_no_sens ({self.num_coils_if_no_sens})")

            # image_shape is already self.image_shape from __init__
            accumulated_image = torch.zeros(self.image_shape, dtype=k_space_data_coils_tensor.dtype, device=self.device)
            for c in range(num_coils_from_data): # Use actual number of coils from data
                accumulated_image += self.nufft_operator.op_adj(k_space_data_coils_tensor[c])
            return accumulated_image
        else:
            # SENSE-like adjoint operation
            num_coils = k_space_data_coils_tensor.shape[0]
            # image_shape for single coil is self.image_shape
            
            coil_adj_output_shape = (num_coils,) + self.image_shape
            coil_adj_output = torch.zeros(coil_adj_output_shape, dtype=k_space_data_coils_tensor.dtype, device=self.device)
            
            for c in range(num_coils):
                coil_adj_output[c, ...] = self.nufft_operator.op_adj(k_space_data_coils_tensor[c])
                
            return self.coil_operator.op_adj(coil_adj_output)

# Example of how one might set up kwargs_2d_nufft for NUFFTOperator
# default_kwargs_2d_nufft = {
#     'oversamp': 2.0,
#     'width': 4,
#     'beta': 13.9085 
# }
