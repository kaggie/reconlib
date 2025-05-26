"""Module for defining Operator classes for MRI reconstruction."""

import torch
import numpy as np
from abc import ABC, abstractmethod
from scipy.special import i0 # For _iternufft2d_kaiser_bessel_kernel
from scipy.signal.windows import tukey # For SlidingWindowNUFFTOperator

# Conditional imports for external NUFFT libraries
try:
    from pynufft import NUFFT as pynufft_NUFFT_lib
    PYNUFFT_AVAILABLE = True
except ImportError:
    PYNUFFT_AVAILABLE = False

try:
    import sigpy
    from sigpy.linop import NUFFT as sigpy_NUFFT_op 
    SIGPY_AVAILABLE = True
except ImportError:
    SIGPY_AVAILABLE = False


# Operator Base Class
class Operator(ABC):
    @abstractmethod
    def op(self, x): pass
    @abstractmethod
    def op_adj(self, y): pass

# --- Internal 2D NUFFT Helper Functions ---
def _iternufft2d_kaiser_bessel_kernel(r, width, beta):
    mask = r < (width / 2)
    val_inside_sqrt = torch.clamp(1 - (2 * r[mask] / width)**2, min=0.0)
    z = torch.sqrt(val_inside_sqrt)
    kb = torch.zeros_like(r)
    kb_numpy_values = i0(beta * z.cpu().numpy())
    kb[mask] = torch.from_numpy(kb_numpy_values.astype(np.float32)).to(r.device) / float(i0(beta))
    return kb

def _iternufft2d_estimate_density_compensation(kx, ky):
    radius = torch.sqrt(kx**2 + ky**2)
    dcf = radius + 1e-3
    dcf /= dcf.max()
    return dcf

def _iternufft2d_nufft2d2_adjoint(kx, ky, kspace_data, image_shape, oversamp=2.0, width=4, beta=13.9085):
    device = kx.device
    Nx, Ny = image_shape
    Nx_oversamp, Ny_oversamp = int(Nx * oversamp), int(Ny * oversamp)
    kx_scaled, ky_scaled = (kx + 0.5) * Nx_oversamp, (ky + 0.5) * Ny_oversamp
    dcf = _iternufft2d_estimate_density_compensation(kx, ky).to(device)
    kspace_data_weighted = kspace_data * dcf
    grid = torch.zeros((Nx_oversamp, Ny_oversamp), dtype=torch.complex64, device=device)
    weight_grid = torch.zeros((Nx_oversamp, Ny_oversamp), dtype=torch.float32, device=device)
    half_width = width // 2
    for dx in range(-half_width, half_width + 1):
        for dy in range(-half_width, half_width + 1):
            x_idx, y_idx = torch.floor(kx_scaled + dx).long(), torch.floor(ky_scaled + dy).long()
            x_dist, y_dist = kx_scaled - x_idx.float(), ky_scaled - y_idx.float()
            r = torch.sqrt(x_dist**2 + y_dist**2)
            w = _iternufft2d_kaiser_bessel_kernel(r, width, beta)
            x_idx_mod, y_idx_mod = x_idx % Nx_oversamp, y_idx % Ny_oversamp
            for i in range(kspace_data_weighted.shape[0]):
                grid[x_idx_mod[i], y_idx_mod[i]] += kspace_data_weighted[i] * w[i]
                weight_grid[x_idx_mod[i], y_idx_mod[i]] += w[i]
    weight_grid = torch.where(weight_grid == 0, torch.ones_like(weight_grid), weight_grid)
    grid = grid / weight_grid
    img = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(grid)))
    start_x, start_y = (Nx_oversamp - Nx) // 2, (Ny_oversamp - Ny) // 2
    return img[start_x:start_x + Nx, start_y:start_y + Ny]

def _iternufft2d_nufft2d2_forward(kx, ky, image, oversamp=2.0, width=4, beta=13.9085):
    device = image.device 
    Nx, Ny = image.shape
    Nx_oversamp, Ny_oversamp = int(Nx * oversamp), int(Ny * oversamp)
    pad_x, pad_y = (Nx_oversamp - Nx) // 2, (Ny_oversamp - Ny) // 2
    image_padded = torch.zeros((Nx_oversamp, Ny_oversamp), dtype=torch.complex64, device=device)
    image_padded[pad_x:pad_x + Nx, pad_y:pad_y + Ny] = image
    kspace_cart = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image_padded)))
    kx_scaled, ky_scaled = (kx + 0.5) * Nx_oversamp, (ky + 0.5) * Ny_oversamp
    half_width = width // 2
    kspace_data = torch.zeros(kx.shape[0], dtype=torch.complex64, device=device)
    weight_sum = torch.zeros(kx.shape[0], dtype=torch.float32, device=device) 
    for dx in range(-half_width, half_width + 1):
        for dy in range(-half_width, half_width + 1):
            x_idx, y_idx = torch.floor(kx_scaled + dx).long(), torch.floor(ky_scaled + dy).long()
            x_dist, y_dist = kx_scaled - x_idx.float(), ky_scaled - y_idx.float()
            r = torch.sqrt(x_dist**2 + y_dist**2)
            w = _iternufft2d_kaiser_bessel_kernel(r, width, beta)
            x_idx_mod, y_idx_mod = x_idx % Nx_oversamp, y_idx % Ny_oversamp
            kspace_data += kspace_cart[x_idx_mod, y_idx_mod] * w
            weight_sum += w
    weight_sum = torch.where(weight_sum == 0, torch.ones_like(weight_sum), weight_sum)
    kspace_data /= weight_sum
    return kspace_data

class NUFFTOperator(Operator):
    """
    NUFFT Operator using PyTorch for 2D (iterative gridding) or 3D (direct NDFT or external libraries).
    For 3D NDFT (direct or external libraries), k_trajectory coordinates are assumed to be 
    normalized in [-0.5, 0.5] for each dimension if using 'direct' backend.
    For 'pynufft' or 'sigpy' backends, k_trajectory is scaled internally to [-pi, pi].
    """
    def __init__(self, k_trajectory, image_shape, device='cpu', 
                 nufft_backend_2d='iternufft2d', 
                 nufft_backend_3d='direct', 
                 **kwargs_nufft):
        self.image_shape = image_shape
        self.device = torch.device(device) 
        self.nufft_backend_2d = nufft_backend_2d
        self.nufft_backend_3d = nufft_backend_3d
        self.kwargs_nufft = kwargs_nufft

        if not isinstance(k_trajectory, torch.Tensor):
            k_trajectory = torch.tensor(k_trajectory, dtype=torch.float32)
        self.k_trajectory = k_trajectory.to(self.device)

        self.dimensionality = len(image_shape)
        self.pynufft_plan_3d = None
        self.sigpy_nufft_op_3d = None
        self.grid_flat_3d = None 

        if self.dimensionality == 3:
            if self.k_trajectory.ndim != 2 or self.k_trajectory.shape[1] != 3:
                raise ValueError(f"For 3D, k_trajectory must have shape (num_k_points, 3), got {self.k_trajectory.shape}")
            if self.nufft_backend_3d == 'pynufft':
                if PYNUFFT_AVAILABLE:
                    try:
                        self.pynufft_plan_3d = pynufft_NUFFT_lib()
                        Nd = tuple(self.image_shape) 
                        oversamp_factor = self.kwargs_nufft.get('oversamp', 2.0)
                        kernel_width = self.kwargs_nufft.get('width', 6) 
                        Kd = tuple(int(i * oversamp_factor) for i in Nd)
                        Jd = tuple(kernel_width for _ in Nd)
                        k_traj_pynufft = self.k_trajectory.cpu().numpy() * (2 * np.pi)
                        pynufft_plan_kwargs = {k: v for k, v in self.kwargs_nufft.items() if k in ['batch', 'device_id', 'fft_type']}
                        self.pynufft_plan_3d.plan(k_traj_pynufft, Nd, Kd, Jd, **pynufft_plan_kwargs)
                        print("INFO: Using pynufft for 3D NUFFT.")
                    except Exception as e:
                        print(f"WARNING: pynufft planning failed: {e}. Falling back to 'direct' NDFT for 3D.")
                        self.nufft_backend_3d = 'direct'
                else:
                    print("WARNING: pynufft library not found for 3D NUFFT. Falling back to 'direct' NDFT.")
                    self.nufft_backend_3d = 'direct'
            elif self.nufft_backend_3d == 'sigpy':
                if SIGPY_AVAILABLE:
                    try:
                        k_traj_sigpy = self.k_trajectory.cpu().numpy() * (2 * np.pi)
                        sigpy_kwargs = self.kwargs_nufft.copy()
                        if 'oversamp' in sigpy_kwargs and 'osf' not in sigpy_kwargs:
                            sigpy_kwargs['osf'] = sigpy_kwargs.pop('oversamp')
                        self.sigpy_nufft_op_3d = sigpy_NUFFT_op(oshape=self.image_shape, coord=k_traj_sigpy, **sigpy_kwargs)
                        print("INFO: Using SigPy for 3D NUFFT.")
                    except Exception as e:
                        print(f"WARNING: SigPy NUFFT initialization failed: {e}. Falling back to 'direct' NDFT for 3D.")
                        self.nufft_backend_3d = 'direct'
                else:
                    print("WARNING: SigPy library not found for 3D NUFFT. Falling back to 'direct' NDFT.")
                    self.nufft_backend_3d = 'direct'
            if self.nufft_backend_3d == 'direct': 
                coords_z = torch.linspace(-0.5, 0.5, image_shape[0], device=self.device, dtype=torch.float32)
                coords_y = torch.linspace(-0.5, 0.5, image_shape[1], device=self.device, dtype=torch.float32)
                coords_x = torch.linspace(-0.5, 0.5, image_shape[2], device=self.device, dtype=torch.float32)
                grid_z, grid_y, grid_x = torch.meshgrid(coords_z, coords_y, coords_x, indexing='ij')
                self.grid_flat_3d = torch.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), dim=1)
        elif self.dimensionality == 2:
             if self.k_trajectory.ndim != 2 or self.k_trajectory.shape[1] != 2:
                raise ValueError(f"For 2D NUFFT, k_trajectory must have shape (num_k_points, 2), got {self.k_trajectory.shape}")
        else:
            raise ValueError(f"Unsupported dimensionality: {self.dimensionality}. Must be 2 or 3.")

    def op(self, image_data_tensor):
        image_data_tensor = torch.as_tensor(image_data_tensor, dtype=torch.complex64, device=self.device)
        if image_data_tensor.shape != self.image_shape:
             raise ValueError(f"Input image_data_tensor shape {image_data_tensor.shape} does not match expected {self.image_shape}")
        if self.dimensionality == 2:
            kx, ky = self.k_trajectory[:, 0], self.k_trajectory[:, 1]
            nufft_params_2d = {k: v for k, v in self.kwargs_nufft.items() if k in ['oversamp', 'width', 'beta']}
            return _iternufft2d_nufft2d2_forward(kx, ky, image_data_tensor, **nufft_params_2d)
        elif self.dimensionality == 3:
            if self.nufft_backend_3d == 'pynufft' and self.pynufft_plan_3d:
                img_np = image_data_tensor.detach().cpu().numpy()
                k_space_np = self.pynufft_plan_3d.op(img_np)
                return torch.from_numpy(k_space_np).to(self.device)
            elif self.nufft_backend_3d == 'sigpy' and self.sigpy_nufft_op_3d:
                img_np = image_data_tensor.detach().cpu().numpy()
                k_space_np = self.sigpy_nufft_op_3d.forward(img_np)
                return torch.from_numpy(k_space_np).to(self.device)
            elif self.nufft_backend_3d == 'direct':
                image_flat = image_data_tensor.flatten().unsqueeze(0) 
                dot_product_matrix = torch.matmul(self.k_trajectory, self.grid_flat_3d.T)
                exponent_matrix = -2j * torch.pi * dot_product_matrix
                kspace_data = torch.sum(image_flat * torch.exp(exponent_matrix), dim=1)
                return kspace_data
            else: raise RuntimeError("3D NUFFT backend not properly initialized or fallback failed.")
        else: raise ValueError(f"Unsupported dimensionality: {self.dimensionality}")

    def op_adj(self, k_space_data_tensor, output_voxel_coords_flat=None): # Added output_voxel_coords_flat
        k_space_data_tensor = torch.as_tensor(k_space_data_tensor, dtype=torch.complex64, device=self.device)
        if self.dimensionality == 2:
            kx, ky = self.k_trajectory[:, 0], self.k_trajectory[:, 1]
            nufft_params_2d = {k: v for k, v in self.kwargs_nufft.items() if k in ['oversamp', 'width', 'beta']}
            return _iternufft2d_nufft2d2_adjoint(kx, ky, k_space_data_tensor, self.image_shape, **nufft_params_2d)
        elif self.dimensionality == 3:
            if self.nufft_backend_3d == 'pynufft' and self.pynufft_plan_3d:
                if output_voxel_coords_flat is not None:
                    print("Warning: NUFFTOperator.op_adj with pynufft backend does not support output_voxel_coords_flat. Ignoring.")
                k_space_np = k_space_data_tensor.detach().cpu().numpy()
                img_np = self.pynufft_plan_3d.adj(k_space_np)
                return torch.from_numpy(img_np).to(self.device)
            elif self.nufft_backend_3d == 'sigpy' and self.sigpy_nufft_op_3d:
                if output_voxel_coords_flat is not None:
                    print("Warning: NUFFTOperator.op_adj with sigpy backend does not support output_voxel_coords_flat. Ignoring.")
                k_space_np = k_space_data_tensor.detach().cpu().numpy()
                img_np = self.sigpy_nufft_op_3d.adjoint(k_space_np)
                return torch.from_numpy(img_np).to(self.device)
            elif self.nufft_backend_3d == 'direct':
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
                    return image_flat # Return flat vector as per new functionality
            else: raise RuntimeError("3D NUFFT backend not properly initialized or fallback failed for adjoint.")
        else: raise ValueError(f"Unsupported dimensionality: {self.dimensionality}")

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
                # Call op_adj of base NUFFT for each coil. If output_voxel_coords_flat is relevant, 
                # this base call might need to pass it, but MRIForwardOperator doesn't know about it.
                # This implies SlidingWindowNUFFTOperator should wrap the MRIForwardOperator,
                # or NUFFTOperator's op_adj needs to be context-aware if it's part of MRIForwardOperator.
                # For now, assuming standard op_adj call.
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
            # For 2D, or non-direct 3D, op_adj will just call base_nufft_operator.op_adj
        
        if self.base_nufft_operator.nufft_backend_3d != 'direct' and self.dimensionality == 3:
            print("Warning: SlidingWindowNUFFTOperator op_adj is optimized for 'direct' 3D NDFT backend. "
                  "Using other backends will call the base operator's op_adj directly.")

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
            # scipy.signal.windows.tukey: alpha is ratio of taper to constant section
            # For PyTorch, we might need to implement it or use available if any.
            # Simple implementation for now, or use scipy if available.
            # Using scipy.signal.windows.tukey for robustness.
            win_1d = torch.from_numpy(tukey(dim_size, alpha=alpha, sym=True).astype(np.float32))
            individual_windows.append(win_1d)
        
        if len(block_shape_tuple) == 1:
            return individual_windows[0]
        elif len(block_shape_tuple) == 2:
            return torch.outer(individual_windows[0], individual_windows[1])
        elif len(block_shape_tuple) == 3:
            # (H,W,D) -> W varies fastest. If block_shape is (D,H,W) -> tukey for D, H, W
            # Dims for outer product: (D) outer (H) -> (D,H). Then (D,H).flatten outer (W) -> (D*H, W) -> reshape(D,H,W)
            # Or, more generally:
            window_nd = individual_windows[0]
            for i in range(1, len(individual_windows)):
                window_nd = torch.outer(window_nd.flatten(), individual_windows[i]).reshape(*window_nd.shape, individual_windows[i].shape[0])
            # Final shape should match block_shape_tuple. This needs careful reshaping.
            # Example: (D,H,W). win_d (D), win_h (H), win_w (W)
            # window = win_d[:, None, None] * win_h[None, :, None] * win_w[None, None, :]
            # This is element-wise broadcast multiplication.
            current_window = torch.ones(block_shape_tuple, dtype=torch.float32)
            for d_idx, win_1d in enumerate(individual_windows):
                view_shape = [1] * len(block_shape_tuple)
                view_shape[d_idx] = block_shape_tuple[d_idx]
                current_window *= win_1d.view(view_shape)
            return current_window
        else:
            raise ValueError("SlidingWindow currently supports 1D, 2D, or 3D blocks.")


    def op(self, image_data_tensor):
        # print("SlidingWindowNUFFTOperator.op currently calls base operator directly.")
        return self.base_nufft_operator.op(image_data_tensor)

    def op_adj(self, k_space_data_tensor):
        if self.dimensionality != 3 or self.base_nufft_operator.nufft_backend_3d != 'direct':
            # print("Warning: SlidingWindowNUFFTOperator.op_adj is optimized for 3D 'direct' NDFT. "
            #       "Calling base operator's op_adj for current configuration.")
            return self.base_nufft_operator.op_adj(k_space_data_tensor) # No custom args

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
