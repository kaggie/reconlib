"""Module for defining Reconstructor classes."""

import torch
import numpy as np # Retained for user's 3D gridding code, though torch.i0 is used
from abc import ABC, abstractmethod
from reconlib.utils import calculate_density_compensation
# For _gr_2d_kaiser_bessel_kernel, if using the scipy version
from scipy.special import i0 as scipy_i0
# Import ADMM to check its instance type
from reconlib.optimizers import ADMM 


# --- Helper functions for RegriddingReconstructor ---

# User's 3D Kaiser-Bessel kernel (renamed)
# Note: This version uses torch.i0, requires PyTorch 1.7+
def _gr_3d_kaiser_bessel_kernel(dist, width, beta, device='cpu'):
    """
    Computes Kaiser-Bessel kernel values.
    dist: Euclidean distance from grid point.
    width: Kernel width.
    beta: Kernel beta parameter.
    """
    mask = dist < (width / 2)
    vals = torch.zeros_like(dist, device=device)
    arg = beta * torch.sqrt(1 - (2 * dist[mask] / width)**2)
    vals[mask] = torch.i0(arg) / torch.i0(torch.tensor(beta, device=device)) 
    return vals

# User's 3D gridding function (renamed)
def _gr_grid_3d_noncartesian(kspace_samples, signal, grid_size_tuple, width, beta, device='cpu'):
    """
    Grids 3D non-Cartesian k-space samples to a Cartesian grid using Kaiser-Bessel kernel.
    kspace_samples: (N, 3) tensor of kx, ky, kz coordinates, scaled to grid units.
    signal: (N,) tensor of complex k-space signal values.
    grid_size_tuple: Tuple (Gx, Gy, Gz) for grid dimensions.
    width: Kernel width.
    beta: Kernel beta parameter.
    """
    kspace_samples = torch.as_tensor(kspace_samples, device=device, dtype=torch.float32)
    signal = torch.as_tensor(signal, device=device, dtype=torch.complex64)

    Gx, Gy, Gz = grid_size_tuple
    grid = torch.zeros(grid_size_tuple, dtype=torch.complex64, device=device)
    weights = torch.zeros(grid_size_tuple, dtype=torch.float32, device=device)
    
    half_width = width / 2
    
    for i in range(kspace_samples.shape[0]):
        kx, ky, kz = kspace_samples[i]
        s = signal[i]
        
        min_x = int(torch.floor(kx - half_width).item())
        max_x = int(torch.ceil(kx + half_width).item())
        min_y = int(torch.floor(ky - half_width).item())
        max_y = int(torch.ceil(ky + half_width).item())
        min_z = int(torch.floor(kz - half_width).item())
        max_z = int(torch.ceil(kz + half_width).item())
        
        for gx in range(min_x, max_x + 1):
            for gy in range(min_y, max_y + 1):
                for gz in range(min_z, max_z + 1):
                    gxi, gyi, gzi = gx % Gx, gy % Gy, gz % Gz
                    dist = torch.sqrt((kx - gx)**2 + (ky - gy)**2 + (kz - gz)**2)
                    w = _gr_3d_kaiser_bessel_kernel(dist.unsqueeze(0), width, beta, device=device).squeeze()
                    if w > 1e-9: 
                        grid[gxi, gyi, gzi] += s * w
                        weights[gxi, gyi, gzi] += w
                        
    return grid / (weights + 1e-9) 

# Adapted 2D Kaiser-Bessel kernel (from operators.py, uses scipy_i0)
def _gr_2d_kaiser_bessel_kernel(r, width, beta, device='cpu'):
    r = torch.as_tensor(r, device=device, dtype=torch.float32)
    mask = r < (width / 2)
    val_inside_sqrt = torch.clamp(1 - (2 * r[mask] / width)**2, min=0.0)
    z = torch.sqrt(val_inside_sqrt)
    kb = torch.zeros_like(r)
    kb_numpy_values = scipy_i0(beta * z.cpu().numpy()) 
    kb[mask] = torch.from_numpy(kb_numpy_values.astype(np.float32)).to(device) / float(scipy_i0(beta))
    return kb

# Adapted 2D gridding function (from operators.py _iternufft2d_nufft2d2_adjoint)
def _gr_grid_2d_noncartesian(kspace_samples_2d, signal_2d, grid_size_2d_tuple, width, beta, device='cpu'):
    kspace_samples_2d = torch.as_tensor(kspace_samples_2d, device=device, dtype=torch.float32)
    signal_2d = torch.as_tensor(signal_2d, device=device, dtype=torch.complex64)
    Gx, Gy = grid_size_2d_tuple 
    kx_scaled = kspace_samples_2d[:,0] 
    ky_scaled = kspace_samples_2d[:,1]
    grid = torch.zeros((Gx, Gy), dtype=torch.complex64, device=device)
    weights_grid = torch.zeros((Gx, Gy), dtype=torch.float32, device=device)
    half_width = width / 2.0 
    for i in range(kspace_samples_2d.shape[0]):
        kx_s, ky_s = kx_scaled[i], ky_scaled[i] 
        sig_s = signal_2d[i]
        min_gx = int(torch.floor(kx_s - half_width).item())
        max_gx = int(torch.ceil(kx_s + half_width).item())
        min_gy = int(torch.floor(ky_s - half_width).item())
        max_gy = int(torch.ceil(ky_s + half_width).item())
        for gx_idx_img_coord in range(min_gx, max_gx + 1):
            for gy_idx_img_coord in range(min_gy, max_gy + 1):
                dist = torch.sqrt((kx_s - gx_idx_img_coord)**2 + (ky_s - gy_idx_img_coord)**2)
                w = _gr_2d_kaiser_bessel_kernel(dist.unsqueeze(0), width, beta, device=device).squeeze()
                if w > 1e-9: 
                    gx_target = gx_idx_img_coord % Gx
                    gy_target = gy_idx_img_coord % Gy
                    grid[gx_target, gy_target] += sig_s * w
                    weights_grid[gx_target, gy_target] += w
    return grid / (weights_grid + 1e-9)


class Reconstructor(ABC):
    """
    Abstract base class for reconstructors.
    Defines the interface for the reconstruct method.
    """
    @abstractmethod
    def reconstruct(self, mri_data_obj, initial_guess=None, verbose=False):
        """
        Performs the reconstruction.
        Args:
            mri_data_obj: An MRIData object containing the data to reconstruct.
            initial_guess: Optional initial guess for the image.
            verbose: Optional flag for optimizer verbosity.
        Returns:
            The reconstructed image (PyTorch tensor).
        """
        pass

class IterativeReconstructor(Reconstructor):
    """
    Orchestrates iterative image reconstruction using a forward operator,
    regularizer, and optimizer.
    """
    def __init__(self, forward_operator, regularizer, optimizer):
        self.forward_operator = forward_operator
        self.regularizer = regularizer # This is prox_regularizer for ADMM
        self.optimizer = optimizer

    def reconstruct(self, mri_data_obj, initial_guess=None, verbose=False):
        if not hasattr(self.forward_operator, 'device'):
            raise AttributeError("Forward operator must have a 'device' attribute.")
        device = self.forward_operator.device
        k_space_data_np = mri_data_obj.k_space_data
        k_space_data = torch.from_numpy(k_space_data_np).to(dtype=torch.complex64, device=device)

        if initial_guess is not None:
            if not isinstance(initial_guess, torch.Tensor):
                initial_guess = torch.tensor(initial_guess, device=device)
            initial_guess = initial_guess.to(device=device, dtype=torch.complex64)
            if hasattr(self.forward_operator, 'image_shape') and initial_guess.shape != self.forward_operator.image_shape:
                raise ValueError(f"Provided initial_guess shape {initial_guess.shape} does not match "
                                 f"forward_operator.image_shape {self.forward_operator.image_shape}.")
        else:
            initial_guess = self.forward_operator.op_adj(k_space_data).to(dtype=torch.complex64)

        if hasattr(self.optimizer, 'verbose') and verbose is not None:
            self.optimizer.verbose = verbose
        
        # Default call for optimizers like FISTA
        reconstructed_image = self.optimizer.solve(
            k_space_data=k_space_data,
            forward_op=self.forward_operator,
            regularizer=self.regularizer, # For FISTA, this is the only regularizer
            initial_guess=initial_guess
        )
        return reconstructed_image

class RegriddingReconstructor(Reconstructor):
    """
    Performs image reconstruction using Kaiser-Bessel gridding.
    Supports 2D and 3D non-Cartesian k-space data.
    """
    def __init__(self, width_2d=4.0, beta_2d=13.9085, width_3d=3.0, beta_3d=13.855, grid_oversampling_factor=2.0):
        self.width_2d = width_2d
        self.beta_2d = beta_2d
        self.width_3d = width_3d
        self.beta_3d = beta_3d
        self.grid_oversampling_factor = grid_oversampling_factor

    def reconstruct(self, mri_data_obj, density_compensation_method='radial_simple', device='cpu'):
        k_space_data_np = mri_data_obj.k_space_data
        k_trajectory_np = mri_data_obj.k_trajectory
        image_shape = mri_data_obj.image_shape

        k_space_data = torch.from_numpy(k_space_data_np).to(dtype=torch.complex64, device=device)
        k_trajectory = torch.from_numpy(k_trajectory_np).to(dtype=torch.float32, device=device)

        num_coils = k_space_data.shape[0]
        Ndims = k_trajectory.shape[1]

        if len(image_shape) != Ndims:
             raise ValueError(f"image_shape dimension {len(image_shape)} does not match k_trajectory dimension {Ndims}")

        target_grid_shape = tuple(int(s * self.grid_oversampling_factor) for s in image_shape)
        dcf = calculate_density_compensation(
            k_trajectory, image_shape, 
            method=density_compensation_method, 
            device=device
        ).to(dtype=torch.complex64) 

        coil_images_sum_sq = torch.zeros(image_shape, dtype=torch.float32, device=device)
        for c in range(num_coils):
            signal_c = k_space_data[c, :] * dcf 
            scaled_k_trajectory = torch.zeros_like(k_trajectory)
            for dim_idx in range(Ndims):
                 scaled_k_trajectory[:, dim_idx] = (k_trajectory[:, dim_idx] + 0.5) * target_grid_shape[dim_idx]

            if Ndims == 2:
                gridded_k_coil_c = _gr_grid_2d_noncartesian(
                    scaled_k_trajectory, signal_c, 
                    target_grid_shape, self.width_2d, self.beta_2d, device
                )
            elif Ndims == 3:
                gridded_k_coil_c = _gr_grid_3d_noncartesian(
                    scaled_k_trajectory, signal_c, 
                    target_grid_shape, self.width_3d, self.beta_3d, device
                )
            else:
                raise ValueError(f"Unsupported number of dimensions: {Ndims}. Must be 2 or 3.")

            img_coil_c_oversampled = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(gridded_k_coil_c), norm='ortho'))
            start_indices = [(tg - os) // 2 for tg, os in zip(target_grid_shape, image_shape)]
            
            if Ndims == 2:
                img_coil_c_cropped = img_coil_c_oversampled[
                    start_indices[0] : start_indices[0] + image_shape[0],
                    start_indices[1] : start_indices[1] + image_shape[1]
                ]
            elif Ndims == 3:
                img_coil_c_cropped = img_coil_c_oversampled[
                    start_indices[0] : start_indices[0] + image_shape[0],
                    start_indices[1] : start_indices[1] + image_shape[1],
                    start_indices[2] : start_indices[2] + image_shape[2]
                ]
            else: 
                 img_coil_c_cropped = img_coil_c_oversampled 
            coil_images_sum_sq += torch.abs(img_coil_c_cropped)**2
        final_image = torch.sqrt(coil_images_sum_sq)
        return final_image

class ConstrainedReconstructor(IterativeReconstructor):
    """
    Extends IterativeReconstructor to handle optimizers (like ADMM) that can 
    incorporate quadratic constraint regularizers directly into their sub-problems,
    in addition to a main proximal regularizer.
    """
    def __init__(self, forward_operator, optimizer, prox_regularizer=None, constraint_regularizers_list=None):
        """
        Initializes the ConstrainedReconstructor.

        Args:
            forward_operator: An instance of a forward operator.
            optimizer: An instance of an optimizer (e.g., ADMM).
            prox_regularizer: The main regularizer that has a .prox() method (e.g., L1, TV).
                              Can be None if only quadratic constraints are used.
            constraint_regularizers_list (list, optional): A list of regularizers
                (e.g., GradientMatchingRegularizer instances) that define quadratic terms
                or other terms handled directly in the optimizer's x-update.
        """
        super().__init__(forward_operator, prox_regularizer, optimizer) # Stores prox_regularizer as self.regularizer
        self.constraint_regularizers_list = constraint_regularizers_list if constraint_regularizers_list is not None else []

    def reconstruct(self, mri_data_obj, initial_guess=None, verbose=False):
        """
        Performs the reconstruction.
        If the optimizer is ADMM, it passes both prox_regularizer and constraint_regularizers_list.
        Otherwise, it behaves like IterativeReconstructor and raises an error if constraint_regularizers_list is non-empty.
        """
        # Initial data handling (device, k-space, initial_guess) is same as IterativeReconstructor
        if not hasattr(self.forward_operator, 'device'):
            raise AttributeError("Forward operator must have a 'device' attribute.")
        device = self.forward_operator.device
        k_space_data_np = mri_data_obj.k_space_data
        k_space_data = torch.from_numpy(k_space_data_np).to(dtype=torch.complex64, device=device)

        if initial_guess is not None:
            if not isinstance(initial_guess, torch.Tensor):
                initial_guess = torch.tensor(initial_guess, device=device)
            initial_guess = initial_guess.to(device=device, dtype=torch.complex64)
            if hasattr(self.forward_operator, 'image_shape') and initial_guess.shape != self.forward_operator.image_shape:
                raise ValueError(f"Provided initial_guess shape {initial_guess.shape} does not match "
                                 f"forward_operator.image_shape {self.forward_operator.image_shape}.")
        else:
            initial_guess = self.forward_operator.op_adj(k_space_data).to(dtype=torch.complex64)

        if hasattr(self.optimizer, 'verbose') and verbose is not None:
            self.optimizer.verbose = verbose
        
        # Check if the optimizer is ADMM and can handle quadratic_plus_prox_regularizers
        if isinstance(self.optimizer, ADMM):
            reconstructed_image = self.optimizer.solve(
                k_space_data=k_space_data,
                forward_op=self.forward_operator,
                prox_regularizer=self.regularizer, # self.regularizer is prox_regularizer from __init__
                initial_guess=initial_guess,
                quadratic_plus_prox_regularizers=self.constraint_regularizers_list
            )
        else:
            # For other optimizers (like FISTA), ensure no constraint_regularizers are passed if not supported
            if self.constraint_regularizers_list: # If list is not empty
                raise ValueError(
                    f"Optimizer {type(self.optimizer).__name__} does not support "
                    "constraint_regularizers_list. Use ADMM or an optimizer "
                    "adapted for such regularizers."
                )
            # Call the original solve method from IterativeReconstructor (or Optimizer base if FISTA overrode it)
            # IterativeReconstructor's reconstruct method calls self.optimizer.solve with 4 args.
            reconstructed_image = super().reconstruct(mri_data_obj, initial_guess=initial_guess, verbose=verbose)
            # The above call to super().reconstruct will internally call self.optimizer.solve.
            # This is fine if the base IterativeReconstructor.reconstruct handles the standard optimizer.solve call.
            # Let's directly call self.optimizer.solve for clarity, ensuring it matches the expected signature
            # for non-ADMM optimizers.
            # reconstructed_image = self.optimizer.solve(
            # k_space_data=k_space_data,
            # forward_op=self.forward_operator,
            # regularizer=self.regularizer, 
            # initial_guess=initial_guess
            # )
            # The super().reconstruct call is actually more robust as it re-does the data prep,
            # but for this specific path, it might be okay.
            # However, the base IterativeReconstructor.reconstruct already calls self.optimizer.solve correctly.

        return reconstructed_image
