"""Module for defining Regularizer classes for MRI reconstruction."""

import torch
import numpy as np
from abc import ABC, abstractmethod

# Attempt to import pytorch_wavelets
try:
    from pytorch_wavelets import DTCWTForward, DTCWTInverse, DWTForward, DWTInverse
    PYTORCH_WAVELETS_AVAILABLE = True
except ImportError:
    PYTORCH_WAVELETS_AVAILABLE = False

# Attempt to import pywt
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

# Operator class is needed for SparsityTransform
try:
    # Assuming reconlib.operators is in python path or local
    from reconlib.operators import Operator 
except ImportError:
    # Fallback placeholder if reconlib.operators is not found (e.g. during isolated testing)
    class Operator(ABC):
        @abstractmethod
        def op(self, x): pass
        @abstractmethod
        def op_adj(self, y): pass
    # print("Warning: reconlib.operators.Operator not found. Using a placeholder Operator base class.")


class Regularizer(ABC):
    """
    Abstract base class for regularizers.
    Defines the interface for the proximal operator.
    """
    @abstractmethod
    def prox(self, x, step_size):
        """
        Proximal operator.

        Args:
            x: Input data (PyTorch tensor).
            step_size: Step size parameter. The actual thresholding value is typically
                       calculated as self.lambda_param * step_size.

        Returns:
            Result of the proximal operation (PyTorch tensor).
        """
        pass

class SparsityTransform(Operator):
    """
    Applies a sparsity transform (e.g., wavelet) and its inverse.
    Inherits from Operator to use op() for forward and op_adj() for inverse.
    Supports pytorch_wavelets and pywt (as fallback).
    """
    def __init__(self, image_shape, transform_type='wavelet', wavelet_name='db4', 
                 level=None, axes=None, device='cpu', **kwargs):
        self.image_shape = image_shape
        self.transform_type = transform_type
        self.wavelet_name = wavelet_name
        self.level = level # Can be None initially
        self.axes = axes if axes is not None else tuple(range(len(image_shape)))
        self.device = torch.device(device)
        self.kwargs = kwargs # For DTCWTForward primarily

        self.fwd_transform_ptw = None
        self.inv_transform_ptw = None
        self.use_pytorch_wavelets_backend = False
        self.use_pywt_backend = False
        self._pywt_coeffs_structure_template = None # For pywt reconstruction

        if self.transform_type == 'wavelet':
            min_dim_for_level_calc = min(self.image_shape[ax] for ax in self.axes) if self.axes else min(self.image_shape)

            if PYTORCH_WAVELETS_AVAILABLE:
                self.use_pytorch_wavelets_backend = True
                # print("SparsityTransform: Using pytorch_wavelets backend.")
                if self.level is None:
                    if PYWT_AVAILABLE:
                        try: self.level = pywt.dwt_max_level(min_dim_for_level_calc, pywt.Wavelet(self.wavelet_name))
                        except: self.level = 3 # Default
                    else: self.level = 3 # Default
                
                J = self.level
                is_3d_image = len(self.image_shape) == 3

                if is_3d_image and len(self.axes) == 3: # Full 3D DWT
                    self.fwd_transform_ptw = DWTForward(J=J, wave=wavelet_name, mode='symmetric', **self.kwargs).to(self.device)
                    self.inv_transform_ptw = DWTInverse(wave=wavelet_name, mode='symmetric', **self.kwargs).to(self.device)
                elif not is_3d_image and len(self.axes) == 2: # 2D
                    try: # Try DTCWT first for 2D
                        self.fwd_transform_ptw = DTCWTForward(J=J, **self.kwargs).to(self.device)
                        self.inv_transform_ptw = DTCWTInverse(**self.kwargs).to(self.device)
                    except Exception: # Fallback to DWT if DTCWT fails
                        self.fwd_transform_ptw = DWTForward(J=J, wave=wavelet_name, mode='symmetric', **self.kwargs).to(self.device)
                        self.inv_transform_ptw = DWTInverse(wave=wavelet_name, mode='symmetric', **self.kwargs).to(self.device)
                else: # Fallback or less common configurations
                    # print(f"SparsityTransform (pytorch_wavelets): Attempting default DWT for shape {self.image_shape}, axes {self.axes}.")
                    self.fwd_transform_ptw = DWTForward(J=J, wave=wavelet_name, mode='symmetric', **self.kwargs).to(self.device)
                    self.inv_transform_ptw = DWTInverse(wave=wavelet_name, mode='symmetric', **self.kwargs).to(self.device)
            
            elif PYWT_AVAILABLE:
                self.use_pywt_backend = True
                # print("SparsityTransform: pytorch_wavelets not found. Using pywt backend.")
                if self.level is None:
                    try: self.level = pywt.dwt_max_level(min_dim_for_level_calc, pywt.Wavelet(self.wavelet_name))
                    except: self.level = 3 # Default
            else:
                print("SparsityTransform: No wavelet backend (pytorch_wavelets or pywt) available. Transform will be identity.")
        else:
            print(f"SparsityTransform: Type '{self.transform_type}' not 'wavelet'. Transform will be identity.")

    def _convert_pywt_coeffs_to_tensor_tuple(self, pywt_coeffs_list):
        tensor_coeffs = []
        for item in pywt_coeffs_list:
            if isinstance(item, np.ndarray):
                tensor_coeffs.append(torch.from_numpy(item).to(self.device))
            elif isinstance(item, tuple): # For pywt.wavedec2 details like (cH, cV, cD)
                for sub_item in item:
                    tensor_coeffs.append(torch.from_numpy(sub_item).to(self.device))
            elif isinstance(item, dict): # For pywt.wavedecn details
                for key in sorted(item.keys()): # Consistent order
                    tensor_coeffs.append(torch.from_numpy(item[key]).to(self.device))
        return tuple(tensor_coeffs)

    def _convert_tensor_tuple_to_pywt_coeffs(self, tensor_coeffs_tuple, pywt_structure_template):
        idx = 0
        reconstructed_list = []
        for item_template in pywt_structure_template:
            if isinstance(item_template, np.ndarray):
                reconstructed_list.append(tensor_coeffs_tuple[idx].cpu().numpy())
                idx += 1
            elif isinstance(item_template, tuple):
                sub_list = []
                for _ in item_template:
                    sub_list.append(tensor_coeffs_tuple[idx].cpu().numpy()); idx += 1
                reconstructed_list.append(tuple(sub_list))
            elif isinstance(item_template, dict):
                sub_dict = {}
                for key in sorted(item_template.keys()):
                    sub_dict[key] = tensor_coeffs_tuple[idx].cpu().numpy(); idx += 1
                reconstructed_list.append(sub_dict)
        if idx != len(tensor_coeffs_tuple):
            raise ValueError("Mismatch in tensor tuple length during pywt coefficient reconstruction.")
        return reconstructed_list
        
    def op(self, image_data_tensor):
        image_data_tensor = torch.as_tensor(image_data_tensor, device=self.device)
        # Ensure tensor matches expected dimensions if possible, or at least has same number of elements
        if image_data_tensor.numel() == np.prod(self.image_shape):
            image_data_tensor = image_data_tensor.reshape(self.image_shape)
        else:
            raise ValueError(f"Input tensor numel {image_data_tensor.numel()} != expected {np.prod(self.image_shape)}")


        if self.use_pytorch_wavelets_backend and self.fwd_transform_ptw:
            reshaped_tensor = image_data_tensor.unsqueeze(0).unsqueeze(0).float() # N, C, ...
            return self.fwd_transform_ptw(reshaped_tensor)
        elif self.use_pywt_backend:
            np_data = image_data_tensor.detach().cpu().numpy()
            # pywt.wavedecn is general, wavedec2 is specific for 2D
            if len(self.axes) == 2 and len(self.image_shape) == 2 : # Strictly 2D data and 2 axes for transform
                 self._pywt_coeffs_structure_template = pywt.wavedec2(np_data, wavelet=self.wavelet_name, level=self.level, axes=self.axes)
            else: # Use wavedecn for >2D or if axes don't match simple 2D case
                 self._pywt_coeffs_structure_template = pywt.wavedecn(np_data, wavelet=self.wavelet_name, level=self.level, axes=self.axes)
            return self._convert_pywt_coeffs_to_tensor_tuple(self._pywt_coeffs_structure_template)
        else: # Identity transform
            return image_data_tensor 

    def op_adj(self, coeffs_input):
        if self.use_pytorch_wavelets_backend and self.inv_transform_ptw:
            reconstructed_tensor = self.inv_transform_ptw(coeffs_input) # input is tuple from op()
            return reconstructed_tensor.squeeze(0).squeeze(0).to(dtype=coeffs_input[0].dtype if isinstance(coeffs_input, tuple) else coeffs_input.dtype, device=self.device)
        elif self.use_pywt_backend:
            if self._pywt_coeffs_structure_template is None:
                raise RuntimeError("SparsityTransform (pywt): op() must be called before op_adj() to set coefficient structure template.")
            pywt_coeffs_list = self._convert_tensor_tuple_to_pywt_coeffs(coeffs_input, self._pywt_coeffs_structure_template)
            
            if len(self.axes) == 2 and len(self.image_shape) == 2:
                reconstructed_np = pywt.waverec2(pywt_coeffs_list, wavelet=self.wavelet_name, axes=self.axes)
            else:
                reconstructed_np = pywt.waverecn(pywt_coeffs_list, wavelet=self.wavelet_name, axes=self.axes)
            # Ensure shape matches original image_shape, remove potential extra dims from waverec
            reconstructed_np = np.reshape(reconstructed_np, self.image_shape)
            return torch.from_numpy(reconstructed_np.astype(np.promote_types(reconstructed_np.dtype, np.float32))).to(self.device)
        else: # Identity transform
            return torch.as_tensor(coeffs_input[0] if isinstance(coeffs_input, tuple) else coeffs_input, device=self.device)


class L1Regularizer(Regularizer):
    """L1 Regularizer, potentially with a sparsity transform."""
    def __init__(self, lambda_param, sparsity_transform=None):
        self.lambda_param = lambda_param
        self.sparsity_transform = sparsity_transform

    def _soft_threshold_tensor(self, x_tensor, threshold_val):
        if threshold_val < 0: raise ValueError("Threshold must be non-negative.")
        if x_tensor.is_complex():
            abs_x = torch.abs(x_tensor)
            # Avoid division by zero for abs_x == 0
            non_zero_mask = abs_x != 0
            scaled_factor = torch.zeros_like(abs_x, dtype=torch.float32) # Ensure float for division
            scaled_factor[non_zero_mask] = threshold_val / abs_x[non_zero_mask]
            return (1 - scaled_factor).clamp(min=0.0) * x_tensor
        else: # Real tensor
            return torch.sign(x_tensor) * torch.maximum(torch.abs(x_tensor) - threshold_val, 
                                                        torch.tensor(0.0, device=x_tensor.device, dtype=x_tensor.dtype))

    def _soft_threshold(self, data_coeffs, threshold_val):
        if isinstance(data_coeffs, (list, tuple)): # For wavelet coefficients (list/tuple of tensors)
            return tuple(self._soft_threshold_tensor(t, threshold_val) for t in data_coeffs)
        elif isinstance(data_coeffs, torch.Tensor):
            return self._soft_threshold_tensor(data_coeffs, threshold_val)
        else:
            raise TypeError(f"Unsupported data type for soft thresholding: {type(data_coeffs)}")

    def prox(self, x_tensor, step_size):
        # x_tensor is assumed to be on the correct device already by caller (e.g. optimizer)
        threshold_val = self.lambda_param * step_size
        
        if self.sparsity_transform:
            # Check if the transform is effectively an identity
            is_identity_transform = not (self.sparsity_transform.use_pytorch_wavelets_backend or self.sparsity_transform.use_pywt_backend)
            
            if is_identity_transform:
                # print("L1Regularizer: Sparsity transform is identity (no backend). Applying L1 in image domain.")
                return self._soft_threshold(x_tensor, threshold_val)

            original_dtype = x_tensor.dtype
            original_device = x_tensor.device
            
            # SparsityTransform.op handles device transfer internally based on its init device
            coeffs = self.sparsity_transform.op(x_tensor) 
            thresholded_coeffs = self._soft_threshold(coeffs, threshold_val)
            reconstructed_image = self.sparsity_transform.op_adj(thresholded_coeffs)
            
            # Ensure final output is on original device and dtype
            return reconstructed_image.to(device=original_device, dtype=original_dtype)
        else: # No sparsity_transform provided, apply L1 in image domain
            return self._soft_threshold(x_tensor, threshold_val)

class L2Regularizer(Regularizer):
    """L2 Regularizer (Tikhonov)."""
    def __init__(self, lambda_param):
        self.lambda_param = lambda_param

    def prox(self, x_tensor, step_size):
        # x_tensor is assumed to be on the correct device
        denominator = 1 + self.lambda_param * step_size
        if denominator == 0: # Avoid division by zero; unlikely for positive lambda, step_size
            return torch.zeros_like(x_tensor)
        return x_tensor / denominator

class TVRegularizer(Regularizer):
    """
    Anisotropic Total Variation (TV) Regularizer.
    Uses Chambolle's projection algorithm to solve the proximal problem.
    """
    def __init__(self, lambda_param, max_chambolle_iter=25, tol_chambolle=1e-5, verbose_chambolle=False):
        self.lambda_param = lambda_param
        self.max_chambolle_iter = max_chambolle_iter
        self.tol_chambolle = tol_chambolle
        self.verbose_chambolle = verbose_chambolle

    def _gradient(self, u_tensor):
        """Computes the discrete gradient using forward differences.
           Boundary: grad_dim[..., N-1] = 0 - u_tensor[..., N-1] for each dim.
        """
        N_dim = u_tensor.ndim
        grads = []
        
        if N_dim == 2: # (H, W)
            zeros_y_boundary = torch.zeros_like(u_tensor[-1:, :], dtype=u_tensor.dtype, device=u_tensor.device)
            grads.append(torch.diff(u_tensor, dim=0, append=zeros_y_boundary))
            zeros_x_boundary = torch.zeros_like(u_tensor[:, -1:], dtype=u_tensor.dtype, device=u_tensor.device)
            grads.append(torch.diff(u_tensor, dim=1, append=zeros_x_boundary))
        elif N_dim == 3: # (D, H, W)
            zeros_z_boundary = torch.zeros_like(u_tensor[-1:, :, :], dtype=u_tensor.dtype, device=u_tensor.device)
            grads.append(torch.diff(u_tensor, dim=0, append=zeros_z_boundary))
            zeros_y_boundary = torch.zeros_like(u_tensor[:, -1:, :], dtype=u_tensor.dtype, device=u_tensor.device)
            grads.append(torch.diff(u_tensor, dim=1, append=zeros_y_boundary))
            zeros_x_boundary = torch.zeros_like(u_tensor[:, :, -1:], dtype=u_tensor.dtype, device=u_tensor.device)
            grads.append(torch.diff(u_tensor, dim=2, append=zeros_x_boundary))
        else: # General N-dim case
            for d in range(N_dim):
                boundary_shape_list = list(u_tensor.shape)
                boundary_shape_list[d] = 1 
                zeros_at_boundary = torch.zeros(boundary_shape_list, dtype=u_tensor.dtype, device=u_tensor.device)
                grads.append(torch.diff(u_tensor, dim=d, append=zeros_at_boundary))
                
        return tuple(grads)

    def _divergence(self, grad_tensor_tuple):
        """Computes the discrete divergence (adjoint of the forward difference gradient used in _gradient).
           div_d = p_d[i] - p_d[i-1], with p_d[-1]=0.
        """
        N_dim = len(grad_tensor_tuple)
        div_sum = torch.zeros_like(grad_tensor_tuple[0], device=grad_tensor_tuple[0].device, dtype=grad_tensor_tuple[0].dtype)

        for d in range(N_dim):
            p_d = grad_tensor_tuple[d] 
            
            rolled_pd = torch.roll(p_d, shifts=1, dims=d)
            
            idx_first_slice = [slice(None)] * p_d.ndim
            idx_first_slice[d] = 0
            rolled_pd[tuple(idx_first_slice)] = 0.0 
            
            div_d_comp = p_d - rolled_pd 
            div_sum = div_sum + div_d_comp
        return div_sum

    def prox(self, x_tensor, step_size):
        if not isinstance(x_tensor, torch.Tensor):
            target_device = x_tensor.device if hasattr(x_tensor, 'device') else 'cpu'
            x_tensor = torch.tensor(x_tensor, device=target_device, dtype=torch.float32) 
        
        device = x_tensor.device
        original_dtype = x_tensor.dtype

        if not x_tensor.is_floating_point() and not x_tensor.is_complex():
            x_tensor_proc = x_tensor.to(torch.float32)
        else:
            x_tensor_proc = x_tensor

        effective_lambda = self.lambda_param * step_size
        if effective_lambda <= 1e-9: 
            return x_tensor.to(original_dtype)

        f_param_chambolle = x_tensor_proc / effective_lambda
        N_dim = x_tensor_proc.ndim
        
        p_old_list = []
        for _ in range(N_dim):
            p_old_list.append(torch.zeros_like(x_tensor_proc, dtype=x_tensor_proc.dtype, device=device)) 
        p_old = tuple(p_old_list)

        sigma_dual = 1.0 / (2.0 * N_dim) 

        for iter_num in range(self.max_chambolle_iter):
            term_in_grad = self._divergence(p_old) - f_param_chambolle
            grad_of_term = self._gradient(term_in_grad)

            p_temp_list = []
            for i in range(N_dim):
                p_temp_list.append(p_old[i] + sigma_dual * grad_of_term[i])
            
            p_new_list = []
            real_dtype_for_one = x_tensor_proc.real.dtype if x_tensor_proc.is_complex() else x_tensor_proc.dtype
            one_tensor = torch.tensor(1.0, device=device, dtype=real_dtype_for_one)
            
            for p_comp_temp in p_temp_list:
                magnitude = torch.abs(p_comp_temp)
                p_new_list.append(p_comp_temp / torch.maximum(one_tensor, magnitude))
            
            p_new = tuple(p_new_list)

            norm_p_old_sum = sum(torch.linalg.norm(p_old[i].flatten()) for i in range(N_dim)) + 1e-9 
            norm_diff_sum = sum(torch.linalg.norm(p_new[i].flatten() - p_old[i].flatten()) for i in range(N_dim))
            diff_criterion = norm_diff_sum / norm_p_old_sum
            
            p_old = p_new 

            if self.verbose_chambolle and (iter_num % 10 == 0 or iter_num == self.max_chambolle_iter -1) :
                print(f"  TVRegularizer Chambolle Iter {iter_num+1}/{self.max_chambolle_iter}, RelChange_p: {diff_criterion:.2e}")

            if diff_criterion < self.tol_chambolle:
                if self.verbose_chambolle:
                    print(f"  TVRegularizer Chambolle converged at iter {iter_num+1}, RelChange_p: {diff_criterion:.2e}")
                break
        
        reconstructed_image = x_tensor_proc - effective_lambda * self._divergence(p_new)
        return reconstructed_image.to(original_dtype)

class GradientMatchingRegularizer(Regularizer):
    """
    Regularizer for penalizing the L2 norm of the difference between the gradient
    of the image and the gradient of a constraint (reference) image.
    Term: 0.5 * lambda_gm * || grad(x) - grad(x_ref) ||_2^2
    This is a quadratic term. It's designed to be incorporated into optimizers 
    that can handle such terms in the x-update step directly (e.g., a modified ADMM),
    rather than via a simple proximal operator for FISTA-like methods.
    """
    def __init__(self, constraint_image_tensor, lambda_gm, device='cpu'):
        self.lambda_gm = lambda_gm
        self.device = torch.device(device)
        
        # Constraint image is typically real-valued for gradient matching.
        # Convert to float32 for consistent gradient computation.
        self.constraint_image_tensor = torch.as_tensor(
            constraint_image_tensor, 
            device=self.device, 
            dtype=torch.float32 
        )
        
        # Precompute gradient of the constraint image
        self.grad_constraint_image_tensor_tuple = self._gradient(self.constraint_image_tensor)

    def _gradient(self, u_tensor):
        """Computes the discrete gradient using forward differences.
           (Adapted from TVRegularizer for self-containment)
        """
        N_dim = u_tensor.ndim
        grads = []
        
        # Ensure u_tensor is on the correct device and float for gradient computation
        u_tensor_proc = torch.as_tensor(u_tensor, device=self.device)
        if not u_tensor_proc.is_floating_point():
            u_tensor_proc = u_tensor_proc.to(torch.float32)

        if N_dim == 2: # (H, W)
            zeros_y_boundary = torch.zeros_like(u_tensor_proc[-1:, :], dtype=u_tensor_proc.dtype, device=self.device)
            grads.append(torch.diff(u_tensor_proc, dim=0, append=zeros_y_boundary))
            zeros_x_boundary = torch.zeros_like(u_tensor_proc[:, -1:], dtype=u_tensor_proc.dtype, device=self.device)
            grads.append(torch.diff(u_tensor_proc, dim=1, append=zeros_x_boundary))
        elif N_dim == 3: # (D, H, W)
            zeros_z_boundary = torch.zeros_like(u_tensor_proc[-1:, :, :], dtype=u_tensor_proc.dtype, device=self.device)
            grads.append(torch.diff(u_tensor_proc, dim=0, append=zeros_z_boundary))
            zeros_y_boundary = torch.zeros_like(u_tensor_proc[:, -1:, :], dtype=u_tensor_proc.dtype, device=self.device)
            grads.append(torch.diff(u_tensor_proc, dim=1, append=zeros_y_boundary))
            zeros_x_boundary = torch.zeros_like(u_tensor_proc[:, :, -1:], dtype=u_tensor_proc.dtype, device=self.device)
            grads.append(torch.diff(u_tensor_proc, dim=2, append=zeros_x_boundary))
        else: 
            for d in range(N_dim):
                boundary_shape_list = list(u_tensor_proc.shape)
                boundary_shape_list[d] = 1 
                zeros_at_boundary = torch.zeros(boundary_shape_list, dtype=u_tensor_proc.dtype, device=self.device)
                grads.append(torch.diff(u_tensor_proc, dim=d, append=zeros_at_boundary))
        return tuple(grads)

    def _divergence(self, grad_tensor_tuple):
        """Computes the discrete divergence (adjoint of the forward difference gradient).
           (Adapted from TVRegularizer for self-containment)
        """
        N_dim = len(grad_tensor_tuple)
        # Ensure first component is on the correct device for zeros_like
        first_grad_comp = torch.as_tensor(grad_tensor_tuple[0], device=self.device)
        div_sum = torch.zeros_like(first_grad_comp, device=self.device, dtype=first_grad_comp.dtype)

        for d in range(N_dim):
            p_d = torch.as_tensor(grad_tensor_tuple[d], device=self.device, dtype=first_grad_comp.dtype)
            rolled_pd = torch.roll(p_d, shifts=1, dims=d)
            
            idx_first_slice = [slice(None)] * p_d.ndim
            idx_first_slice[d] = 0
            rolled_pd[tuple(idx_first_slice)] = 0.0 
            
            div_d_comp = p_d - rolled_pd 
            div_sum = div_sum + div_d_comp
        return div_sum

    def get_rhs_term(self):
        """
        Calculates lambda_gm * div(grad(x_ref)). This term is added to the RHS
        in the x-update of a modified ADMM optimizer.
        Term: lambda_gm * G^T (G x_ref)
        """
        # self.grad_constraint_image_tensor_tuple is grad(x_ref)
        return self.lambda_gm * self._divergence(self.grad_constraint_image_tensor_tuple)

    def get_lhs_operator_product(self, v_tensor):
        """
        Calculates lambda_gm * div(grad(v_tensor)). This term is added to the LHS
        operator in the x-update of a modified ADMM optimizer.
        Term: lambda_gm * G^T G v
        """
        # Ensure v_tensor is processed on the correct device and dtype
        # The gradient and divergence methods handle this internally if v_tensor isn't already.
        # For consistency with constraint_image_tensor, ensure float32 for v_tensor processing
        v_tensor_proc = torch.as_tensor(v_tensor, device=self.device, dtype=torch.float32)
        grad_v_tuple = self._gradient(v_tensor_proc)
        return self.lambda_gm * self._divergence(grad_v_tuple)

    def prox(self, x_tensor, step_size):
        """This regularizer is quadratic and is handled by modifying the x-update step
           in ADMM directly, rather than through a proximal operator.
        """
        raise NotImplementedError(
            "GradientMatchingRegularizer does not have a simple proximal operator. "
            "It should be incorporated directly into the x-update of an ADMM-like optimizer."
        )
